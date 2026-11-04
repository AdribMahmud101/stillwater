"""Hardening for StillWater public production use.

Security headers, per-IP rate limiting, favicon 204, concurrency cap.
Never log request bodies or Authorization headers — this module performs
no logging of request content and intentionally ignores Authorization.
Pure stdlib + starlette/fastapi only.
"""
import os
import time
import threading

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from fastapi import FastAPI

# CSP must allow self, unpkg.com (Leaflet js/css), tile.openstreetmap.org (tiles),
# data: and blob: for media/img. We keep it strict otherwise.
def _inline_script_hashes():
    """sha256 source hashes of inline <script> blocks in the bundled UI, so inline JS runs without 'unsafe-inline'."""
    import base64, hashlib, re
    from pathlib import Path
    try:
        html = (Path(__file__).parent / "web" / "index.html").read_text(encoding="utf-8")
    except OSError:
        return ""
    out = []
    for body in re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, re.S):
        out.append("'sha256-" + base64.b64encode(hashlib.sha256(body.encode("utf-8")).digest()).decode() + "'")
    return " ".join(out)


CSP_VALUE = (
    "default-src 'self'; "
    "script-src 'self' https://unpkg.com " + _inline_script_hashes() + "; "
    "style-src 'self' https://unpkg.com 'unsafe-inline'; "
    "img-src 'self' https://tile.openstreetmap.org data: blob:; "
    "media-src 'self' data: blob:; "
    "connect-src 'self'; "
    "font-src 'self' https://unpkg.com data:; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "frame-ancestors 'none'"
)


def _parse_int_env(name: str, default: int) -> int:
    try:
        v = os.environ.get(name)
        if v is None or v == "":
            return default
        iv = int(v)
        return iv
    except Exception:
        return default


def _trust_proxy() -> bool:
    return os.environ.get("TRUST_PROXY", "0") == "1"


def _get_client_ip(request: Request) -> str:
    # Trust X-Forwarded-For only if TRUST_PROXY=1
    if _trust_proxy():
        # starlette headers are case-insensitive
        xff = request.headers.get("x-forwarded-for")
        if xff:
            # client is first entry
            first = xff.split(",")[0].strip()
            if first:
                return first
    if request.client:
        return request.client.host
    return "unknown"


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        response = await call_next(request)
        # Add security headers on every response
        response.headers["Content-Security-Policy"] = CSP_VALUE
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-Frame-Options"] = "DENY"
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """In-memory per-IP rate limiting.

    POST /api/scans: default 5 per hour, env RATE_LIMIT_SCANS_PER_HOUR
    Other /api routes: looser, default 60 per minute
    Returns 429 with Retry-After.
    """
    def __init__(self, app):
        super().__init__(app)
        self.scans_store: dict[str, list[float]] = {}
        self.other_store: dict[str, list[float]] = {}
        self.lock = threading.Lock()
        # fixed looser limit for other /api routes
        self.other_limit = _parse_int_env("RATE_LIMIT_API_PER_MIN", 60)
        self.other_window = 60.0
        self.scans_window = 3600.0

    def _get_scans_limit(self) -> int:
        return _parse_int_env("RATE_LIMIT_SCANS_PER_HOUR", 5)

    def _check(self, store: dict[str, list[float]], ip: str, limit: int, window: float):
        if limit <= 0:  # 0 or negative disables this limit
            return False, 0
        now = time.monotonic()
        with self.lock:
            lst = store.get(ip)
            if lst is None:
                lst = []
                store[ip] = lst
            cutoff = now - window
            # prune old entries
            # modify in place to keep same list object
            pruned = [t for t in lst if t > cutoff]
            if len(pruned) != len(lst):
                lst[:] = pruned
            if len(lst) >= limit:
                oldest = min(lst) if lst else now
                retry = int(oldest + window - now) + 1
                if retry < 1:
                    retry = 1
                return True, retry
            else:
                lst.append(now)
                return False, 0

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        path = request.url.path
        # only rate limit /api routes; favicon, index, etc. are excluded
        if path.startswith("/api"):
            is_scan_post = request.method == "POST" and path.rstrip("/") == "/api/scans"
            ip = _get_client_ip(request)
            if is_scan_post:
                limit = self._get_scans_limit()  # <= 0 disables
                limited, retry = self._check(self.scans_store, ip, limit, self.scans_window)
                if limited:
                    # Security headers will be added by outer middleware, but add here as fallback
                    headers = {"Retry-After": str(retry)}
                    resp = JSONResponse(status_code=429, content={"detail": "Too Many Requests: rate limit exceeded"}, headers=headers)
                    # ensure headers even if SecurityHeadersMiddleware not outermost due to ordering
                    resp.headers["Content-Security-Policy"] = CSP_VALUE
                    resp.headers["X-Content-Type-Options"] = "nosniff"
                    resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
                    resp.headers["X-Frame-Options"] = "DENY"
                    return resp
            else:
                limited, retry = self._check(self.other_store, ip, self.other_limit, self.other_window)
                if limited:
                    headers = {"Retry-After": str(retry)}
                    resp = JSONResponse(status_code=429, content={"detail": "Too Many Requests: rate limit exceeded"}, headers=headers)
                    resp.headers["Content-Security-Policy"] = CSP_VALUE
                    resp.headers["X-Content-Type-Options"] = "nosniff"
                    resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
                    resp.headers["X-Frame-Options"] = "DENY"
                    return resp
        return await call_next(request)


class ConcurrencyMiddleware(BaseHTTPMiddleware):
    """Global concurrency cap for POST /api/scans.

    Default 1, env MAX_CONCURRENT_SCANS. Returns 429 with Retry-After and
    clear message instead of queueing. Never queues.
    """
    def __init__(self, app, max_concurrent: int | None = None):
        super().__init__(app)
        if max_concurrent is None:
            max_concurrent = _parse_int_env("MAX_CONCURRENT_SCANS", 1)
        if max_concurrent < 1:
            max_concurrent = 1
        self.max_concurrent = max_concurrent
        self.semaphore = threading.Semaphore(max_concurrent)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        if request.method == "POST" and request.url.path.rstrip("/") == "/api/scans":
            acquired = self.semaphore.acquire(blocking=False)
            if not acquired:
                headers = {"Retry-After": "5"}
                resp = JSONResponse(
                    status_code=429,
                    content={"detail": "Too many concurrent scans, please try again later."},
                    headers=headers,
                )
                resp.headers["Content-Security-Policy"] = CSP_VALUE
                resp.headers["X-Content-Type-Options"] = "nosniff"
                resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
                resp.headers["X-Frame-Options"] = "DENY"
                return resp
            try:
                response = await call_next(request)
                return response
            finally:
                self.semaphore.release()
        return await call_next(request)


def setup_hardening(app: FastAPI) -> None:
    """Wire hardening into a FastAPI app with minimal edits.

    Adds /favicon.ico 204 route and middleware for security headers,
    rate limiting, and concurrency. Call once after `app = FastAPI(...)`.
    """
    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        # 204 No Content, with security headers added by middleware
        return Response(status_code=204)

    # Read env at setup time for concurrency cap (tests set env before create_app)
    max_concurrent = _parse_int_env("MAX_CONCURRENT_SCANS", 1)
    if max_concurrent < 1:
        max_concurrent = 1

    # Order: last added is outermost. SecurityHeaders must be outermost
    # so it decorates 429/503 responses from inner middlewares.
    app.add_middleware(ConcurrencyMiddleware, max_concurrent=max_concurrent)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
