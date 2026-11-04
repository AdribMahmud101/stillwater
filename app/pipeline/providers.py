"""Wire real pipeline stages and metered clients for production runner.

Privacy disclosure: the privacy stage sends every decoded raw frame (JPEG)
to the hosted Meta SAM detector via ``MetaSAMClient.segment_image`` and
blurs only after all per-frame detections complete. Transmitted frames leave
this machine, and detector recall is not guaranteed -- a zero mask means the
provider reported no person, not that no person was present.

Cost discipline: scan-level cost is a measured total accumulated in a
per-scan in-memory :class:`ScanMeter`. The ledger is reserved once per scan
(by ``run_budgeted`` for ``SCAN_MAX_USD``) and settled once with the meter
total. Inner provider calls never reserve on the ledger.

Multi-frame witnessing: verify delegates to decide_track, which describes up
to 3 crops (best + alt_crops) per witness in parallel. Each describe call
goes through the same ScanMeter-metered chat clients, so SCAN_MAX_USD
admission applies to every extra witness call.

Tracking cost: the video track stage sends only a low-cost ffmpeg proxy of
the privacy-processed video to hosted SAM (see
``app.pipeline.sam_track.track_video_proxy``) and maps results back to
full-resolution coordinates. Set ``TRACK_PROXY=0`` for full-quality
tracking.
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import math
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation
from pathlib import Path

from app.clients.cache import CallCache
from app.clients.costs import BudgetExceeded, CostLedger
from app.clients.meta import MetaSAMClient
from app.pipeline.frame_privacy import blur_per_frame, blur_image
from app.pipeline.ingest import ingest_video, ingest_image, probe_video, probe_image
from app.pipeline.keyframes import extract_keyframes
from app.pipeline.sam_track import track_video, track_image, track_video_proxy, proxy_enabled

SCAN_MAX_DEFAULT = Decimal("1.00")

# Glimmer (Modal, L4) metering defaults.
# GLIMMER_GPU_PER_SECOND is an estimate of list price (L4 $0.80/hr / 3600),
# not a measured billed rate.
GLIMMER_MAX_CALL_DEFAULT = Decimal("0.05")
GLIMMER_GPU_PER_SECOND_DEFAULT = Decimal("0.000222")
MODAL_SERVICE_IDENTITY = "stillwater-glimmer:GlimmerService"
MODAL_APP_NAME = "stillwater-glimmer"
MODAL_CLASS_NAME = "GlimmerService"

# Meta Spark (hosted Responses API) metering defaults.
# META_SPARK_INPUT_PER_M / META_SPARK_OUTPUT_PER_M are unverified
# placeholders (USD per million tokens) for the unpublished Meta API price.
META_SPARK_INPUT_PER_M_DEFAULT = Decimal("1.0")
META_SPARK_OUTPUT_PER_M_DEFAULT = Decimal("3.0")
META_SPARK_MODEL_DEFAULT = "muse-spark-1.3"
META_SPARK_ENDPOINT = "https://api.meta.ai/v1/responses"
META_SPARK_TIMEOUT = 120
META_SPARK_MIN_MAX_TOKENS = 3000

import contextvars as _cv
_SCAN_METER = _cv.ContextVar('stillwater_scan_meter', default=None)
_PROGRESS = _cv.ContextVar('stillwater_progress', default=None)

def set_current_progress(callback):
    _PROGRESS.set(callback)

def get_current_progress():
    return _PROGRESS.get()

def clear_current_progress():
    _PROGRESS.set(None)

def _report_progress(stage, done, total, unit):
    callback = get_current_progress()
    if callback is not None:
        try:
            callback(stage, done, total, unit)
        except Exception:
            import logging
            logging.getLogger(__name__).exception('Progress callback failed')
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp'}
VIDEO_SUFFIXES = {'.mp4', '.mov', '.webm'}


def _env_get(env, *names, default=None):
    if isinstance(env, dict):
        for n in names:
            if n in env and env[n] is not None:
                return env[n]
        return default
    for n in names:
        v = getattr(env, n, None)
        if v is not None:
            return v
    try:
        get = getattr(env, "get", None)
        if callable(get):
            for n in names:
                v = get(n)
                if v is not None:
                    return v
    except Exception:
        pass
    return default


def _parse_budget(value) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value).strip())
    except (InvalidOperation, ValueError, AttributeError):
        return None


def _daily_budget_from_env(env) -> Decimal | None:
    raw = _env_get(
        env,
        "DAILY_BUDGET_USD",
        "daily_budget_usd",
        "daily_budget",
        "budget",
        "BUDGET",
        default=None,
    )
    return _parse_budget(raw)


def _scan_max_from_env(env, daily_budget: Decimal) -> Decimal:
    raw = _env_get(env, "SCAN_MAX_USD", "scan_max_usd", "SCAN_MAX", default=None)
    if raw is None:
        requested = SCAN_MAX_DEFAULT
    else:
        parsed = _parse_budget(raw)
        if parsed is None or not parsed.is_finite() or parsed <= 0:
            raise ValueError("SCAN_MAX_USD must be a positive finite Decimal")
        requested = parsed
    if not isinstance(daily_budget, Decimal) or not daily_budget.is_finite() or daily_budget <= 0:
        raise ValueError("daily budget must be a positive finite Decimal")
    # Must be <= daily budget: clamp down to daily budget.
    if requested > daily_budget:
        return daily_budget
    return requested


def _verify_workers_from_env() -> int:
    raw = os.environ.get("VERIFY_WORKERS", "4")
    try:
        n = int(str(raw).strip())
    except Exception:
        n = 4
    if n < 1:
        return 1
    if n > 32:
        return 32
    return n


def glimmer_warm_timeout() -> float:
    raw = os.environ.get("GLIMMER_WARM_TIMEOUT", "600")
    try:
        v = float(str(raw).strip())
    except Exception:
        return 600.0
    if not math.isfinite(v) or v <= 0:
        return 600.0
    return v


def glimmer_call_timeout() -> float:
    raw = os.environ.get("GLIMMER_CALL_TIMEOUT", "180")
    try:
        v = float(str(raw).strip())
    except Exception:
        return 180.0
    if not math.isfinite(v) or v <= 0:
        return 180.0
    return v


class ScanMeter:
    """In-memory per-scan meter. Thread-safe via a lock. Never touches ledger."""

    def __init__(self, limit) -> None:
        if not isinstance(limit, Decimal):
            try:
                limit = Decimal(str(limit))
            except Exception as e:
                raise ValueError("limit must be a Decimal") from e
        if not limit.is_finite() or limit <= 0:
            raise ValueError("ScanMeter limit must be a positive finite Decimal")
        self.limit = limit
        self._total = Decimal("0")
        self._reserved = Decimal("0")
        self.count = 0
        self._lock = threading.Lock()

    def _coerce_nonneg(self, v, name: str) -> Decimal:
        if not isinstance(v, Decimal):
            try:
                v = Decimal(str(v))
            except Exception as e:
                raise ValueError(f"{name} must be a Decimal") from e
        if not v.is_finite() or v < 0:
            raise ValueError(f"{name} must be a finite nonnegative Decimal")
        return v

    def check(self, upper) -> None:
        upper = self._coerce_nonneg(upper, "upper bound")
        with self._lock:
            if self._total + self._reserved + upper > self.limit:
                raise BudgetExceeded("daily budget exceeded")

    def add(self, actual) -> Decimal:
        actual = self._coerce_nonneg(actual, "actual cost")
        with self._lock:
            self._total += actual
            self.count += 1
            return self._total

    def reserve(self, upper):
        """Atomically reserve an upper bound; raises BudgetExceeded if cap exceeded."""
        upper = self._coerce_nonneg(upper, "upper bound")
        with self._lock:
            if self._total + self._reserved + upper > self.limit:
                raise BudgetExceeded("daily budget exceeded")
            self._reserved += upper
            return upper

    def release(self, reserved) -> None:
        try:
            r = self._coerce_nonneg(reserved, "reserved")
        except Exception:
            return
        with self._lock:
            self._reserved -= r
            if self._reserved < 0:
                self._reserved = Decimal("0")

    def commit(self, reserved, actual) -> Decimal:
        """Convert a prior reservation into measured cost atomically."""
        actual = self._coerce_nonneg(actual, "actual cost")
        try:
            r = self._coerce_nonneg(reserved, "reserved")
        except Exception:
            r = Decimal("0")
        with self._lock:
            self._reserved -= r
            if self._reserved < 0:
                self._reserved = Decimal("0")
            self._total += actual
            self.count += 1
            return self._total

    def total(self) -> Decimal:
        with self._lock:
            return self._total

    @property
    def total_usd(self) -> Decimal:
        return self.total()

    def __str__(self):
        return str(self.total())

    def __repr__(self):
        return f"ScanMeter(total={self.total()!s}, limit={self.limit!s})"


def get_current_scan_meter():
    return _SCAN_METER.get()


def set_current_scan_meter(meter):
    _SCAN_METER.set(meter)


def clear_current_scan_meter():
    try:
        _SCAN_METER.set(None)
    except Exception:
        pass


class MeasuredCost:
    """Ledger-backed measured total; str() resolves at read time."""

    def __init__(self, ledger):
        self._ledger = ledger

    def total(self) -> Decimal:
        try:
            import sqlite3

            path = getattr(self._ledger, "path", None)
            if path is None:
                return Decimal("0")
            with sqlite3.connect(Path(path)) as db:
                rows = db.execute("SELECT usd FROM costs").fetchall()
            total = Decimal("0")
            for (v,) in rows:
                try:
                    total += Decimal(str(v))
                except (InvalidOperation, ValueError):
                    continue
            return total
        except Exception:
            return Decimal("0")

    def __str__(self):
        return str(self.total())

    def __repr__(self):
        return f"MeasuredCost({self.total()!s})"


def _try_import(path, name):
    try:
        mod = __import__(path, fromlist=[name])
        return getattr(mod, name, None)
    except Exception:
        return None


def _parse_per_frame(env) -> Decimal:
    raw = _env_get(env, "SAM_PER_FRAME", default="0.001")
    try:
        val = Decimal(str(raw).strip())
    except (InvalidOperation, ValueError, AttributeError):
        val = Decimal("0.001")
    if not val.is_finite() or val < 0:
        raise ValueError("SAM_PER_FRAME must be a finite, nonnegative Decimal")
    return val


def _parse_rate(env, name: str, default: str) -> Decimal:
    raw = _env_get(env, name, default=default)
    try:
        val = Decimal(str(raw).strip())
    except (InvalidOperation, ValueError, AttributeError):
        val = Decimal(default)
    if not val.is_finite() or val < 0:
        raise ValueError(f"{name} must be a finite, nonnegative Decimal")
    return val


def _parse_glimmer_max_call(env) -> Decimal:
    return _parse_rate(env, "GLIMMER_MAX_CALL_USD", str(GLIMMER_MAX_CALL_DEFAULT))


def _parse_glimmer_gpu_per_second(env) -> Decimal:
    # Estimate of list price (L4 list price $0.80/hr / 3600s); not billed cost.
    return _parse_rate(env, "GLIMMER_GPU_PER_SECOND", str(GLIMMER_GPU_PER_SECOND_DEFAULT))


def _warm_callable_for_client(client):
    if client is None:
        return None
    try:
        fn = getattr(client, "warm", None)
    except Exception:
        return None
    if callable(fn):
        return fn
    # Unwrap metered wrappers to find the raw warm().
    try:
        raw = getattr(client, "_raw", None)
        if raw is not None:
            fn2 = getattr(raw, "warm", None)
            if callable(fn2):
                return fn2
    except Exception:
        pass
    return None


def start_glimmer_warmup(warm_fn=None, client=None):
    """Start Glimmer warm-up in a background daemon thread; return a Future.

    Never raises and never blocks. The future resolves to True (warmed) or
    False (failed). Consumes no scan budget.
    """
    fn = warm_fn
    if fn is None and client is not None:
        fn = _warm_callable_for_client(client)
    fut: concurrent.futures.Future = concurrent.futures.Future()
    if fn is None:
        # No warm callable: try the Modal warm endpoint directly in background.
        def _modal_run():
            try:
                import modal  # type: ignore
                handle = modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)()
                warm_m = getattr(handle, "warm", None)
                if warm_m is None:
                    warm_m = getattr(handle, "ping", None)
                remote = getattr(warm_m, "remote", None) if warm_m is not None else None
                if not callable(remote):
                    if not fut.done():
                        fut.set_result(False)
                    return
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    inner = ex.submit(remote)
                    try:
                        inner.result(timeout=glimmer_warm_timeout())
                        if not fut.done():
                            fut.set_result(True)
                    except BaseException:
                        try:
                            inner.cancel()
                        except Exception:
                            pass
                        if not fut.done():
                            fut.set_result(False)
            except Exception:
                if not fut.done():
                    fut.set_result(False)
        threading.Thread(target=_modal_run, name="glimmer-warm", daemon=True).start()
        return fut
    def _run():
        try:
            try:
                res = fn()
            except TypeError:
                try:
                    res = fn(30)
                except Exception:
                    if not fut.done():
                        fut.set_result(False)
                    return
            if isinstance(res, concurrent.futures.Future):
                try:
                    res = res.result(timeout=glimmer_warm_timeout())
                except BaseException:
                    if not fut.done():
                        fut.set_result(False)
                    return
            if not fut.done():
                fut.set_result(bool(res))
        except Exception:
            if not fut.done():
                fut.set_result(False)
    threading.Thread(target=_run, name="glimmer-warm", daemon=True).start()
    return fut


def await_glimmer_warmup(fut):
    """Await a warm-up future with a bounded timeout; never raises.

    Returns 'ok' | 'failed' | 'timeout'.
    """
    if fut is None:
        return 'failed'
    try:
        res = fut.result(timeout=glimmer_warm_timeout())
        return 'ok' if res else 'failed'
    except concurrent.futures.TimeoutError:
        return 'timeout'
    except Exception:
        return 'failed'


def prewarm_glimmer():
    """Fire-and-forget: start the Glimmer GPU container while earlier stages run.

    Never raises and never blocks the scan; a failure only means verify waits for a cold start.
    Returns the background Future (ignored by legacy callers).
    """
    try:
        fut = start_glimmer_warmup()
        # Also best-effort legacy ping for older deployments; never raises.
        def _legacy_ping():
            try:
                import modal  # type: ignore
                modal.Cls.from_name(MODAL_APP_NAME, MODAL_CLASS_NAME)().ping.remote()
            except Exception as exc:  # noqa: BLE001
                import logging
                logging.getLogger("stillwater.prewarm").warning("glimmer prewarm failed: %s", type(exc).__name__)
        # Only run legacy ping if warm future already failed fast? Keep it
        # simple: the warm thread above already covers it; log on failure via callback.
        def _log(f):
            try:
                ok = f.result()
            except Exception:
                ok = False
            if not ok:
                logging.getLogger("stillwater.prewarm").warning("glimmer prewarm failed: warm returned False")
        try:
            fut.add_done_callback(_log)
        except Exception:
            pass
        return fut
    except Exception as exc:  # noqa: BLE001
        import logging
        logging.getLogger("stillwater.prewarm").warning("glimmer prewarm failed: %s", type(exc).__name__)
        fut: concurrent.futures.Future = concurrent.futures.Future()
        try:
            fut.set_result(False)
        except Exception:
            pass
        return fut


def _modal_credentials_available(env) -> bool:
    tid = _env_get(env, "MODAL_TOKEN_ID", default=None)
    tsec = _env_get(env, "MODAL_TOKEN_SECRET", default=None)
    if tid is not None and str(tid).strip() and tsec is not None and str(tsec).strip():
        return True
    try:
        if os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"):
            return True
    except Exception:
        pass
    try:
        if (Path.home() / ".modal.toml").is_file():
            return True
    except Exception:
        pass
    return False


def _build_modal_remote_chat():
    """Build a Modal remote_chat callable for stillwater-glimmer:GlimmerService.

    Fail closed with a clear error if Modal is not reachable.
    """
    try:
        import modal  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "witness_b (Glimmer) is not reachable: Modal package is unavailable"
        ) from e
    try:
        cls = getattr(getattr(modal, "Cls"), "from_name", None)
        if not callable(cls):
            raise RuntimeError("Modal Cls.from_name is unavailable")
        handle = cls(MODAL_APP_NAME, MODAL_CLASS_NAME)
        # A Modal Cls handle must be instantiated before its methods exist.
        instance = handle() if callable(handle) else handle
        remote_method = getattr(instance, "chat", None)
        if remote_method is None:
            raise RuntimeError("GlimmerService.chat is unavailable")
        remote_fn = getattr(remote_method, "remote", None)
        if not callable(remote_fn):
            raise RuntimeError("GlimmerService.chat.remote is unavailable")

        def remote_chat(payload):
            return remote_fn(payload)

        # Attach warm for ModalGlimmerClient.warm() discovery.
        try:
            warm_method = getattr(instance, "warm", None)
            warm_remote = getattr(warm_method, "remote", None) if warm_method is not None else None
            if callable(warm_remote):
                remote_chat.warm = warm_remote  # type: ignore[attr-defined]
            else:
                ping_method = getattr(instance, "ping", None)
                ping_remote = getattr(ping_method, "remote", None) if ping_method is not None else None
                if callable(ping_remote):
                    remote_chat.warm = ping_remote  # type: ignore[attr-defined]
        except Exception:
            pass
        return remote_chat
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(
            "witness_b (Glimmer) is not reachable: Modal "
            "stillwater-glimmer:GlimmerService lookup failed"
        ) from e


def _extract_inference_seconds(response, elapsed: float):
    """Return measured GPU seconds from the client envelope or timed elapsed."""
    try:
        meta = response.get("_stillwater") if isinstance(response, dict) else None
        cand = None
        if isinstance(meta, dict) and "inference_seconds" in meta:
            cand = meta["inference_seconds"]
        elif isinstance(response, dict) and "inference_seconds" in response:
            cand = response["inference_seconds"]
        if isinstance(cand, bool):
            raise ValueError("bad seconds")
        if isinstance(cand, (int, float, Decimal)):
            val = float(cand)
            import math

            if math.isfinite(val) and val >= 0:
                return Decimal(str(cand))
    except Exception:
        pass
    try:
        el = float(elapsed)
        if el >= 0 and el < 10**6:
            return Decimal(str(el))
    except Exception:
        pass
    return Decimal("0")


# ---------------------------------------------------------------------------
# Meta Spark second witness (hosted Responses API).
# ---------------------------------------------------------------------------

def _meta_retry_delay(attempt: int) -> float:
    # Small exponential backoff for 429/5xx/timeouts; reused pattern for
    # hosted-provider retries. Kept tiny so live-demo latency stays ~6 s.
    try:
        return min(4.0, 0.25 * (2.0 ** max(0, attempt)))
    except Exception:
        return 0.25


def _meta_status_retryable(status) -> bool:
    try:
        code = int(status)
    except Exception:
        return False
    return code == 429 or 500 <= code <= 599


class MetaSparkChat:
    """Hosted Meta Muse Spark chat client with chat-completions interface.

    Accepts ``chat(messages, **options)`` where ``messages`` use
    chat-completions style parts (``{'type': 'text', 'text': ...}`` and
    ``{'type': 'image_url', 'image_url': {'url': data-url}}``) and returns
    ``{'choices': [{'message': {'content': text}}], 'usage': {...}}`` so
    :func:`app.pipeline.witness.describe_container` can use it directly.

    Translation: system message -> ``instructions``; user text ->
    ``input_text``; ``image_url`` -> ``input_image``.
    """

    ENDPOINT = META_SPARK_ENDPOINT

    def __init__(self, api_key, *, model: str = META_SPARK_MODEL_DEFAULT,
                 opener=None, timeout: int = META_SPARK_TIMEOUT,
                 max_output_tokens: int = META_SPARK_MIN_MAX_TOKENS,
                 max_retries: int = 3):
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("MODEL_API_KEY is required for Meta Spark")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model is required")
        self._key = api_key
        self.model = model.strip()
        self._open = opener or urllib.request.urlopen
        try:
            self._timeout = int(timeout)
        except Exception:
            self._timeout = META_SPARK_TIMEOUT
        if self._timeout <= 0:
            self._timeout = META_SPARK_TIMEOUT
        try:
            self._max_output_tokens = int(max_output_tokens)
        except Exception:
            self._max_output_tokens = META_SPARK_MIN_MAX_TOKENS
        if self._max_output_tokens < META_SPARK_MIN_MAX_TOKENS:
            self._max_output_tokens = META_SPARK_MIN_MAX_TOKENS
        try:
            self._max_retries = int(max_retries)
        except Exception:
            self._max_retries = 3
        if self._max_retries < 0:
            self._max_retries = 0
        self._log = logging.getLogger("stillwater.meta_spark")

    @staticmethod
    def _text_of(content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for p in content:
                if not isinstance(p, dict):
                    continue
                t = p.get("type")
                if t == "text" and isinstance(p.get("text"), str):
                    parts.append(p["text"])
                elif t == "input_text" and isinstance(p.get("text"), str):
                    parts.append(p["text"])
            return "".join(parts)
        return ""

    def _translate(self, messages):
        instructions = ""
        inputs = []
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                txt = self._text_of(content)
                instructions = (instructions + "\n" + txt).strip() if instructions else txt.strip()
                continue
            # user / assistant / tool entries become Responses input items.
            if isinstance(content, str):
                if not content:
                    continue
                inputs.append({"role": role, "content": [{"type": "input_text", "text": content}]})
            elif isinstance(content, list):
                conv = []
                for p in content:
                    if not isinstance(p, dict):
                        continue
                    t = p.get("type")
                    if t in ("text", "input_text") and isinstance(p.get("text"), str):
                        conv.append({"type": "input_text", "text": p["text"]})
                    elif t == "image_url":
                        url = p.get("image_url")
                        if isinstance(url, dict):
                            url = url.get("url")
                        if isinstance(url, str) and url.startswith("data:"):
                            conv.append({"type": "input_image", "image_url": url})
                        elif isinstance(url, str) and url:
                            # Only data URLs are sent; remote URLs are not fetched.
                            raise ValueError("only data-url images are supported")
                if conv:
                    inputs.append({"role": role, "content": conv})
        return instructions, inputs

    def _payload(self, messages, options):
        instructions, inputs = self._translate(messages)
        requested = options.get("max_tokens", options.get("max_output_tokens", self._max_output_tokens))
        try:
            requested = int(requested)
        except Exception:
            requested = self._max_output_tokens
        if requested < META_SPARK_MIN_MAX_TOKENS:
            requested = META_SPARK_MIN_MAX_TOKENS
        payload = {
            "model": self.model,
            "max_output_tokens": requested,
            "reasoning": {"effort": "low"},
            "input": inputs,
        }
        if instructions:
            payload["instructions"] = instructions
        return payload

    @staticmethod
    def _extract_text(body) -> str:
        texts = []
        try:
            for item in body.get("output", []) or []:
                if not isinstance(item, dict) or item.get("type") != "message":
                    continue
                for c in item.get("content", []) or []:
                    if isinstance(c, dict) and c.get("type") == "output_text" and isinstance(c.get("text"), str):
                        texts.append(c["text"])
            return "".join(texts)
        except Exception:
            pass
        return "".join(texts)

    def _post_once(self, payload: dict):
        req = urllib.request.Request(
            self.ENDPOINT,
            data=json.dumps(payload).encode(),
            headers={"Authorization": "Bearer " + self._key, "Content-Type": "application/json"},
            method="POST",
        )
        with self._open(req, timeout=self._timeout) as resp:
            raw = resp.read()
        try:
            body = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        except Exception as e:
            raise RuntimeError("meta witness: invalid response encoding") from e
        if not isinstance(body, dict):
            raise RuntimeError("meta witness: invalid response")
        status = body.get("status")
        # Reasoning tokens consume the budget; incomplete means the JSON was
        # cut off and must not be trusted.
        if status != "completed":
            raise RuntimeError(f"meta witness: response status {status!r} is not completed")
        text = self._extract_text(body)
        if not isinstance(text, str) or not text:
            raise RuntimeError("meta witness: empty response text")
        usage = body.get("usage") or {}
        try:
            in_tok = int(usage.get("input_tokens", 0) or 0)
            out_tok = int(usage.get("output_tokens", 0) or 0)
        except Exception:
            in_tok, out_tok = 0, 0
        if in_tok < 0:
            in_tok = 0
        if out_tok < 0:
            out_tok = 0
        return {"choices": [{"message": {"content": text}}],
                "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok,
                          "input_tokens": in_tok, "output_tokens": out_tok}}

    def chat(self, messages, **options):
        payload = self._payload(messages, options or {})
        last_err: Exception | None = None
        attempts = max(1, self._max_retries + 1)
        for attempt in range(attempts):
            try:
                return self._post_once(payload)
            except RuntimeError as e:
                msg = str(e)
                # Incomplete/invalid responses are not retried blindly except
                # transport-level failures; status errors fail fast.
                # Sanitized RuntimeError path below, but retry only on
                # transport markers we raised internally.
                if "status" in msg or "empty response" in msg or "invalid response" in msg:
                    # Do not log payload/image data or keys.
                    self._log.warning("meta witness failed: %s", type(e).__name__)
                    raise
                last_err = e
                self._log.warning("meta witness failed: %s", type(e).__name__)
                if attempt < attempts - 1:
                    time.sleep(_meta_retry_delay(attempt))
            except (urllib.error.HTTPError,) as e:
                code = getattr(e, "code", None)
                self._log.warning("meta witness http failed: %s", type(e).__name__)
                if _meta_status_retryable(code) and attempt < attempts - 1:
                    last_err = e
                    time.sleep(_meta_retry_delay(attempt))
                    continue
                raise RuntimeError("meta witness: request failed") from None
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                self._log.warning("meta witness transport failed: %s", type(e).__name__)
                last_err = e
                if attempt < attempts - 1:
                    time.sleep(_meta_retry_delay(attempt))
                    continue
                break
            except (ValueError, TypeError, KeyError) as e:
                self._log.warning("meta witness failed: %s", type(e).__name__)
                raise RuntimeError("meta witness: request failed") from None
        raise RuntimeError("meta witness: request failed") from None


class ScanMeteredModalGlimmer:
    """Scan-metered wrapper around the Modal Glimmer client.

    Pre-call admission uses GLIMMER_MAX_CALL_USD as the upper bound; after
    the call the measured cost (measured GPU seconds times
    GLIMMER_GPU_PER_SECOND, an estimate of L4 list price) is added to the
    per-scan ScanMeter. Never touches the ledger. Per-call timeout
    (GLIMMER_CALL_TIMEOUT) fails closed by raising so decide yields unsure.
    """

    def __init__(self, raw, *, meter_provider, max_call_usd, gpu_per_second):
        if raw is None or not callable(getattr(raw, "chat", None)):
            raise ValueError("modal glimmer raw client with chat() is required")
        self._raw = raw
        self._meter_provider = meter_provider
        self._max_call_usd = max_call_usd
        self._gpu_per_second = gpu_per_second
        try:
            self.model = getattr(raw, "model", None)
        except Exception:
            self.model = None

    def _meter(self):
        try:
            return self._meter_provider()
        except Exception:
            return None

    def warm(self, timeout=None):
        """Delegate to the raw warm() with a short timeout; never raises."""
        try:
            fn = getattr(self._raw, "warm", None)
            if not callable(fn):
                return False
            if timeout is None:
                timeout = 30.0
            try:
                timeout = float(timeout)
            except Exception:
                timeout = 30.0
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(fn)
                try:
                    return bool(fut.result(timeout=timeout))
                except BaseException:
                    try:
                        fut.cancel()
                    except Exception:
                        pass
                    return False
        except Exception:
            return False

    def _bounded_raw_chat(self, messages, options):
        timeout = glimmer_call_timeout()
        try:
            if isinstance(options, dict) and options.get("timeout") is not None:
                timeout = float(options["timeout"])
        except Exception:
            timeout = glimmer_call_timeout()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(self._raw.chat, messages, **options)
            try:
                return fut.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                try:
                    fut.cancel()
                except Exception:
                    pass
                raise RuntimeError("Glimmer call timed out")

    def chat(self, messages, **options):
        meter = self._meter()
        reserved = None
        if meter is not None:
            reserve_fn = getattr(meter, "reserve", None)
            if callable(reserve_fn):
                reserved = reserve_fn(self._max_call_usd)
            else:
                meter.check(self._max_call_usd)
        started = time.monotonic()
        try:
            response = self._bounded_raw_chat(messages, options)
        except Exception:
            if meter is not None and reserved is not None:
                try:
                    meter.release(reserved)
                except Exception:
                    pass
            raise
        elapsed = time.monotonic() - started
        seconds = _extract_inference_seconds(
            response if isinstance(response, dict) else {}, elapsed
        )
        try:
            cost = seconds * self._gpu_per_second
        except Exception:
            cost = Decimal("0")
        if not isinstance(cost, Decimal) or not cost.is_finite() or cost < 0:
            cost = Decimal("0")
        if meter is not None:
            try:
                if reserved is not None and hasattr(meter, "commit"):
                    meter.commit(reserved, cost)
                else:
                    meter.add(cost)
            except Exception:
                if reserved is not None:
                    try:
                        meter.release(reserved)
                    except Exception:
                        pass
                raise
        if isinstance(response, dict):
            meta = dict(response.get("_stillwater") or {}) if isinstance(response.get("_stillwater"), dict) else {}
            meta.update(
                {
                    "cached": meta.get("cached", False),
                    "inference_seconds": float(seconds),
                    "estimated_cost_usd": str(cost),
                }
            )
            return {**response, "_stillwater": meta}
        return response

    def chat_with_meta(self, messages, **options):
        meter = self._meter()
        reserved = None
        if meter is not None:
            reserve_fn = getattr(meter, "reserve", None)
            if callable(reserve_fn):
                reserved = reserve_fn(self._max_call_usd)
            else:
                meter.check(self._max_call_usd)
        fn = getattr(self._raw, "chat_with_meta", None)
        started = time.monotonic()
        try:
            if callable(fn):
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    fut = ex.submit(fn, messages, **options)
                    try:
                        response, cached = fut.result(timeout=glimmer_call_timeout())
                    except concurrent.futures.TimeoutError:
                        try:
                            fut.cancel()
                        except Exception:
                            pass
                        raise RuntimeError("Glimmer call timed out")
                elapsed = time.monotonic() - started
                base = response if isinstance(response, dict) else {}
                seconds = _extract_inference_seconds(base, elapsed)
                try:
                    cost = seconds * self._gpu_per_second
                except Exception:
                    cost = Decimal("0")
                if meter is not None:
                    if reserved is not None and hasattr(meter, "commit"):
                        meter.commit(reserved, cost)
                    else:
                        meter.add(cost)
                return response, cached
            response = self.chat(messages, **options)
            # self.chat already accounted; avoid double-count: release our reservation.
            if reserved is not None:
                pass
            cached = False
            try:
                cached = bool(response.get("_stillwater", {}).get("cached", False))
            except Exception:
                cached = False
            return response, cached
        except Exception:
            if meter is not None and reserved is not None:
                # If inner chat already committed, release is a no-op-safe decrement;
                # guard by checking: commit already subtracted. To avoid double-release
                # accounting errors, only release when commit did not happen.
                # Here the exception path means commit did not happen (except the
                # self.chat success path which returns, not raises).
                try:
                    # Detect whether we already delegated accounting: if fn was None
                    # and self.chat succeeded we would have returned. So release.
                    meter.release(reserved)
                except Exception:
                    pass
            raise

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.__dict__["_raw"], name)


def _segment_image_fn(client):
    """Resolve a segment_image callable from a raw or wrapped SAM client."""
    if client is None:
        raise RuntimeError("SAM client is not configured")
    fn = getattr(client, "segment_image", None)
    if callable(fn):
        return fn
    inner = getattr(client, "client", None)
    fn = getattr(inner, "segment_image", None) if inner is not None else None
    if callable(fn):
        return fn
    # Scan-metered wrappers expose _raw.
    raw = getattr(client, "_raw", None)
    if raw is not None:
        fn = getattr(raw, "segment_image", None)
        if callable(fn):
            return fn
    raise RuntimeError("SAM client does not expose segment_image")


def _chat_upper_bound(messages, max_tokens: int, output_per_m: Decimal, input_per_m: Decimal) -> Decimal:
    try:
        mt = int(max_tokens)
    except Exception:
        mt = 512
    if mt <= 0:
        mt = 512
    # Estimated input tokens: rough char-based estimate capped conservatively.
    try:
        text_len = 0
        if isinstance(messages, list):
            for m in messages:
                if isinstance(m, dict):
                    c = m.get("content", "")
                    if isinstance(c, str):
                        text_len += len(c)
                    elif isinstance(c, list):
                        for part in c:
                            if isinstance(part, dict) and isinstance(part.get("text"), str):
                                text_len += len(part["text"])
                elif isinstance(m, str):
                    text_len += len(m)
        est_input = max(64, text_len // 4)
    except Exception:
        est_input = 256
    return (Decimal(est_input) * input_per_m / Decimal(1000000)) + (Decimal(mt) * output_per_m / Decimal(1000000))


def _measured_chat_cost(response, input_per_m: Decimal, output_per_m: Decimal) -> Decimal:
    try:
        usage = None
        if isinstance(response, dict):
            for k in ("usage", "meta", "_meta", "tokens"):
                v = response.get(k)
                if isinstance(v, dict):
                    usage = v
                    break
            if usage is None and "prompt_tokens" in response:
                usage = response
        if usage is None:
            return Decimal("0")
        def _num(*keys):
            for k in keys:
                v = usage.get(k)
                if v is not None:
                    try:
                        return Decimal(str(v))
                    except Exception:
                        continue
            return Decimal("0")
        inp = _num("prompt_tokens", "input_tokens", "input")
        out = _num("completion_tokens", "output_tokens", "output")
        return (inp * input_per_m / Decimal(1000000)) + (out * output_per_m / Decimal(1000000))
    except Exception:
        return Decimal("0")


class ScanMeteredChat:
    """Baseten/Glimmer chat wrapper metered via ScanMeter; never touches ledger."""

    def __init__(self, raw, *, meter_provider, input_per_m, output_per_m, default_max_tokens: int = 512):
        self._raw = raw
        self._meter_provider = meter_provider
        self._input_per_m = input_per_m
        self._output_per_m = output_per_m
        self._default_max_tokens = default_max_tokens
        try:
            self.model = getattr(raw, "model", None)
        except Exception:
            self.model = None

    def _meter(self):
        try:
            m = self._meter_provider()
        except Exception:
            m = None
        return m

    def _charged_call(self, upper, call_fn):
        meter = self._meter()
        reserved = None
        if meter is not None:
            reserve_fn = getattr(meter, "reserve", None)
            if callable(reserve_fn):
                reserved = reserve_fn(upper)
            else:
                meter.check(upper)
        try:
            response, cached = call_fn()
        except Exception:
            if meter is not None and reserved is not None:
                try:
                    meter.release(reserved)
                except Exception:
                    pass
            raise
        cost = _measured_chat_cost(response if isinstance(response, dict) else {}, self._input_per_m, self._output_per_m)
        if meter is not None:
            if reserved is not None and hasattr(meter, "commit"):
                meter.commit(reserved, cost)
            else:
                meter.add(cost)
        return response, cached, cost

    def chat(self, messages, **options):
        meter = self._meter()
        mt = options.get("max_tokens", self._default_max_tokens)
        upper = _chat_upper_bound(messages, mt, self._output_per_m, self._input_per_m)
        # Call raw transport.
        fn = getattr(self._raw, "chat_with_meta", None)
        def _do():
            if callable(fn):
                response, cached = fn(messages, **options)
            else:
                fn2 = getattr(self._raw, "chat", None)
                if not callable(fn2):
                    raise RuntimeError("chat client is not configured")
                resp = fn2(messages, **options)
                if isinstance(resp, tuple) and len(resp) == 2:
                    response, cached = resp
                elif isinstance(resp, dict) and "response" in resp:
                    response = resp["response"] if isinstance(resp.get("response"), dict) else resp
                    cached = resp.get("cached", False)
                else:
                    response, cached = resp, False
            return response, cached
        # Reserve/commit via helper while preserving meter None path.
        if meter is None:
            response, cached = _do()
            cost = _measured_chat_cost(response if isinstance(response, dict) else {}, self._input_per_m, self._output_per_m)
        else:
            response, cached, cost = self._charged_call(upper, _do)
        if isinstance(response, dict):
            return {**response, "_stillwater": {"cached": cached, "estimated_cost_usd": str(cost)}}
        return response

    def chat_with_meta(self, messages, **options):
        mt = options.get("max_tokens", self._default_max_tokens)
        upper = _chat_upper_bound(messages, mt, self._output_per_m, self._input_per_m)
        fn = getattr(self._raw, "chat_with_meta", None)
        def _do():
            if callable(fn):
                response, cached = fn(messages, **options)
            else:
                fn2 = getattr(self._raw, "chat", None)
                if not callable(fn2):
                    raise RuntimeError("chat client is not configured")
                response = fn2(messages, **options)
                cached = False
                if isinstance(response, tuple) and len(response) == 2:
                    response, cached = response
            return response, cached
        meter = self._meter()
        if meter is None:
            response, cached = _do()
            cost = _measured_chat_cost(response if isinstance(response, dict) else {}, self._input_per_m, self._output_per_m)
        else:
            response, cached, cost = self._charged_call(upper, _do)
        return response, cached

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.__dict__["_raw"], name)


class ScanMeteredSAM:
    """SAM wrapper metered via ScanMeter; never touches ledger."""

    def __init__(self, raw, *, meter_provider, per_frame, minimum_frames: int = 1):
        self._raw = raw
        self._meter_provider = meter_provider
        self._per_frame = per_frame
        self._minimum_frames = minimum_frames

    def _meter(self):
        try:
            return self._meter_provider()
        except Exception:
            return None

    def upload_video(self, video):
        return self._raw.upload_video(video)

    def delete_file(self, file_id):
        return self._raw.delete_file(file_id)

    def segment_image(self, image: bytes, mime: str, prompt: str):
        meter = self._meter()
        reserved = None
        if meter is not None:
            reserve_fn = getattr(meter, "reserve", None)
            if callable(reserve_fn):
                reserved = reserve_fn(self._per_frame)
            else:
                meter.check(self._per_frame)
        try:
            text = self._raw.segment_image(image, mime, prompt)
        except Exception:
            if meter is not None and reserved is not None:
                try:
                    meter.release(reserved)
                except Exception:
                    pass
            raise
        if meter is not None:
            if reserved is not None and hasattr(meter, "commit"):
                meter.commit(reserved, self._per_frame)
            else:
                meter.add(self._per_frame)
        return text

    def segment_video_file_with_usage(self, file_id, prompt):
        meter = self._meter()
        reserved = None
        if meter is not None:
            reserve_fn = getattr(meter, "reserve", None)
            if callable(reserve_fn):
                reserved = reserve_fn(self._per_frame)
            else:
                meter.check(self._per_frame)
        try:
            result = self._raw.segment_video_file_with_meta(file_id, prompt)
        except Exception:
            if meter is not None and reserved is not None:
                try:
                    meter.release(reserved)
                except Exception:
                    pass
            raise
        if meter is not None:
            if reserved is not None and hasattr(meter, "commit"):
                meter.commit(reserved, self._per_frame)
            else:
                meter.add(self._per_frame)
        if isinstance(result, dict):
            return {**result, "estimated_cost_usd": str(self._per_frame)}
        return result

    def segment_video_file(self, file_id, prompt):
        r = self.segment_video_file_with_usage(file_id, prompt)
        if isinstance(r, dict):
            return r.get("text")
        return r

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.__dict__["_raw"], name)


def _witness_b_selection(env, *, modal_remote_chat=None, modal_glimmer_client=None) -> str:
    raw = _env_get(env, "WITNESS_B", "witness_b", default=None)
    if raw is not None:
        sel = str(raw).strip().lower()
        if sel not in ("meta", "glimmer"):
            raise ValueError("WITNESS_B must be 'meta' or 'glimmer'")
        return sel
    # Default is 'meta'; an explicitly injected Modal transport preserves the
    # legacy glimmer wiring so existing callers/tests keep working.
    if modal_remote_chat is not None or modal_glimmer_client is not None:
        return "glimmer"
    return "meta"


def _build_meta_spark_raw(env, *, meta_spark_client=None):
    if meta_spark_client is not None:
        if not callable(getattr(meta_spark_client, "chat", None)):
            raise ValueError("meta_spark_client fake must expose chat()")
        return meta_spark_client
    key = _env_get(env, "MODEL_API_KEY", default="")
    if not isinstance(key, str) or not key.strip():
        raise RuntimeError("witness_b (meta) is not configured: MODEL_API_KEY is missing")
    model = _env_get(env, "META_SPARK_MODEL", default=META_SPARK_MODEL_DEFAULT) or META_SPARK_MODEL_DEFAULT
    try:
        return MetaSparkChat(str(key), model=str(model))
    except ValueError as e:
        raise RuntimeError(f"witness_b (meta) is not configured: {e}") from e
    except Exception as e:
        raise RuntimeError("witness_b (meta) is not reachable: client build failed") from e


def make_clients(env, *, data_dir, modal_remote_chat=None, modal_glimmer_client=None, meta_spark_client=None):
    """Build the exact clients dict expected by production.build_runner."""
    root = Path(data_dir)
    root.mkdir(parents=True, exist_ok=True)

    budget = _daily_budget_from_env(env)
    if budget is None:
        # Default matches DeploymentConfig; build_runner still validates.
        budget = Decimal("5.00")
    scan_max = _scan_max_from_env(env, budget)
    upper_bound = scan_max

    ledger = CostLedger(root / "ledger.sqlite3", budget)
    cache = CallCache(root / "cache")
    measured = MeasuredCost(ledger)

    model_key = _env_get(env, "MODEL_API_KEY", default="")
    baseten_key = _env_get(env, "BASETEN_API_KEY", default="")
    glimmer_url = _env_get(
        env, "GLIMMER_BASE_URL", "GLIMMER_URL", "MODEL_BASE_URL", default=""
    )
    glimmer_model = _env_get(env, "GLIMMER_MODEL", default="muse-glimmer")
    baseten_model = _env_get(
        env, "BASETEN_MODEL", "MODEL_ID", "MODEL_NAME",
        default="meta-models/muse-spark-open-1026",
    )
    meta_model = _env_get(env, "META_SPARK_MODEL", default=META_SPARK_MODEL_DEFAULT) or META_SPARK_MODEL_DEFAULT
    meta_model = str(meta_model)
    day_default = _env_get(env, "DAY", "TODAY", default="1970-01-01")
    per_frame = _parse_per_frame(env)
    input_per_m = _parse_rate(env, "WITNESS_INPUT_PER_M", "1.0")
    output_per_m = _parse_rate(env, "WITNESS_OUTPUT_PER_M", "1.0")
    # META_SPARK_INPUT_PER_M / META_SPARK_OUTPUT_PER_M are unverified
    # placeholders (USD per million tokens) for the unpublished Meta API price.
    meta_input_per_m = _parse_rate(env, "META_SPARK_INPUT_PER_M", str(META_SPARK_INPUT_PER_M_DEFAULT))
    meta_output_per_m = _parse_rate(env, "META_SPARK_OUTPUT_PER_M", str(META_SPARK_OUTPUT_PER_M_DEFAULT))
    glimmer_max_call = _parse_glimmer_max_call(env)
    glimmer_gpu_per_second = _parse_glimmer_gpu_per_second(env)

    BasetenClient = _try_import("app.clients.baseten", "BasetenClient")
    GlimmerClient = _try_import("app.clients.glimmer", "GlimmerClient")
    ModalGlimmerClient = _try_import("app.clients.modal_glimmer", "ModalGlimmerClient")
    scout_scene = _try_import("app.pipeline.scout_client", "scout_scene")
    scout_concepts = _try_import("app.pipeline.scout", "scout_concepts")
    describe_container = _try_import("app.pipeline.witness", "describe_container")
    decide_track = _try_import("app.pipeline.decide", "decide_track")
    Witness = _try_import("app.pipeline.decide", "Witness")
    extract_evidence = _try_import("app.pipeline.evidence", "extract_evidence")
    extract_evidence_image = _try_import("app.pipeline.evidence", "extract_evidence_image")
    create_work_order = _try_import("app.pipeline.orders", "create_work_order")

    # Meter provider: current per-scan meter when inside a runner scan,
    # otherwise None (standalone stage calls use an ephemeral meter below).
    def _meter_provider():
        return get_current_scan_meter()

    witness_b_sel = _witness_b_selection(env, modal_remote_chat=modal_remote_chat,
                                         modal_glimmer_client=modal_glimmer_client)

    # --- raw transports (real where available) ---
    baseten_raw = None
    if BasetenClient is not None and baseten_key and str(baseten_key).strip():
        try:
            baseten_raw = BasetenClient(str(baseten_key), cache, model=str(baseten_model))
        except Exception:
            baseten_raw = None

    glimmer_raw = None
    if GlimmerClient is not None and glimmer_url and str(glimmer_url).strip() and model_key and str(model_key).strip():
        try:
            glimmer_raw = GlimmerClient(
                str(glimmer_url), str(model_key), cache, model=str(glimmer_model)
            )
        except Exception:
            glimmer_raw = None

    # --- witness_b transports ---
    modal_raw = None
    metered_modal_glimmer = None
    meta_raw = None
    metered_meta = None
    if witness_b_sel == "meta":
        try:
            meta_raw = _build_meta_spark_raw(env, meta_spark_client=meta_spark_client)
        except (RuntimeError, ValueError):
            raise
        except Exception as e:
            raise RuntimeError("witness_b (meta) is not reachable: client build failed") from e
        try:
            metered_meta = ScanMeteredChat(
                meta_raw,
                meter_provider=_meter_provider,
                input_per_m=meta_input_per_m,
                output_per_m=meta_output_per_m,
                default_max_tokens=META_SPARK_MIN_MAX_TOKENS,
            )
        except Exception as e:
            raise RuntimeError("witness_b (meta) is not reachable: metered client failed") from e
    else:
        if modal_glimmer_client is not None:
            # Injectable fake for tests: must expose chat().
            if not callable(getattr(modal_glimmer_client, "chat", None)):
                raise ValueError("modal_glimmer_client fake must expose chat()")
            modal_raw = modal_glimmer_client
        elif modal_remote_chat is not None:
            if ModalGlimmerClient is None:
                raise RuntimeError("witness_b (Glimmer) is not reachable: ModalGlimmerClient unavailable")
            if not callable(modal_remote_chat):
                raise ValueError("modal_remote_chat must be callable")
            modal_raw = ModalGlimmerClient(
                modal_remote_chat, cache, service_identity=MODAL_SERVICE_IDENTITY
            )
        elif _modal_credentials_available(env):
            # Modal credentials present (~/.modal.toml or MODAL_TOKEN_ID /
            # MODAL_TOKEN_SECRET): build the real Modal Glimmer client.
            # Fail closed with a clear error if Modal is not reachable.
            if ModalGlimmerClient is None:
                raise RuntimeError(
                    "witness_b (Glimmer) is not reachable: "
                    "app.clients.modal_glimmer.ModalGlimmerClient is unavailable"
                )
            remote = _build_modal_remote_chat()
            try:
                modal_raw = ModalGlimmerClient(
                    remote, cache, service_identity=MODAL_SERVICE_IDENTITY
                )
            except Exception as e:
                raise RuntimeError(
                    "witness_b (Glimmer) is not reachable: Modal "
                    "stillwater-glimmer:GlimmerService client failed"
                ) from e
        else:
            raise RuntimeError(
                "witness_b (Glimmer) is not configured: Modal credentials "
                "missing (~/.modal.toml or MODAL_TOKEN_ID/MODAL_TOKEN_SECRET) "
                "and no Glimmer endpoint configured"
            )

        try:
            metered_modal_glimmer = ScanMeteredModalGlimmer(
                modal_raw,
                meter_provider=_meter_provider,
                max_call_usd=glimmer_max_call,
                gpu_per_second=glimmer_gpu_per_second,
            )
        except Exception:
            metered_modal_glimmer = None
            # Fail closed: do not silently fall back to unmetered client.
            raise RuntimeError(
                "witness_b (Glimmer) is not reachable: metered Modal client failed"
            )

    def _scan_chat(raw):
        if raw is None:
            return None
        try:
            return ScanMeteredChat(
                raw,
                meter_provider=_meter_provider,
                input_per_m=input_per_m,
                output_per_m=output_per_m,
            )
        except Exception:
            return raw

    metered_baseten = _scan_chat(baseten_raw)
    metered_glimmer = _scan_chat(glimmer_raw)

    # Real hosted SAM client. MetaSAMClient(api_key, cache) is the true
    # constructor; let its own ValueError propagate when misconfigured.
    sam_raw = MetaSAMClient(str(model_key), cache)
    # Scan-metered SAM wrapper for track path (never reserves on ledger).
    try:
        sam_client = ScanMeteredSAM(
            sam_raw,
            meter_provider=_meter_provider,
            per_frame=per_frame,
            minimum_frames=int(_env_get(env, "SAM_MIN_FRAMES", default=1) or 1),
        )
    except Exception:
        sam_client = sam_raw

    def _make_detector(active_client):
        segment_fn = _segment_image_fn(active_client)

        def budgeted_detector(frame_jpeg: bytes):
            # No cache and no sampling: every decoded frame is checked
            # against the per-scan ScanMeter and sent exactly once. Any
            # failure or exhausted meter raises so blur_per_frame publishes
            # nothing (fail closed). Never reserves on the ledger here;
            # the scan-level reservation covers the whole scan.
            if not isinstance(frame_jpeg, (bytes, bytearray)) or not bytes(frame_jpeg):
                raise RuntimeError("invalid frame bytes")
            meter = get_current_scan_meter()
            ephemeral = False
            if meter is None:
                # Standalone stage call outside a runner scan: enforce the
                # same per-scan limit with an ephemeral meter and record the
                # measured total to the ledger without reserving.
                meter = ScanMeter(scan_max)
                ephemeral = True
            # Pre-call admission with atomic reservation so concurrent
            # detector calls cannot jointly exceed SCAN_MAX_USD.
            reserve_fn = getattr(meter, "reserve", None)
            reserved = None
            if callable(reserve_fn):
                reserved = reserve_fn(per_frame)
            else:
                # Pre-call admission: fail closed when the call would exceed
                # SCAN_MAX_USD.
                meter.check(per_frame)
            try:
                text = segment_fn(bytes(frame_jpeg), "image/jpeg", "person")
            except Exception:
                if reserved is not None:
                    try:
                        meter.release(reserved)
                    except Exception:
                        pass
                raise
            if not isinstance(text, str):
                if reserved is not None:
                    try:
                        meter.release(reserved)
                    except Exception:
                        pass
                raise RuntimeError("detector did not return text")
            # Measured usage: per-frame price.
            try:
                if reserved is not None and hasattr(meter, "commit"):
                    meter.commit(reserved, per_frame)
                else:
                    meter.add(per_frame)
            except Exception:
                if reserved is not None:
                    try:
                        meter.release(reserved)
                    except Exception:
                        pass
                raise
            if ephemeral:
                try:
                    ledger.record(str(day_default), meter.total())
                except Exception:
                    pass
            return {"completed": True, "text": text}

        return budgeted_detector

    def _is_image_path(p) -> bool:
        try:
            return Path(p).suffix.lower() in IMAGE_SUFFIXES
        except Exception:
            return False

    # --- stage closures matching production.py call signatures ---
    def ingest(raw, normalized):
        raw_p, norm_p = Path(raw), Path(normalized)
        if raw_p.suffix.lower() in IMAGE_SUFFIXES or norm_p.suffix.lower() in ('.jpg', '.jpeg'):
            return ingest_image(raw_p, norm_p)
        return ingest_video(Path(raw), Path(normalized))

    def probe(path):
        p = Path(path)
        if p.suffix.lower() in IMAGE_SUFFIXES or p.suffix.lower() in ('.jpg', '.jpeg'):
            try:
                return probe_image(p)
            except Exception:
                return probe_video(p)
        return probe_video(p)

    def privacy(normalized, private, *args):
        # production tries privacy(normalized, private, sam_client) then
        # (normalized, private). Every decoded frame is sent to the hosted
        # detector via the metered detector below; fail closed.
        active = args[0] if args else sam_raw
        # If the caller passed a wrapper without segment_image, fall back
        # to the raw image client for privacy.
        try:
            _segment_image_fn(active)
        except Exception:
            active = sam_raw
        detector = _make_detector(active)
        if _is_image_path(normalized) or _is_image_path(private):
            return blur_image(Path(normalized), Path(private), detector)
        return blur_per_frame(Path(normalized), Path(private), detector,
                              progress=lambda done, total: _report_progress("privacy", done, total, "frames"))

    def scout(private):
        p = Path(private)
        if _is_image_path(p):
            import cv2 as _cv2
            import numpy as _np
            data = p.read_bytes()
            frame_arr = _cv2.imdecode(_np.frombuffer(data, dtype=_np.uint8), _cv2.IMREAD_COLOR)
            if frame_arr is None or frame_arr.size == 0:
                raise RuntimeError("scout image cannot be opened")
            ok, enc = _cv2.imencode('.jpg', frame_arr, [_cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise RuntimeError("scout image could not be encoded")
            jpeg = enc.tobytes()
            if scout_scene is None or scout_concepts is None:
                raise RuntimeError("scout helpers are unavailable in this deployment")
            client = metered_baseten if metered_baseten is not None else baseten_raw
            if client is None:
                raise RuntimeError("Baseten scout client is not configured (BASETEN_API_KEY)")
            from app.pipeline.keyframes import Keyframe
            frames = [Keyframe(0, 0.0, 999.0, jpeg)]
            raw = scout_scene(client, frames)
            return scout_concepts(raw)
        frames = extract_keyframes(Path(private))
        if scout_scene is None or scout_concepts is None:
            raise RuntimeError("scout helpers are unavailable in this deployment")
        client = metered_baseten if metered_baseten is not None else baseten_raw
        if client is None:
            raise RuntimeError("Baseten scout client is not configured (BASETEN_API_KEY)")
        raw = scout_scene(client, frames)
        if scout_concepts is None:
            raise RuntimeError("scout helpers are unavailable in this deployment")
        return scout_concepts(raw)

    def track(private, concepts):
        p = Path(private)
        if _is_image_path(p):
            import cv2 as _cv2
            import numpy as _np
            data = p.read_bytes()
            arr = _cv2.imdecode(_np.frombuffer(data, dtype=_np.uint8), _cv2.IMREAD_COLOR)
            if arr is None or arr.size == 0:
                raise RuntimeError("track image cannot be opened")
            h, w = arr.shape[:2]
            return track_image(sam_client, data, list(concepts), int(w), int(h), progress=lambda d, t: _report_progress("track", d, t, "concepts"))
        # Probe first so a missing/unreadable file raises before any bytes
        # are faked; then call the real track_video with real names.
        info = probe_video(Path(private))
        video = Path(private).read_bytes()
        # Tracking proxy: send only the low-cost ffmpeg proxy of the
        # privacy-processed video to SAM, mapping results back to
        # full-resolution coordinates. TRACK_PROXY=0 keeps the original
        # full-quality path. Parallelism (TRACK_WORKERS) is preserved
        # inside track_video / track_video_proxy.
        if proxy_enabled():
            return track_video_proxy(sam_client, video, list(concepts), int(info.width), int(info.height), progress=lambda d, t: _report_progress("track", d, t, "concepts"))
        return track_video(sam_client, video, list(concepts), int(info.width), int(info.height), progress=lambda d, t: _report_progress("track", d, t, "concepts"))

    def extractor(private_video, item):
        if _is_image_path(private_video):
            if extract_evidence_image is None:
                if extract_evidence is None:
                    raise RuntimeError("evidence extractor is unavailable in this deployment")
                # Fall back: adapt video extractor is not valid for images.
                raise RuntimeError("image evidence extractor is unavailable")
            return extract_evidence_image(private_video, item)
        if extract_evidence is None:
            raise RuntimeError("evidence extractor is unavailable in this deployment")
        return extract_evidence(private_video, item)

    witness_a = None
    witness_b = None
    if Witness is not None and describe_container is not None:
        def _mk_describe(raw):
            if raw is None:
                return None

            def _describe(image: bytes, mime: str):
                return describe_container(raw, image, mime)

            return _describe

        da = _mk_describe(metered_baseten if metered_baseten is not None else baseten_raw)
        if witness_b_sel == "meta":
            db_raw = metered_meta if metered_meta is not None else meta_raw
            db = _mk_describe(db_raw)
            if da is not None:
                try:
                    witness_a = Witness(provider="baseten", model=str(baseten_model), describe=da, label="Muse Spark Open")
                except Exception:
                    witness_a = None
            if db is not None:
                try:
                    witness_b = Witness(provider="meta", model=meta_model, describe=db, label="Muse Spark 1.3")
                except Exception as e:
                    raise RuntimeError("witness_b (meta) is not reachable: Witness build failed") from e
            if witness_a is None:
                witness_a = {"provider": "baseten", "model": str(baseten_model)}
            if witness_b is None:
                raise RuntimeError("witness_b (meta) is not reachable: Witness build failed")
        else:
            # witness_b prefers the metered Modal Glimmer client so both
            # witnesses produce the same attribute JSON via describe_container;
            # invalid Glimmer JSON yields 'unsure' (human review), never agreement.
            modal_active = metered_modal_glimmer if metered_modal_glimmer is not None else modal_raw
            http_active = metered_glimmer if metered_glimmer is not None else glimmer_raw
            db_raw = modal_active if modal_active is not None else http_active
            db = _mk_describe(db_raw)
            if da is not None:
                try:
                    witness_a = Witness(provider="baseten", model=str(baseten_model), describe=da, label="Muse Spark Open")
                except Exception:
                    witness_a = None
            if db is not None:
                try:
                    witness_b = Witness(provider="glimmer", model=str(glimmer_model), describe=db, label="Muse Glimmer 30B")
                except Exception:
                    witness_b = None
            if witness_a is None:
                witness_a = {"provider": "baseten", "model": str(baseten_model)}
            if witness_b is None:
                # Fail closed: never silently substitute another provider.
                raise RuntimeError(
                    "witness_b (Glimmer) is not configured: Modal credentials "
                    "missing (~/.modal.toml or MODAL_TOKEN_ID/MODAL_TOKEN_SECRET) "
                    "and no Glimmer endpoint configured"
                )
    else:
        raise RuntimeError("witness helpers are unavailable in this deployment")

    def verify(private, tracks):
        if decide_track is None:
            raise RuntimeError("decide stage is unavailable in this deployment")
        if isinstance(witness_a, dict) and baseten_raw is None and metered_baseten is None:
            raise RuntimeError("witness_a (Baseten Spark) is not configured")
        if isinstance(witness_b, dict):
            # Fail closed; witness_b must be a real Witness (no substitution).
            raise RuntimeError("witness_b is not configured")
        wa = witness_a if not isinstance(witness_a, dict) else None
        wb = witness_b if not isinstance(witness_b, dict) else None
        if wa is None or wb is None:
            raise RuntimeError("witnesses are not configured")
        if not isinstance(tracks, list):
            raise RuntimeError("verify did not return list")
        if not tracks:
            return []
        _report_progress("verify", 0, len(tracks), "containers")
        # Await any background Glimmer warm-up started at scan start with a
        # bounded timeout; failure/timeout proceeds normally (cold start).
        # Warmed clients expose warm(); meta witness_b has no warm and skips.
        try:
            warm_fn = _warm_callable_for_client(metered_modal_glimmer if metered_modal_glimmer is not None else modal_raw)
            if warm_fn is None and wb is not None:
                warm_fn = _warm_callable_for_client(wb)
            # If warm-up was started via start_glimmer_warmup elsewhere, just
            # ensure the container is warm now (bounded); never fail closed.
            if warm_fn is not None:
                try:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _ex:
                        _f = _ex.submit(warm_fn)
                        try:
                            _f.result(timeout=glimmer_warm_timeout())
                        except BaseException:
                            try:
                                _f.cancel()
                            except Exception:
                                pass
                except Exception:
                    pass
        except Exception:
            pass
        workers = _verify_workers_from_env()
        call_timeout = glimmer_call_timeout() if use_glimmer else None
        def _one_bounded(item):
            if call_timeout is None:
                return decide_track(private, item, wa, wb, extractor=extractor)
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _ex:
                _f = _ex.submit(decide_track, private, item, wa, wb, extractor=extractor)
                try:
                    return _f.result(timeout=call_timeout)
                except concurrent.futures.TimeoutError:
                    try:
                        _f.cancel()
                    except Exception:
                        pass
                    tid = item.get("track_id") if isinstance(item, dict) else None
                    if not isinstance(tid, str) or not tid:
                        tid = "unknown"
                    entry = {
                        "track_id": tid,
                        "evidence": {"track_id": tid, "attributes": {"witness_b": "unsure"}},
                        "risk": "review",
                        "rule_id": "witness-timeout",
                    }
                    try:
                        if isinstance(item, dict) and isinstance(item.get("concept"), str) and item["concept"]:
                            entry["concept"] = item["concept"]
                    except Exception:
                        pass
                    return entry
        completed = 0
        progress_lock = threading.Lock()
        def _one_with_progress(item):
            nonlocal completed
            result = _one_bounded(item)
            with progress_lock:
                completed += 1
                _report_progress("verify", completed, len(tracks), "containers")
            return result
        if len(tracks) == 1 or workers == 1:
            return [_one_with_progress(t) for t in tracks]
        # Bounded parallelism preserving container order; fail closed on any exception.
        with ThreadPoolExecutor(max_workers=min(workers, len(tracks))) as ex:
            futures = [_one_bounded and ex.submit(_cv.copy_context().run, _one_with_progress, t) for t in tracks]
            try:
                results = [f.result() for f in futures]
            except BaseException:
                for f in futures:
                    try:
                        f.cancel()
                    except Exception:
                        pass
                raise
            return results

    def act(containers):
        if create_work_order is None:
            # containers from decide_track already carry work_order entries.
            out = []
            for c in containers:
                if isinstance(c, dict) and "work_order" in c:
                    out.append(c["work_order"])
            return out
        out = []
        for c in containers:
            if isinstance(c, dict) and "work_order" in c:
                out.append(c["work_order"])
            elif isinstance(c, dict) and "track_id" in c and "risk" in c:
                out.append(create_work_order(c["track_id"], c["risk"]))
        return out

    use_glimmer = (witness_b_sel == "glimmer")
    # Expose warm-up starter for production runner: background daemon thread
    # at scan start; awaited at verify with GLIMMER_WARM_TIMEOUT.
    def _scan_warm_starter():
        try:
            target = metered_modal_glimmer if metered_modal_glimmer is not None else modal_raw
            if target is None:
                target = witness_b
            return start_glimmer_warmup(client=target)
        except Exception:
            return None
    return {
        "ledger": ledger,
        "cache": cache,
        "sam_client": sam_client,
        "witness_a": witness_a,
        "witness_b": witness_b,
        "scout": scout,
        "ingest": ingest,
        "privacy": privacy,
        "track": track,
        "probe": probe,
        "extractor": extractor,
        "verify": verify,
        "act": act,
        "prewarm": prewarm_glimmer if use_glimmer else None,
        "warm_starter": _scan_warm_starter if use_glimmer else None,
        "modal_glimmer": metered_modal_glimmer if metered_modal_glimmer is not None else modal_raw,
        "upper_bound_usd": upper_bound,
        "scan_max_usd": scan_max,
        "cost_usd": measured,
    }


def make_runner_from_env(env):
    """Return a production runner, or None when scanning is disabled/misconfigured."""
    if _env_get(env, "SCANNING_ENABLED", default="0") != "1":
        return None
    model_key = _env_get(env, "MODEL_API_KEY", default="")
    baseten_key = _env_get(env, "BASETEN_API_KEY", default="")
    if not model_key or not str(model_key).strip():
        return None
    if not baseten_key or not str(baseten_key).strip():
        return None
    budget = _daily_budget_from_env(env)
    if budget is None:
        # build_runner requires an explicit positive finite Decimal budget.
        raw_default = os.environ.get("DAILY_BUDGET_USD")
        budget = _parse_budget(raw_default) if raw_default is not None else None
    if budget is None:
        return None
    data_dir = _env_get(env, "DATA_DIR", "data_dir", default=None)
    if data_dir is None:
        try:
            data_dir = os.environ.get("DATA_DIR", "./data")
        except Exception:
            data_dir = "./data"
    try:
        from app.pipeline.production import build_runner
    except Exception:
        return None
    try:
        clients = make_clients(env, data_dir=data_dir)
    except Exception:
        return None
    # Reconcile stale reservations once at startup (industry-standard
    # startup reconciliation): settle reservations older than 2h at the
    # conservative per-scan estimate. Never raises.
    try:
        _rec = getattr(clients.get("ledger"), "reconcile_stale", None)
        if callable(_rec):
            try:
                _n = _rec(2 * 3600)
            except TypeError:
                _n = _rec(2 * 3600, Decimal("0.35"))
            try:
                logging.getLogger("stillwater").info("reconciled %s stale reservations", int(_n))
            except Exception:
                pass
    except Exception:
        pass
    settings = {"daily_budget_usd": budget, "data_dir": str(data_dir)}
    try:
        # Ensure build_runner sees the enabling env vars.
        os.environ["SCANNING_ENABLED"] = "1"
        if "MODEL_API_KEY" not in os.environ and isinstance(env, dict) and "MODEL_API_KEY" in env:
            os.environ["MODEL_API_KEY"] = str(env["MODEL_API_KEY"])
        if "BASETEN_API_KEY" not in os.environ and isinstance(env, dict) and "BASETEN_API_KEY" in env:
            os.environ["BASETEN_API_KEY"] = str(env["BASETEN_API_KEY"])
        return build_runner(settings, clients=clients)
    except Exception:
        return None
