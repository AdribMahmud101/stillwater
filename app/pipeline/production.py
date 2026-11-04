"""Production runner composing injected stages with fail-closed checks and budget."""
import concurrent.futures
import datetime
import os
import logging
import math
import shutil
import tempfile
import threading
import time
import uuid
from decimal import Decimal
from pathlib import Path

from app.clients.costs import CostLedger
from app.events import STAGES, make_event
from app.pipeline.budget_runner import run_budgeted


def _get_daily_budget(settings):
    if settings is None:
        return None
    if isinstance(settings, dict):
        for k in ("daily_budget_usd", "daily_budget", "budget", "BUDGET"):
            if k in settings:
                return settings[k]
        return None
    for k in ("daily_budget_usd", "daily_budget", "budget", "BUDGET"):
        if hasattr(settings, k):
            return getattr(settings, k)
    return None


def _extract_scan_id(record):
    if isinstance(record, dict):
        sid = record.get("id")
    else:
        sid = getattr(record, "id", None)
    if isinstance(sid, str) and sid:
        return sid
    return str(uuid.uuid4())


def _allowed_stages():
    try:
        return set(STAGES)
    except Exception:
        return {"ingest", "privacy", "scout", "track", "verify", "decide", "act"}


def _glimmer_warm_timeout() -> float:
    import math
    raw = os.environ.get("GLIMMER_WARM_TIMEOUT", "600")
    try:
        v = float(str(raw).strip())
    except Exception:
        return 600.0
    try:
        if not math.isfinite(v) or v <= 0:
            return 600.0
    except Exception:
        return 600.0
    return v


def _glimmer_call_timeout() -> float:
    import math
    raw = os.environ.get("GLIMMER_CALL_TIMEOUT", "180")
    try:
        v = float(str(raw).strip())
    except Exception:
        return 180.0
    try:
        if not math.isfinite(v) or v <= 0:
            return 180.0
    except Exception:
        return 180.0
    return v


def _find_warm_callable(clients, witness_b):
    # Prefer an explicit warm() on witness_b / glimmer transports (fakes in tests).
    for cand in (witness_b,):
        try:
            fn = getattr(cand, "warm", None)
        except Exception:
            fn = None
        if callable(fn):
            return fn
    if isinstance(clients, dict):
        for k in ("modal_glimmer", "glimmer_client", "witness_b_client", "glimmer", "modal_client"):
            try:
                cand = clients.get(k)
            except Exception:
                cand = None
            if cand is None:
                continue
            try:
                fn = getattr(cand, "warm", None)
            except Exception:
                fn = None
            if callable(fn):
                return fn
        # Legacy fire-and-forget prewarm hook: wrap it as a warm callable.
        try:
            pw = clients.get("prewarm")
        except Exception:
            pw = None
        if callable(pw):
            def _pw(timeout=None):
                try:
                    r = pw()
                    # prewarm may return a Future/thread; await it if possible.
                    if isinstance(r, concurrent.futures.Future):
                        try:
                            lim = float(timeout) if timeout is not None else _glimmer_warm_timeout()
                        except Exception:
                            lim = _glimmer_warm_timeout()
                        return bool(r.result(timeout=lim))
                    return True
                except Exception:
                    return False
            return _pw
    return None


def _start_warm_future(warm_fn):
    fut: concurrent.futures.Future = concurrent.futures.Future()
    if warm_fn is None:
        return None
    def _run():
        try:
            try:
                res = warm_fn()
            except TypeError:
                res = warm_fn(30)
            if isinstance(res, concurrent.futures.Future):
                try:
                    res = res.result(timeout=_glimmer_warm_timeout())
                except BaseException:
                    if not fut.done():
                        fut.set_result(False)
                    return
            ok = bool(res)
            if not fut.done():
                fut.set_result(ok)
        except Exception:
            if not fut.done():
                fut.set_result(False)
    t = threading.Thread(target=_run, name="glimmer-warm", daemon=True)
    t.start()
    return fut


def _await_warm_future(fut):
    # Returns 'ok' | 'failed' | 'timeout'. Never raises.
    if fut is None:
        return None
    try:
        res = fut.result(timeout=_glimmer_warm_timeout())
        return 'ok' if res else 'failed'
    except concurrent.futures.TimeoutError:
        return 'timeout'
    except Exception:
        return 'failed'


def _decide_one_with_timeout(decide_track, private, item, wa, wb, extractor, timeout):
    # Run decide_track bounded; on timeout fail closed to review/unsure.
    if decide_track is None:
        raise RuntimeError("decide_track unavailable")
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(decide_track, private, item, wa, wb, extractor=extractor)
        try:
            return fut.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            try:
                fut.cancel()
            except Exception:
                pass
            tid = None
            try:
                if isinstance(item, dict):
                    tid = item.get("track_id")
            except Exception:
                tid = None
            if not isinstance(tid, str) or not tid:
                tid = "unknown"
            concept = None
            try:
                if isinstance(item, dict):
                    concept = item.get("concept")
            except Exception:
                concept = None
            entry = {
                "track_id": tid,
                "evidence": {"track_id": tid, "attributes": {"witness_b": "unsure"}},
                "risk": "review",
                "rule_id": "witness-timeout",
            }
            if isinstance(concept, str) and concept:
                entry["concept"] = concept
            return entry


def build_runner(settings, *, clients=None):
    if os.environ.get("SCANNING_ENABLED") != "1":
        raise RuntimeError("scanning is not enabled: SCANNING_ENABLED=1 is required")
    model_key = os.environ.get("MODEL_API_KEY")
    baseten_key = os.environ.get("BASETEN_API_KEY")
    if not model_key or not str(model_key).strip() or not baseten_key or not str(baseten_key).strip():
        raise RuntimeError("MODEL_API_KEY and BASETEN_API_KEY must be configured")
    daily_budget = _get_daily_budget(settings)
    if daily_budget is None:
        raise RuntimeError("daily budget is not configured")
    if not isinstance(daily_budget, Decimal) or not daily_budget.is_finite() or daily_budget <= 0:
        raise RuntimeError("daily budget must be a positive finite Decimal")
    clients = dict(clients) if clients is not None else {}
    ledger = clients.get("ledger")
    if ledger is None:
        data_dir = None
        if isinstance(settings, dict):
            data_dir = settings.get("data_dir") or settings.get("DATA_DIR") or settings.get("dataDir")
        else:
            for attr in ("data_dir", "DATA_DIR", "dataDir", "root", "path"):
                if hasattr(settings, attr):
                    data_dir = getattr(settings, attr)
                    break
        if data_dir is None:
            data_dir = os.environ.get("DATA_DIR", "./data")
        ledger = CostLedger(Path(data_dir) / "ledger.sqlite3", daily_budget)
    sam_client = clients.get("sam_client")
    witness_a = clients.get("witness_a")
    witness_b = clients.get("witness_b")
    scout = clients.get("scout")
    ingest = clients.get("ingest")
    privacy = clients.get("privacy")
    track = clients.get("track")
    probe = clients.get("probe")
    extractor = clients.get("extractor")
    upper_bound_usd = clients.get("upper_bound_usd", clients.get("scan_max_usd", daily_budget))
    if not isinstance(upper_bound_usd, Decimal):
        try:
            upper_bound_usd = Decimal(str(upper_bound_usd))
        except Exception as e:
            raise RuntimeError("upper_bound_usd must be a Decimal") from e
    if not upper_bound_usd.is_finite() or upper_bound_usd <= 0:
        raise RuntimeError("upper_bound_usd must be a positive finite Decimal")
    if upper_bound_usd > daily_budget:
        raise RuntimeError("upper_bound_usd must not exceed daily budget")

    def runner(source, record, event_callback=None):
        if event_callback is not None and not callable(event_callback):
            raise ValueError("event_callback must be callable")
        scan_id = _extract_scan_id(record)
        day = None
        if isinstance(record, dict):
            for k in ("day", "date", "created_at", "day_iso"):
                v = record.get(k)
                if isinstance(v, str) and v:
                    day = v
                    break
        else:
            for k in ("day", "date", "created_at"):
                v = getattr(record, k, None)
                if isinstance(v, str) and v:
                    day = v
                    break
        if not isinstance(day, str):
            day = datetime.date.today().isoformat()
        else:
            try:
                datetime.date.fromisoformat(day)
            except Exception:
                day = datetime.date.today().isoformat()
        # Holder exposing the per-scan ScanMeter (created inside _inner) so
        # run_budgeted can read the metered total after failure, including
        # failures before any provider call (total 0).
        holder = {"meter": None, "calls_started": False}

        def _measured_spend():
            m = holder.get("meter")
            if m is None:
                total = Decimal("0")
            else:
                try:
                    total = m.total()
                except Exception:
                    raise RuntimeError("measured spend unavailable")
                if not isinstance(total, Decimal):
                    try:
                        total = Decimal(str(total))
                    except Exception:
                        raise RuntimeError("measured spend unavailable")
            if not total.is_finite() or total < 0:
                raise RuntimeError("measured spend unavailable")
            if holder.get("calls_started"):
                if total < Decimal("0.02"):
                    total = Decimal("0.02")
            return total

        def _inner(source, record, event_callback=None):
            # Fresh per-scan meter; inner stages check/add here and never
            # reserve on the ledger. Thread-safe via ScanMeter lock and
            # thread-local isolation per scan.
            try:
                from app.pipeline.providers import (
                    ScanMeter,
                    clear_current_scan_meter,
                    set_current_scan_meter,
                    set_current_progress,
                    clear_current_progress,
                )
            except Exception as e:
                raise RuntimeError("ScanMeter unavailable") from e
            meter = ScanMeter(upper_bound_usd)
            holder["meter"] = meter
            # Track whether at least one provider call was started (reserve/
            # check/add/commit). If failure happens while calls were in
            # flight the meter total may under-count, so settle conservatively.
            try:
                for _name in ("reserve", "check", "add", "commit"):
                    _orig = getattr(meter, _name, None)
                    if not callable(_orig):
                        continue
                    def _wrap(*a, _o=_orig, **k):
                        holder["calls_started"] = True
                        return _o(*a, **k)
                    try:
                        setattr(meter, _name, _wrap)
                    except Exception:
                        pass
            except Exception:
                pass
            set_current_scan_meter(meter)
            # Start Glimmer warm-up in a background daemon thread at the very
            # start of the scan (before ingest) so the GPU boots while
            # ingest/privacy/scout/track run.
            _warm_fn = _find_warm_callable(clients, witness_b)
            _warm_future = _start_warm_future(_warm_fn) if _warm_fn is not None else None
            # Legacy prewarm hook without warm(): still fire it (non-blocking).
            if _warm_future is None:
                _prewarm = clients.get('prewarm')
                if callable(_prewarm):
                    try:
                        _prewarm()
                    except Exception:
                        pass
            timings: dict = {}
            try:
                allowed = _allowed_stages()

                progress_lock = threading.Lock()
                last_progress = {}

                def emit(stage, status="done", **fields):
                    try:
                        if event_callback is not None and stage in allowed:
                            event_callback(make_event(scan_id, stage, status, **fields))
                    except Exception:
                        logging.getLogger(__name__).exception("Event emission failed")

                def report_progress(stage, done, total, unit):
                    try:
                        with progress_lock:
                            now = time.monotonic()
                            if stage in last_progress and now - last_progress[stage] < 0.1:
                                return
                            last_progress[stage] = now
                            emit(stage, "progress", done=done, total=total, unit=unit)
                    except Exception:
                        logging.getLogger(__name__).exception("Progress emission failed")

                set_current_progress(report_progress)

                def timed(stage, fn, **fields):
                    emit(stage, "start", **fields)
                    start = time.perf_counter()
                    try:
                        out = fn()
                    finally:
                        try:
                            timings[stage] = float(time.perf_counter() - start)
                        except Exception:
                            timings[stage] = 0.0
                    emit(stage, "done")
                    return out

                src_path = Path(source) if not isinstance(source, Path) else source
                with tempfile.TemporaryDirectory(prefix="stillwater-") as tmp:
                    root = Path(tmp)
                    is_image = src_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")
                    is_video = src_path.suffix.lower() in (".mp4", ".mov", ".webm")
                    if not src_path.is_file() or (not is_image and not is_video):
                        raise ValueError("invalid input")
                    media_type = "image" if is_image else "video"
                    raw = root / ("source" + src_path.suffix.lower())
                    shutil.copyfile(src_path, raw)
                    if is_image:
                        normalized = root / "normalized.jpg"
                        private = root / "private.jpg"
                    else:
                        normalized = root / "normalized.mp4"
                        private = root / "private.mp4"
                    def _do_ingest():
                        if ingest is None:
                            raise RuntimeError("ingest client required")
                        ingest(raw, normalized)
                        if not normalized.is_file() or normalized.stat().st_size == 0:
                            raise RuntimeError("ingest did not produce normalized media")
                    timed("ingest", _do_ingest)
                    def _do_privacy():
                        if privacy is None:
                            raise RuntimeError("privacy client required")
                        try:
                            privacy(normalized, private, sam_client)
                        except TypeError:
                            privacy(normalized, private)
                        if not private.is_file() or private.stat().st_size == 0:
                            raise RuntimeError("privacy did not produce private media")
                        if private.samefile(normalized):
                            raise RuntimeError("privacy must create distinct media")
                    privacy_fields = {}
                    if is_video:
                        try:
                            import cv2
                            cap = cv2.VideoCapture(str(normalized))
                            try:
                                frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                fps = cap.get(cv2.CAP_PROP_FPS)
                            finally:
                                cap.release()
                            if frame_total > 0:
                                privacy_fields.update(total=frame_total, unit="frames")
                            if math.isfinite(fps) and fps > 0:
                                privacy_fields["fps"] = fps
                        except Exception:
                            logging.getLogger(__name__).exception("Could not read privacy video metadata")
                    timed("privacy", _do_privacy, **privacy_fields)
                    def _do_scout():
                        if scout is None:
                            raise RuntimeError("scout client required")
                        scout_out = scout(private)
                        if isinstance(scout_out, list):
                            concepts = scout_out
                        elif isinstance(scout_out, dict) and "concepts" in scout_out:
                            concepts = scout_out["concepts"]
                        else:
                            from app.pipeline.scout import scout_concepts
                            concepts = scout_concepts(scout_out)
                        if not isinstance(concepts, list) or not concepts:
                            raise RuntimeError("scout did not produce concepts")
                        return concepts
                    concepts = timed("scout", _do_scout)
                    def _do_track():
                        if track is None:
                            raise RuntimeError("track client required")
                        try:
                            tracks = track(private, concepts)
                        except TypeError:
                            info = probe(private) if probe is not None else None
                            width = getattr(info, "width", 640) if info else 640
                            height = getattr(info, "height", 480) if info else 480
                            tracks = track(sam_client, private.read_bytes(), concepts, width, height)
                        if not isinstance(tracks, list):
                            raise RuntimeError("track did not return list")
                        return tracks
                    tracks = timed("track", _do_track, total=len(concepts), unit="concepts")
                    def _do_verify():
                        # Await the background warm-up with a bounded timeout;
                        # on failure/timeout proceed normally (cold start) and
                        # record the outcome. Never fail the scan here.
                        if _warm_future is not None:
                            try:
                                timings["glimmer_warm"] = _await_warm_future(_warm_future)
                            except Exception:
                                timings["glimmer_warm"] = "failed"
                        containers = None
                        verify_fn = clients.get("verify")
                        if verify_fn is not None:
                            containers = verify_fn(private, tracks)
                        else:
                            if witness_a is None or witness_b is None:
                                raise RuntimeError("witnesses required for verification")
                            ext = extractor
                            if ext is None:
                                from app.pipeline.evidence import extract_evidence as _ext
                                ext = _ext
                            from app.pipeline.decide import decide_track
                            call_timeout = _glimmer_call_timeout()
                            containers = []
                            if tracks:
                                report_progress("verify", 0, len(tracks), "containers")
                            for item in tracks:
                                containers.append(_decide_one_with_timeout(decide_track, private, item, witness_a, witness_b, extractor=ext, timeout=call_timeout))
                                report_progress("verify", len(containers), len(tracks), "containers")
                        if not isinstance(containers, list):
                            raise RuntimeError("verify did not return list")
                        return containers
                    containers = timed("verify", _do_verify, **({"total": len(tracks), "unit": "containers"} if tracks else {}))
                    timed("decide", lambda: None)
                    def _do_act():
                        work_orders = []
                        act_fn = clients.get("act")
                        if act_fn is not None:
                            work_orders = act_fn(containers)
                        else:
                            for c in containers:
                                if isinstance(c, dict) and "work_order" in c:
                                    work_orders.append(c["work_order"])
                        if not isinstance(work_orders, list):
                            raise RuntimeError("act did not return list")
                        return work_orders
                    work_orders = timed("act", _do_act)
                    if getattr(meter, 'count', 1) == 0:
                        raise RuntimeError('no measured provider usage recorded; refusing to report zero cost')
                    cost = meter.total()
                    if not isinstance(cost, Decimal):
                        cost = Decimal(str(cost))
                    if not cost.is_finite() or cost < 0 or cost > upper_bound_usd:
                        raise RuntimeError("cost_usd invalid or exceeds reservation")
                    # Project containers to exactly the keys app/main.py
                    # process_scan forwards to ResultsStore.save. ResultsStore
                    # only accepts WorkOrder instances for work_order, so drop
                    # anything else (there is no orders.py dict serializer;
                    # WorkOrder objects pass through unchanged).
                    try:
                        from app.pipeline.orders import WorkOrder as _WorkOrder
                    except Exception:
                        _WorkOrder = None
                    projected = []
                    for item in containers:
                        if not isinstance(item, dict):
                            continue
                        if "evidence" not in item or "risk" not in item or "rule_id" not in item:
                            continue
                        entry = {
                            "evidence": item["evidence"],
                            "risk": item["risk"],
                            "rule_id": item["rule_id"],
                        }
                        if "checks" in item:
                            entry["checks"] = item["checks"]
                        if "concept" in item:
                            entry["concept"] = item["concept"]
                        if "work_order" in item:
                            wo = item["work_order"]
                            if wo is None:
                                pass
                            elif _WorkOrder is not None and isinstance(wo, _WorkOrder):
                                entry["work_order"] = wo
                            else:
                                # Not a real WorkOrder: omit so save() stays valid.
                                pass
                        projected.append(entry)
                    containers = projected
                    if is_image:
                        annotated_bytes = None
                        try:
                            tracks_by_id = {}
                            if isinstance(tracks, list):
                                for t in tracks:
                                    if isinstance(t, dict) and isinstance(t.get("track_id"), str):
                                        tracks_by_id[t["track_id"]] = t
                            render_tracks = []
                            for c in containers:
                                try:
                                    ev = c.get("evidence")
                                    tid = getattr(ev, "track_id", None)
                                    if not isinstance(tid, str) or not tid:
                                        tid = c.get("track_id")
                                    if not isinstance(tid, str) or not tid:
                                        continue
                                    src_track = tracks_by_id.get(tid)
                                    if not isinstance(src_track, dict):
                                        continue
                                    risk = c.get("risk")
                                    if risk not in ("high", "medium", "low", "review"):
                                        continue
                                    concept = c.get("concept", src_track.get("concept"))
                                    if not isinstance(concept, str) or not concept:
                                        continue
                                    frames = src_track.get("frames")
                                    if not isinstance(frames, list) or not frames:
                                        continue
                                    rf = []
                                    for f in frames:
                                        if not isinstance(f, dict):
                                            continue
                                        idx = f.get("frame")
                                        m = f.get("mask")
                                        if type(idx) is not int or idx != 0:
                                            continue
                                        if not isinstance(m, (set, frozenset)):
                                            continue
                                        rf.append({"frame": 0, "mask": m})
                                    if not rf:
                                        continue
                                    render_tracks.append(
                                        {"track_id": tid, "concept": concept, "risk": risk, "frames": rf}
                                    )
                                except Exception:
                                    continue
                            if render_tracks:
                                from app.pipeline.render import render_image as _render_image
                                dest = root / "rendered-annotated.jpg"
                                _render_image(private, dest, render_tracks)
                                data = dest.read_bytes()
                                if isinstance(data, bytes) and data:
                                    annotated_bytes = data
                        except Exception:
                            annotated_bytes = None
                        return {
                            "private_video": private.read_bytes(),
                            "annotated_video": annotated_bytes,
                            "timeline": None,
                            "media_type": "image",
                            "containers": containers,
                            "work_orders": work_orders,
                            "concepts": concepts,
                            "cost_usd": cost,
                            "privacy_verified": True,
                            "timings": dict(timings),
                        }
                    # Render real annotated bytes + timeline from the private
                    # video and tracked masks. Any failure yields None/None
                    # (never a placeholder string or empty timeline list).
                    annotated_bytes = None
                    timeline = None
                    try:
                        tracks_by_id = {}
                        if isinstance(tracks, list):
                            for t in tracks:
                                if isinstance(t, dict) and isinstance(t.get("track_id"), str):
                                    tracks_by_id[t["track_id"]] = t
                        render_tracks = []
                        for c in containers:
                            try:
                                ev = c.get("evidence")
                                tid = getattr(ev, "track_id", None)
                                if not isinstance(tid, str) or not tid:
                                    tid = c.get("track_id")
                                if not isinstance(tid, str) or not tid:
                                    continue
                                src_track = tracks_by_id.get(tid)
                                if not isinstance(src_track, dict):
                                    continue
                                risk = c.get("risk")
                                if risk not in ("high", "medium", "low", "review"):
                                    continue
                                concept = c.get("concept", src_track.get("concept"))
                                if not isinstance(concept, str) or not concept:
                                    continue
                                frames = src_track.get("frames")
                                if not isinstance(frames, list) or not frames:
                                    continue
                                rf = []
                                for f in frames:
                                    if not isinstance(f, dict):
                                        continue
                                    idx = f.get("frame")
                                    m = f.get("mask")
                                    if type(idx) is not int:
                                        continue
                                    if not isinstance(m, (set, frozenset)):
                                        continue
                                    rf.append({"frame": idx, "mask": m})
                                if not rf:
                                    continue
                                render_tracks.append(
                                    {"track_id": tid, "concept": concept, "risk": risk, "frames": rf}
                                )
                            except Exception:
                                continue
                        if render_tracks:
                            from app.pipeline.render import render_video as _render_video
                            dest = root / "rendered-annotated.mp4"
                            rr = _render_video(private, dest, render_tracks)
                            data = dest.read_bytes()
                            tl = getattr(rr, "timeline", None)
                            if isinstance(data, bytes) and data and isinstance(tl, list) and tl:
                                ok = True
                                for e in tl:
                                    if not isinstance(e, dict) or set(e) != {"track_id", "risk", "frame_index", "timestamp_seconds"}:
                                        ok = False
                                        break
                                if ok:
                                    annotated_bytes = data
                                    timeline = tl
                    except Exception:
                        annotated_bytes = None
                        timeline = None
                    if annotated_bytes is None:
                        timeline = None
                    return {
                        "private_video": private.read_bytes(),
                        "annotated_video": annotated_bytes,
                        "timeline": timeline,
                        "media_type": "video",
                        "containers": containers,
                        "work_orders": work_orders,
                        "concepts": concepts,
                        "cost_usd": cost,
                        "privacy_verified": True,
                        "timings": dict(timings),
                    }
            finally:
                try:
                    clear_current_progress()
                    clear_current_scan_meter()
                except Exception:
                    pass

        return run_budgeted(ledger, day, upper_bound_usd, _inner, source, record, event_callback=event_callback, measured_spend=_measured_spend)

    return runner
