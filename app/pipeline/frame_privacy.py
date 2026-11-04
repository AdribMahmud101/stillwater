"""Experimental per-frame SAM privacy; not wired to the public pipeline.

Each raw frame is sent to the injected detector. Every call must reserve budget,
return only after provider completion, and account for usage. Frame-by-frame
completion avoids video sampling ambiguity but cannot guarantee detector recall.
"""
from __future__ import annotations

import contextvars
import math
import os
import random
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from app.pipeline.masks import to_dedup_tracks
from app.pipeline.privacy import PrivacyError, PrivacyResult
from app.pipeline.sam_parse import parse_sam_video


class FramePrivacyError(PrivacyError):
    pass


def _workers_from_env() -> int:
    raw = os.environ.get("PRIVACY_WORKERS", "12")
    try:
        n = int(str(raw).strip())
    except Exception:
        n = 12
    if n < 1:
        return 1
    if n > 32:
        return 32
    return n


def _retry_after_seconds(exc) -> float | None:
    for attr in ("retry_after", "retry_after_seconds"):
        try:
            v = getattr(exc, attr, None)
            if v is not None:
                f = float(v)
                if math.isfinite(f) and f >= 0:
                    return min(f, 60.0)
        except Exception:
            pass
    try:
        resp = getattr(exc, "response", None)
        headers = getattr(resp, "headers", None)
        if isinstance(headers, dict):
            for k in ("Retry-After", "retry-after"):
                if k in headers and headers[k] is not None:
                    f = float(str(headers[k]).strip())
                    if math.isfinite(f) and f >= 0:
                        return min(f, 60.0)
        elif headers is not None:
            get = getattr(headers, "get", None)
            if callable(get):
                for k in ("Retry-After", "retry-after"):
                    try:
                        v = get(k)
                    except Exception:
                        v = None
                    if v is not None:
                        try:
                            f = float(str(v).strip())
                        except Exception:
                            continue
                        if math.isfinite(f) and f >= 0:
                            return min(f, 60.0)
    except Exception:
        pass
    return None


def _status_code(exc) -> int | None:
    for attr in ("status_code", "status", "code", "http_status"):
        try:
            v = getattr(exc, attr, None)
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                return int(v)
            if isinstance(v, str) and v.strip().isdigit():
                return int(v.strip())
        except Exception:
            continue
    try:
        resp = getattr(exc, "response", None)
        if resp is not None:
            for attr in ("status_code", "status"):
                try:
                    v = getattr(resp, attr, None)
                    if isinstance(v, bool):
                        continue
                    if isinstance(v, (int, float)):
                        return int(v)
                except Exception:
                    continue
    except Exception:
        pass
    name = type(exc).__name__.lower()
    # Best-effort parse like "429", "503" embedded in message.
    try:
        msg = str(exc)
        import re

        m = re.search(r"\b(429|5\d\d)\b", msg)
        if m:
            return int(m.group(1))
        _ = name
    except Exception:
        pass
    return None


def _is_transient(exc: BaseException) -> bool:
    code = _status_code(exc)
    if code is not None:
        if code == 429:
            return True
        if 500 <= code <= 599:
            return True
        return False
    # Timeouts / temporary network errors are transient.
    if isinstance(exc, TimeoutError):
        return True
    name = type(exc).__name__.lower()
    mod = type(exc).__module__.lower()
    transient_names = ("timeout", "timedout", "temporarily", "connectionerror",
                       "connectionreset", "connecterror", "serviceunavailable",
                       "toomanyrequests", "rate-limit", "ratelimit", "throttl",
                       "socketerror", "econn", "http500", "http502", "http503",
                       "http504", "badgateway", "gatewaytimeout")
    hay = (name + " " + mod + " " + str(exc).lower())
    for t in transient_names:
        if t in hay:
            return True
    # Explicit marker attributes.
    for attr in ("transient", "retryable", "is_transient", "is_retryable"):
        try:
            v = getattr(exc, attr, None)
            if v is True:
                return True
        except Exception:
            pass
    return False


def _call_detector_with_retry(budgeted_detector, payload: bytes):
    # Up to 3 retries on transient errors (i.e. up to 4 attempts total).
    attempt = 0
    while True:
        try:
            return budgeted_detector(payload)
        except FramePrivacyError:
            # Invalid/undecodable responses are mapped by callers; detector
            # raising FramePrivacyError is never retried.
            raise
        except PrivacyError:
            raise
        except Exception as exc:
            if not _is_transient(exc):
                raise
            if attempt >= 5:
                raise
            delay = _retry_after_seconds(exc)
            if delay is None:
                delay = (2.0 ** attempt) * 1.0 + random.uniform(0, 0.5)
            else:
                delay = float(delay) + random.uniform(0, 0.05)
            delay = min(max(delay, 0.0), 65.0)
            if delay > 0:
                time.sleep(delay)
            attempt += 1


def _mask_from_result(text, width: int, height: int) -> np.ndarray:
    try:
        tracks = to_dedup_tracks(parse_sam_video(text, 'person'), width, height)
        mask = np.zeros((height, width), dtype=np.uint8)
        for track in tracks:
            for record in track['frames']:
                if type(record['frame']) is not int or record['frame'] != 0:
                    raise ValueError('Unexpected image frame index')
                for x, y in record['mask']:
                    mask[y, x] = 1
    except (ValueError, KeyError, TypeError, IndexError) as exc:
        raise FramePrivacyError('Frame detection could not be decoded.') from exc
    return mask


def _detect_one(frame: np.ndarray, budgeted_detector) -> np.ndarray:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 or not frame.size:
        raise FramePrivacyError('Invalid video frame.')
    height, width = frame.shape[:2]
    ok, encoded = cv2.imencode('.jpg', frame)
    if not ok:
        raise FramePrivacyError('Frame could not be encoded.')
    payload = encoded.tobytes()
    try:
        result = _call_detector_with_retry(budgeted_detector, payload)
    except FramePrivacyError:
        raise
    except PrivacyError as exc:
        raise FramePrivacyError(str(exc) or 'Frame detection did not complete.') from exc
    except Exception as exc:
        import logging
        logging.getLogger('stillwater').error('frame detector error: %s: %s', type(exc).__name__, str(exc)[:300])
        raise FramePrivacyError('Frame detection did not complete.') from exc
    if not isinstance(result, dict) or result.get('completed') is not True or not isinstance(result.get('text'), str):
        raise FramePrivacyError('Frame detection did not complete.')
    return _mask_from_result(result['text'], width, height)


def frame_masks(frames, budgeted_detector):
    """Yield one mask per frame. Detector accepts JPEG bytes and returns a mapping
    with `completed` True and the provider's segmentation `text` string.
    The caller must provide a budget-reserving detector; this cannot be enforced
    by the type contract. Detector calls transmit unblurred frames externally.
    """
    for frame in frames:
        yield _detect_one(frame, budgeted_detector)


def blur_image(source, destination, budgeted_detector):
    """Blur person regions in a single image; fail closed publishing nothing."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or destination.exists():
        raise FramePrivacyError('Destination must be a new file distinct from the source.')
    data = source.read_bytes()
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise FramePrivacyError('Image cannot be opened.')
    mask = _detect_one(frame, budgeted_detector)
    if not isinstance(mask, np.ndarray) or mask.shape != frame.shape[:2]:
        raise FramePrivacyError('Person mask is malformed.')
    kernel = 51
    dilate_kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate((mask != 0).astype(np.uint8), dilate_kernel)
    blurred = cv2.GaussianBlur(frame, (kernel, kernel), 0)
    frame[dilated != 0] = blurred[dilated != 0]
    ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise FramePrivacyError('Privacy output cannot be encoded.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.privacy-', suffix='.jpg', dir=destination.parent)
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        tmp_path.write_bytes(encoded.tobytes())
        check = cv2.imdecode(np.frombuffer(tmp_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if check is None or check.size == 0:
            raise FramePrivacyError('Privacy output could not be verified.')
        os.link(tmp_path, destination)
        return PrivacyResult(destination, 1)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def blur_per_frame(source, destination, budgeted_detector, *, progress=None):
    """Candidate only: detect every decoded frame and publish only after success.

    Bounded parallelism: decode in ordered chunks (workers*2 frames), submit
    one detector call per frame to a ThreadPoolExecutor, wait for ALL results
    in frame order, then blur and write in order. Every frame is sent exactly
    once; any failure cancels the rest and publishes nothing (fail closed).
    """
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or destination.exists():
        raise FramePrivacyError('Destination must be a new file distinct from the source.')
    workers = _workers_from_env()
    chunk_size = max(1, workers * 2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    writer = None
    tmp_path = None
    executor = None
    try:
        if not cap.isOpened():
            raise FramePrivacyError('Video cannot be opened.')
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            raise FramePrivacyError('Video has invalid dimensions or frame rate.')
        fd, tmp = tempfile.mkstemp(prefix='.privacy-', suffix='.mp4', dir=destination.parent)
        os.close(fd)
        tmp_path = Path(tmp)
        writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not writer.isOpened():
            raise FramePrivacyError('Privacy output cannot be encoded.')
        try:
            from app.pipeline.privacy import PrivacyError as _PE
        except Exception:
            _PE = PrivacyError
        kernel = 51
        dilate_kernel = np.ones((5, 5), np.uint8)
        count = 0
        exhausted = False
        executor = ThreadPoolExecutor(max_workers=workers)
        while not exhausted:
            chunk = []
            for _ in range(chunk_size):
                ok, frame = cap.read()
                if not ok:
                    exhausted = True
                    break
                chunk.append(frame)
            if not chunk:
                break
            # Submit one detector call per frame.
            futures = [executor.submit(contextvars.copy_context().run, _detect_one, frm, budgeted_detector) for frm in chunk]
            masks = None
            try:
                # Wait for ALL results in frame order; fail closed on first error.
                masks = [f.result() for f in futures]
            except BaseException:
                for f in futures:
                    try:
                        f.cancel()
                    except Exception:
                        pass
                raise
            # Blur and write strictly in order.
            for frame, mask in zip(chunk, masks):
                if not isinstance(mask, np.ndarray) or mask.shape != frame.shape[:2]:
                    raise FramePrivacyError('Person mask is malformed.')
                try:
                    if mask.dtype not in (np.dtype('bool'), np.dtype('uint8')) or not np.isin(mask, [0, 1, 255]).all():
                        raise FramePrivacyError('Person mask is malformed.')
                except FramePrivacyError:
                    raise
                except Exception as exc:
                    raise FramePrivacyError('Person mask is malformed.') from exc
                dilated = cv2.dilate((mask != 0).astype(np.uint8), dilate_kernel)
                blurred = cv2.GaussianBlur(frame, (kernel, kernel), 0)
                frame[dilated != 0] = blurred[dilated != 0]
                writer.write(frame)
                count += 1
            if progress is not None and total > 0:
                progress(count, total)
        if count == 0:
            raise FramePrivacyError('Video contains no readable frames.')
        if progress is not None:
            progress(count, count)
        writer.release()
        writer = None
        verify = cv2.VideoCapture(str(tmp_path))
        try:
            written = 0
            while True:
                ok, _ = verify.read()
                if not ok:
                    break
                written += 1
            if written != count:
                raise FramePrivacyError('Privacy output could not be verified.')
        finally:
            verify.release()
        os.link(tmp_path, destination)
        return PrivacyResult(destination, count)
    except (FramePrivacyError, PrivacyError):
        raise
    except Exception as exc:
        raise FramePrivacyError('Privacy processing failed; no output was published.') from exc
    finally:
        try:
            cap.release()
        except Exception:
            pass
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
        if executor is not None:
            try:
                executor.shutdown(wait=True, cancel_futures=True)
            except TypeError:
                try:
                    executor.shutdown(wait=True)
                except Exception:
                    pass
            except Exception:
                pass
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
