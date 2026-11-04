"""Track containers in an already privacy-processed video using hosted SAM.

Tracking-proxy mode (default, ``TRACK_PROXY=1``):

Before tracking, a lower-cost copy of the *privacy-processed* (already
blurred) video is created with ffmpeg (short side ``TRACK_PROXY_SHORT_SIDE``
default 480, never upscale, even dimensions; sparse frame selection with
``select='not(mod(n,step))'`` so proxy frame ``j`` maps to source frame
``j*step`` exactly; no audio; libx264 veryfast).  Only the proxy is sent to
SAM (privacy is unchanged: it is derived from the already-blurred video).
Results are mapped back to the full-resolution private video so everything
downstream (evidence extraction, masks, rendering, results) is unchanged.

Because proxy frames are sparser, the ``min_track_frames`` threshold is
scaled for the proxy call to ``ceil(min_track_frames/step)`` (at least 1,
and at least 2 whenever the caller asked for more than one frame) while
``det_min`` is kept unchanged.

Set ``TRACK_PROXY=0`` to bypass the proxy and use the original full-quality
path.
"""
import concurrent.futures
import contextvars
import math
import os
import random
import subprocess
import tempfile
import time
import urllib.error

from app.pipeline.dedup import deduplicate_tracks
from app.pipeline.masks import to_dedup_tracks
from app.pipeline.sam_parse import parse_sam_video


class TrackingError(RuntimeError):
    pass


def _get_worker_count() -> int:
    raw = os.getenv("TRACK_WORKERS")
    if raw is None:
        return 4
    raw = raw.strip()
    if raw == "":
        return 4
    try:
        n = int(raw)
    except Exception:
        return 4
    if n < 1:
        return 1
    if n > 10:
        return 10
    return n


def _is_transient(exc: BaseException) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    for attr in ("code", "status", "status_code"):
        val = getattr(exc, attr, None)
        if val is not None:
            try:
                c = int(val)
                if c == 429 or 500 <= c <= 599:
                    return True
                return False
            except Exception:
                continue
    if isinstance(exc, urllib.error.URLError):
        return True
    if isinstance(exc, OSError):
        msg = str(exc).lower()
        if "timed out" in msg or "timeout" in msg:
            return True
    return False


def _segment_with_retry(client, file_id: str, concept: str):
    max_retries = 2
    attempt = 0
    while True:
        try:
            return client.segment_video_file(file_id, concept)
        except Exception as exc:
            if not _is_transient(exc) or attempt >= max_retries:
                raise
            attempt += 1
            delay = 0.05 * (2 ** (attempt - 1))
            jitter = random.uniform(0, 0.02)
            time.sleep(delay + jitter)


def _segment_image_with_retry(client, image_bytes: bytes, concept: str):
    max_retries = 2
    attempt = 0
    while True:
        try:
            return client.segment_image(image_bytes, 'image/jpeg', concept)
        except Exception as exc:
            if not _is_transient(exc) or attempt >= max_retries:
                raise
            attempt += 1
            delay = 0.05 * (2 ** (attempt - 1))
            jitter = random.uniform(0, 0.02)
            time.sleep(delay + jitter)


# ---------------------------------------------------------------------------
# Tracking proxy helpers.
# ---------------------------------------------------------------------------

def proxy_enabled() -> bool:
    """True unless TRACK_PROXY is explicitly disabled (``0``/false/no/off)."""
    raw = os.getenv("TRACK_PROXY")
    if raw is None:
        return True
    s = str(raw).strip().lower()
    if s == "":
        return True
    return s not in ("0", "false", "no", "off")


def proxy_short_side() -> int:
    raw = os.getenv("TRACK_PROXY_SHORT_SIDE")
    try:
        v = int(str(raw).strip()) if raw is not None else 480
    except Exception:
        return 480
    if v <= 0:
        return 480
    return v


def proxy_fps_target() -> int:
    raw = os.getenv("TRACK_PROXY_FPS")
    try:
        v = int(str(raw).strip()) if raw is not None else 5
    except Exception:
        return 5
    if v <= 0:
        return 5
    return v


def compute_proxy_size(src_w: int, src_h: int, short_side: int) -> tuple[int, int]:
    """Scale so the short side equals short_side; never upscale; even dims."""
    if src_w <= 0 or src_h <= 0:
        raise ValueError('invalid source size')
    short = min(src_w, src_h)
    if short <= short_side:
        return int(src_w), int(src_h)
    scale = short_side / float(short)
    pw = int(round(src_w * scale))
    ph = int(round(src_h * scale))
    if pw < 2:
        pw = 2
    if ph < 2:
        ph = 2
    # Even dimensions required by libx264.
    if pw % 2:
        pw -= 1
    if ph % 2:
        ph -= 1
    if pw < 2:
        pw = 2
    if ph < 2:
        ph = 2
    # Never upscale (rounding could push a dimension above source).
    if pw > src_w:
        pw = int(src_w) - (int(src_w) % 2)
    if ph > src_h:
        ph = int(src_h) - (int(src_h) % 2)
    return pw, ph


def compute_proxy_step(source_fps: float, target_fps: float) -> int:
    """Integer frame step so proxy frame j maps to source frame j*step."""
    try:
        sf = float(source_fps)
        tf = float(target_fps)
    except Exception:
        return 1
    if not math.isfinite(sf) or not math.isfinite(tf) or sf <= 0 or tf <= 0:
        return 1
    step = int(round(sf / tf))
    if step < 1:
        step = 1
    return step


def proxy_min_track_frames(min_track_frames: int, step: int) -> int:
    """Scale min_track_frames for sparser proxy frames; keep det_min as-is."""
    try:
        m = int(min_track_frames)
    except Exception:
        m = 3
    try:
        s = int(step)
    except Exception:
        s = 1
    if s < 1:
        s = 1
    if m < 1:
        m = 1
    scaled = int(math.ceil(m / float(s)))
    if scaled < 1:
        scaled = 1
    if m > 1 and scaled < 2:
        scaled = 2
    return scaled


def probe_video_bytes(video: bytes) -> tuple[int, int, float, int]:
    """Return (width, height, fps, nframes) for video bytes via cv2."""
    import cv2
    if not isinstance(video, bytes) or not video:
        raise ValueError('invalid video bytes')
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(video)
            tmp = f.name
        cap = cv2.VideoCapture(tmp)
        try:
            if not cap.isOpened():
                raise TrackingError('private video cannot be opened')
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if w <= 0 or h <= 0:
                raise TrackingError('private video has invalid size')
            if not math.isfinite(fps) or fps <= 0:
                fps = float(proxy_fps_target())
            return w, h, fps, n
        finally:
            try:
                cap.release()
            except Exception:
                pass
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except Exception:
                pass


def build_proxy_ffmpeg_args(in_path: str, out_path: str, proxy_w: int, proxy_h: int,
                            step: int) -> list[str]:
    """ffmpeg args: scale + exact select mapping, no audio, libx264 veryfast."""
    if step > 1:
        vf = f"scale={proxy_w}:{proxy_h},select='not(mod(n\\,{step}))',setpts=N/FRAME_RATE/TB"
    else:
        vf = f"scale={proxy_w}:{proxy_h},setpts=N/FRAME_RATE/TB"
    return [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-i', in_path,
        '-vf', vf,
        '-an',
        '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
        out_path,
    ]


def make_proxy_video(video: bytes, src_w: int, src_h: int, src_fps: float,
                     short_side: int | None = None,
                     target_fps: int | None = None) -> tuple[bytes, int, int, int]:
    """Create the low-cost proxy copy; return (bytes, pw, ph, step).

    Never upscales: when the source is already at/below the short side and
    ``step == 1`` the original bytes are returned directly without invoking
    ffmpeg.  Temporary files are always removed.
    """
    if short_side is None:
        short_side = proxy_short_side()
    if target_fps is None:
        target_fps = proxy_fps_target()
    pw, ph = compute_proxy_size(int(src_w), int(src_h), int(short_side))
    step = compute_proxy_step(src_fps, target_fps)
    if pw == int(src_w) and ph == int(src_h) and step == 1:
        return bytes(video), pw, ph, step
    in_path = None
    out_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(video)
            in_path = f.name
        fd, out_path = tempfile.mkstemp(suffix='.mp4')
        try:
            os.close(fd)
        except Exception:
            pass
        args = build_proxy_ffmpeg_args(in_path, out_path, pw, ph, step)
        subprocess.run(args, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
        with open(out_path, 'rb') as f:
            proxy = f.read()
        if not proxy:
            raise TrackingError('proxy transcode produced no bytes')
        return proxy, pw, ph, step
    except TrackingError:
        raise
    except Exception as exc:
        raise TrackingError(f'proxy transcode failed: {exc}') from exc
    finally:
        for p in (in_path, out_path):
            if p:
                try:
                    os.unlink(p)
                except Exception:
                    pass


# -- mask/box upscaling (nearest-neighbour) ---------------------------------

def _scale_box(box, sx: float, sy: float, src_w: int, src_h: int):
    l, t, r, b = box
    nl = int(round(float(l) * sx))
    nt = int(round(float(t) * sy))
    nr = int(round(float(r) * sx))
    nb = int(round(float(b) * sy))
    nl = max(0, min(int(src_w), nl))
    nt = max(0, min(int(src_h), nt))
    nr = max(0, min(int(src_w), nr))
    nb = max(0, min(int(src_h), nb))
    if nr < nl:
        nr = nl
    if nb < nt:
        nb = nt
    return [nl, nt, nr, nb]


def _scale_pixels(pixels, sx: float, sy: float, src_w: int, src_h: int):
    out = set()
    for p in pixels:
        try:
            x, y = p
            x = int(x)
            y = int(y)
        except Exception:
            continue
        x0 = int(math.floor(x * sx))
        x1 = int(math.floor((x + 1) * sx))
        y0 = int(math.floor(y * sy))
        y1 = int(math.floor((y + 1) * sy))
        if x1 <= x0:
            x1 = x0 + 1
        if y1 <= y0:
            y1 = y0 + 1
        for xx in range(x0, x1):
            if xx < 0 or xx >= src_w:
                continue
            for yy in range(y0, y1):
                if yy < 0 or yy >= src_h:
                    continue
                out.add((xx, yy))
    return out


def _encode_varint(value: int) -> str:
    v = int(value)
    chars = []
    first = True
    while True:
        chunk = v & 31
        v >>= 5  # arithmetic shift for negatives in Python
        done = (v == 0 and (chunk & 16) == 0) or (v == -1 and (chunk & 16) != 0)
        if not done:
            chunk |= 32
        chars.append(chr(chunk + 48))
        if done:
            break
        first = False
    return ''.join(chars)


def _encode_counts(runs: list[int]) -> str:
    parts = []
    for i, c in enumerate(runs):
        if i >= 3:
            delta = int(c) - int(runs[i - 2])
        else:
            delta = int(c)
        parts.append(_encode_varint(delta))
    return ''.join(parts)


def _pixels_to_rle(pixels: set, offset: list, size: list) -> dict:
    h, w = int(size[0]), int(size[1])
    ox, oy = int(offset[0]), int(offset[1])
    if h <= 0 or w <= 0:
        raise ValueError('invalid mask size')
    local = set()
    for (x, y) in pixels:
        lx = int(x) - ox
        ly = int(y) - oy
        if 0 <= lx < w and 0 <= ly < h:
            local.add((lx, ly))
    total = h * w
    # Column-major flatten: index i -> (i//h, i%h).
    runs: list[int] = []
    cur = 0
    is_one = False
    count = 0
    for i in range(total):
        lx = i // h
        ly = i % h
        bit = (lx, ly) in local
        if bit == is_one:
            count += 1
        else:
            runs.append(count)
            count = 1
            is_one = bit
    runs.append(count)
    return {'size': [h, w], 'counts': _encode_counts(runs)}


def upscale_frame_to_source(frame: dict, sx: float, sy: float,
                            src_w: int, src_h: int, step: int) -> dict:
    """Map one converted frame (proxy coords) back to source coords.

    Keeps the exact output schema: 'frame', 'mask', 'score' plus 'box',
    'mask_rle', 'mask_offset'.
    """
    out = dict(frame)
    try:
        pf = int(frame.get('frame', 0))
    except Exception:
        pf = 0
    out['frame'] = int(pf * step)
    pixels = frame.get('mask')
    if isinstance(pixels, (set, frozenset)):
        up = _scale_pixels(set(pixels), sx, sy, src_w, src_h)
    else:
        up = pixels
    out['mask'] = up
    try:
        out['box'] = _scale_box(frame['box'], sx, sy, src_w, src_h)
    except Exception:
        pass
    # Rebuild a consistent box-local RLE from the upscaled pixels so
    # mask_rle/mask_offset decode to exactly out['mask'].
    try:
        if isinstance(up, (set, frozenset)) and up:
            xs = [p[0] for p in up]
            ys = [p[1] for p in up]
            ox, oy = min(xs), min(ys)
            bw = max(xs) - ox + 1
            bh = max(ys) - oy + 1
            out['mask_offset'] = [int(ox), int(oy)]
            out['mask_rle'] = _pixels_to_rle(set(up), [int(ox), int(oy)], [int(bh), int(bw)])
        else:
            # Empty (should not happen): scale offset/size arithmetically.
            old_off = frame.get('mask_offset', [0, 0])
            old_rle = frame.get('mask_rle', {'size': [1, 1], 'counts': ''})
            old_size = old_rle.get('size', [1, 1]) if isinstance(old_rle, dict) else [1, 1]
            out['mask_offset'] = [max(0, int(round(float(old_off[0]) * sx))),
                                  max(0, int(round(float(old_off[1]) * sy)))]
            out['mask_rle'] = {'size': [max(1, int(round(float(old_size[0]) * sy))),
                                           max(1, int(round(float(old_size[1]) * sx)))],
                               'counts': old_rle.get('counts', '') if isinstance(old_rle, dict) else ''}
    except Exception:
        pass
    return out


def _track_bytes_direct(client, video: bytes, concepts: list[str], width: int, height: int,
                        *, min_track_frames: int = 3, det_min: float = 0.35,
                        merge_iou: float = 0.5, progress=None) -> list[dict]:
    """Core upload/segment/parse/convert/dedup for bytes already at target size."""
    file_id = client.upload_video(video)
    try:
        workers = _get_worker_count()
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = []
            future_to_index = {}
            for idx, concept in enumerate(concepts):
                fut = executor.submit(contextvars.copy_context().run, _segment_with_retry, client, file_id, concept)
                futures.append(fut)
                future_to_index[fut] = idx
            raws = [None] * len(concepts)
            completed = 0
            try:
                for fut in concurrent.futures.as_completed(futures):
                    idx = future_to_index[fut]
                    raws[idx] = fut.result()
                    completed += 1
                    if progress is not None:
                        progress(completed, len(concepts))
            except Exception:
                for f in futures:
                    if not f.done():
                        try:
                            f.cancel()
                        except Exception:
                            pass
                raise
        all_tracks = []
        for concept, raw in zip(concepts, raws):
            parsed = parse_sam_video(raw, concept)
            for track in parsed:
                for frame in track['frames']:
                    score = frame['score']
                    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                        raise TrackingError('SAM returned missing or invalid confidence')
                track['track_id'] = f"{concept}:{track['track_id']}"
            converted = to_dedup_tracks(parsed, width, height)
            for original, track in zip(parsed, converted):
                for source, frame in zip(original['frames'], track['frames']):
                    frame.update({'box': source['box'], 'mask_rle': source['mask_rle'], 'mask_offset': source['mask_offset']})
            all_tracks.extend(converted)
        return deduplicate_tracks(all_tracks, min_track_frames, det_min, merge_iou)
    finally:
        client.delete_file(file_id)


def track_video_proxy(client, video: bytes, concepts: list[str], width: int, height: int,
                      *, min_track_frames: int = 3, det_min: float = 0.35,
                      merge_iou: float = 0.5, progress=None) -> list[dict]:
    """Track via a low-cost ffmpeg proxy, mapping results to source coords.

    Output schema is identical to :func:`track_video` (``frame``, ``mask``,
    ``score``, ``box``, ``mask_rle``, ``mask_offset`` plus dedup aliases), so
    ``deduplicate_tracks``, ``extract_evidence`` and rendering work unchanged.
    Proxy frame ``j`` maps to source frame ``j*step``; nearby source frames
    hold the nearest tracked mask, and masks/boxes are nearest-neighbour upscaled.  Proxy ``min_track_frames``
    is ``ceil(min_track_frames/step)`` (at least 1, at least 2 when the
    caller asked for >1); ``det_min`` is unchanged.
    """
    if not isinstance(video, bytes) or not video or not concepts or len(concepts) > 10:
        raise ValueError('video and one to ten concepts required')
    if any(not isinstance(c, str) or not c.strip() for c in concepts) or len(set(concepts)) != len(concepts):
        raise ValueError('concepts must be nonempty and unique')
    src_w, src_h = int(width), int(height)
    try:
        _, _, src_fps, nframes = probe_video_bytes(video)
    except Exception:
        src_fps = float(proxy_fps_target())
        nframes = 0
    proxy_bytes, pw, ph, step = make_proxy_video(video, src_w, src_h, src_fps)
    # If the proxy is identical to the source, just run the direct path.
    if step == 1 and pw == src_w and ph == src_h:
        return _track_bytes_direct(client, video, concepts, src_w, src_h,
                                   min_track_frames=min_track_frames,
                                   det_min=det_min, merge_iou=merge_iou, progress=progress)
    proxy_min = proxy_min_track_frames(min_track_frames, step)
    proxy_tracks = _track_bytes_direct(client, proxy_bytes, concepts, pw, ph,
                                       min_track_frames=1, det_min=0.0,
                                       merge_iou=merge_iou, progress=progress)
    # Map back to source coordinates, then apply the sparser threshold and
    # det_min filtering via the real dedup (masks already upscaled).
    # Re-apply mean-score/min-frames filtering manually would duplicate
    # dedup; instead filter here then dedup with merge only. Simplest
    # correct route: remap frames, then call deduplicate_tracks with the
    # proxy-scaled min_track_frames and original det_min.
    sx = src_w / float(pw) if pw else 1.0
    sy = src_h / float(ph) if ph else 1.0
    remapped = []
    for track in proxy_tracks:
        nt = dict(track)
        nfs = [upscale_frame_to_source(f, sx, sy, src_w, src_h, step) for f in track['frames']]
        nt['frames'] = nfs
        remapped.append(nt)
    # deduplicate_tracks also validates scores/masks and merges; run it with
    # the scaled threshold. Tracks here already passed through dedup once
    # (with permissive thresholds), so this second pass only enforces the
    # documented proxy threshold + det_min without further merging harm:
    # use merge_iou for consistency.
    tracks = deduplicate_tracks(remapped, proxy_min, det_min, merge_iou)
    if step > 1:
        for track in tracks:
            # Pick the closest real sample; sorting by source frame breaks ties
            # toward the earlier sample. Never hold farther than step - 1.
            chosen = {}
            for frame in sorted(track['frames'], key=lambda f: f['frame']):
                source = frame['frame']
                for index in range(max(0, source - step + 1), source + step):
                    if nframes > 0 and index >= nframes:
                        continue
                    distance = abs(index - source)
                    if index not in chosen or distance < chosen[index][0]:
                        chosen[index] = (distance, frame)
            track['frames'] = [dict(frame, frame=index)
                               for index, (_, frame) in sorted(chosen.items())]
    return tracks


def track_video(client, video: bytes, concepts: list[str], width: int, height: int,
                *, min_track_frames: int = 3, det_min: float = 0.35,
                merge_iou: float = 0.5, progress=None) -> list[dict]:
    """Upload once, segment once per concept, and delete the remote file.

    The caller must enforce privacy and reserve budget before entering. Failure
    to obtain genuine per-frame confidence fails the stage rather than making
    up scores or silently discarding objects.

    By default (``TRACK_PROXY`` unset/``1``) tracking goes through the
    low-cost ffmpeg proxy (:func:`track_video_proxy`); set ``TRACK_PROXY=0``
    for the original full-quality path.
    """
    if not isinstance(video, bytes) or not video or not concepts or len(concepts) > 10:
        raise ValueError('video and one to ten concepts required')
    if any(not isinstance(c, str) or not c.strip() for c in concepts) or len(set(concepts)) != len(concepts):
        raise ValueError('concepts must be nonempty and unique')
    if proxy_enabled():
        try:
            return track_video_proxy(client, video, concepts, width, height,
                                     min_track_frames=min_track_frames,
                                     det_min=det_min, merge_iou=merge_iou, progress=progress)
        except TrackingError:
            raise
        except ValueError:
            raise
    return _track_bytes_direct(client, video, concepts, width, height,
                               min_track_frames=min_track_frames,
                               det_min=det_min, merge_iou=merge_iou, progress=progress)


def track_image(client, image: bytes, concepts: list[str], width: int, height: int,
                *, det_min: float = 0.35, merge_iou: float = 0.5, progress=None) -> list[dict]:
    """Segment a single image once per concept with bounded parallelism.

    Uses the existing hosted image segmentation ``segment_image`` and the
    existing SAM parser as frame 0, then cross-concept dedup by mask IoU.
    min_track_frames is always 1 for images. Never fabricates frames.
    """
    if not isinstance(image, bytes) or not image or not concepts or len(concepts) > 10:
        raise ValueError('image and one to ten concepts required')
    if any(not isinstance(c, str) or not c.strip() for c in concepts) or len(set(concepts)) != len(concepts):
        raise ValueError('concepts must be nonempty and unique')
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ValueError('invalid image size')
    workers = _get_worker_count()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        future_to_index = {}
        for idx, concept in enumerate(concepts):
            fut = executor.submit(contextvars.copy_context().run, _segment_image_with_retry, client, image, concept)
            futures.append(fut)
            future_to_index[fut] = idx
        raws = [None] * len(concepts)
        completed = 0
        try:
            for fut in concurrent.futures.as_completed(futures):
                idx = future_to_index[fut]
                raws[idx] = fut.result()
                completed += 1
                if progress is not None:
                    progress(completed, len(concepts))
        except Exception:
            for f in futures:
                if not f.done():
                    try:
                        f.cancel()
                    except Exception:
                        pass
            raise
    all_tracks = []
    for concept, raw in zip(concepts, raws):
        parsed = parse_sam_video(raw, concept)
        for track in parsed:
            for frame in track['frames']:
                if frame.get('frame_index', 0) != 0:
                    raise TrackingError('SAM image result must be frame 0')
                score = frame['score']
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                    raise TrackingError('SAM returned missing or invalid confidence')
            track['track_id'] = f"{concept}:{track['track_id']}"
        converted = to_dedup_tracks(parsed, width, height)
        for original, track in zip(parsed, converted):
            for source, frame in zip(original['frames'], track['frames']):
                frame.update({'box': source['box'], 'mask_rle': source['mask_rle'], 'mask_offset': source['mask_offset']})
        all_tracks.extend(converted)
    return deduplicate_tracks(all_tracks, 1, det_min, merge_iou)
