"""Extract visual evidence from already privacy-filtered, constant-FPS video.

The caller must establish that the input has passed privacy processing and
that track frame indices refer to that same video; this module cannot verify
those claims. Bytes remain in memory and are not written to the filesystem.

Evidence selection: for every track the best frame maximises
  score = sharpness x mask-area-fraction x detection-confidence
where sharpness is the variance of the Laplacian on the crop. Frames whose
mask touches the frame border are deprioritised: when any fully-inside mask
exists only those compete for best (mask fully inside when possible). When
all crops are uniformly blurry (sharpness 0 everywhere) the ranking falls
back to mask-area-fraction x confidence so selection stays deterministic.
Alt crops: the top distinct frames at least 5 frames apart (best + up to 2
alternates, i.e. up to 3 crops total) are returned as ``alt_crops`` JPEG
bytes for multi-frame witnessing. Privacy-processed video only.
"""
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


class EvidenceError(RuntimeError):
    pass


@dataclass(frozen=True)
class Evidence:
    track_id: str
    frame_index: int
    timestamp_seconds: float
    frame_jpeg: bytes
    crop_jpeg: bytes
    mask_png: bytes
    crop_box: tuple[int, int, int, int]
    alt_crops: tuple[bytes, ...] = field(default=())


def _laplacian_variance(crop) -> float:
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception:
        return 0.0


def _validate_track(track: dict):
    try:
        track_id = track['track_id']
        frames = track['frames']
        if not isinstance(track_id, str) or not track_id or not isinstance(frames, list) or not frames:
            raise ValueError
        for item in frames:
            score, index, mask = item['score'], item['frame'], item['mask']
            if type(index) is not int or index < 0 or isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1 or not isinstance(mask, (set, frozenset)) or not mask:
                raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceError('Invalid track or missing evidence.') from exc
    return track_id, frames


def _crop_for(frame, mask: np.ndarray, width: int, height: int, padding: int):
    ys, xs = np.nonzero(mask)
    x1, y1 = max(0, int(xs.min()) - padding), max(0, int(ys.min()) - padding)
    x2, y2 = min(width, int(xs.max()) + 1 + padding), min(height, int(ys.max()) + 1 + padding)
    crop = frame[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)


def extract_evidence(video: Path, track: dict, *, padding: int = 12) -> Evidence:
    """Return a real frame, object crop and full-frame binary mask.

    Tracks use the decoded deduplication contract: frames with integer 'frame',
    'score', and nonempty sets of (x,y) mask pixels. Best frame maximises
    sharpness x area-fraction x score with earliest-frame tie breaking.
    Timestamps assume constant FPS.
    """
    if type(padding) is not int or padding < 0:
        raise ValueError('padding must be a nonnegative integer')
    track_id, frames = _validate_track(track)
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise EvidenceError('Private video cannot be opened.')
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        max_index = max(item['frame'] for item in frames)
        if width <= 0 or height <= 0 or total <= max_index or not math.isfinite(fps) or fps <= 0:
            raise EvidenceError('Evidence frame is missing or video metadata is invalid.')
        # Validate mask coordinates first (fail closed before decoding).
        for item in frames:
            for pixel in item['mask']:
                if not isinstance(pixel, tuple) or len(pixel) != 2 or any(type(v) is not int for v in pixel):
                    raise EvidenceError('Mask coordinates must be integer pixels.')
                x, y = pixel
                if not (0 <= x < width and 0 <= y < height):
                    raise EvidenceError('Mask coordinate is outside the video frame.')
        # Sequential reads avoid using an approximate seek as evidence.
        wanted = {item['frame'] for item in frames}
        decoded: dict[int, object] = {}
        for idx in range(max_index + 1):
            ok, frame = cap.read()
            if not ok:
                raise EvidenceError('Evidence frame could not be decoded.')
            if idx in wanted:
                decoded[idx] = frame
        # Score every candidate frame.
        scored = []
        for item in frames:
            index = item['frame']
            frame = decoded[index]
            mask = np.zeros((height, width), dtype=np.uint8)
            for (x, y) in item['mask']:
                mask[y, x] = 255
            crop, box = _crop_for(frame, mask, width, height, padding)
            sharp = _laplacian_variance(crop)
            area_frac = float(len(item['mask'])) / float(width * height)
            conf = float(item['score'])
            raw = sharp * area_frac * conf
            fallback = area_frac * conf
            touches_border = any(
                x == 0 or x == width - 1 or y == 0 or y == height - 1
                for (x, y) in item['mask']
            )
            scored.append({
                'frame': index, 'score': conf, 'sharp': sharp,
                'raw': raw, 'fallback': fallback,
                'inside': not touches_border,
                'box': box, 'crop': crop, 'full': frame, 'mask': mask,
            })
        pool = [s for s in scored if s['inside']] or scored
        use_raw = any(s['raw'] > 0 for s in pool)
        key = (lambda s: (s['raw'], s['frame'])) if use_raw else (lambda s: (s['fallback'], s['frame']))
        # Best: max score, earliest-frame tie break.
        best = min(pool, key=lambda s: (-key(s)[0], s['frame']))
        # Rank all candidates by the same key for alternates.
        ranked = sorted(pool, key=lambda s: (-key(s)[0], s['frame']))
        # Top distinct frames at least 5 apart (best + up to 2 alternates).
        chosen = [best]
        for cand in ranked:
            if cand['frame'] == best['frame']:
                continue
            if all(abs(cand['frame'] - c['frame']) >= 5 for c in chosen):
                chosen.append(cand)
            if len(chosen) >= 3:
                break
        # Encode best.
        encoded = []
        for extension, array in (('.jpg', best['full']), ('.jpg', best['crop']), ('.png', best['mask'])):
            ok, data = cv2.imencode(extension, array)
            if not ok:
                raise EvidenceError('Evidence could not be encoded.')
            encoded.append(data.tobytes())
        # Encode alternates as crop JPEGs.
        alt_crops: list[bytes] = []
        for extra in chosen[1:]:
            ok, data = cv2.imencode('.jpg', extra['crop'])
            if not ok:
                raise EvidenceError('Evidence could not be encoded.')
            alt_crops.append(data.tobytes())
        return Evidence(track_id, best['frame'], best['frame'] / fps, *encoded, best['box'], tuple(alt_crops))
    except cv2.error as exc:
        raise EvidenceError('Evidence extraction failed.') from exc
    finally:
        cap.release()


def extract_evidence_image(image: Path, track: dict, *, padding: int = 12) -> Evidence:
    """Extract evidence from a single privacy-filtered image (frame 0, t=0.0)."""
    if type(padding) is not int or padding < 0:
        raise ValueError('padding must be a nonnegative integer')
    try:
        track_id = track['track_id']
        frames = track['frames']
        if not isinstance(track_id, str) or not track_id or not isinstance(frames, list) or not frames:
            raise ValueError
        for item in frames:
            score, index, mask = item['score'], item['frame'], item['mask']
            if type(index) is not int or index != 0 or isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1 or not isinstance(mask, (set, frozenset)) or not mask:
                raise ValueError
        selected = min(frames, key=lambda item: (-item['score'], item['frame']))
    except (KeyError, TypeError, ValueError) as exc:
        raise EvidenceError('Invalid track or missing evidence.') from exc
    import numpy as _np
    data = Path(image).read_bytes()
    frame = cv2.imdecode(_np.frombuffer(data, dtype=_np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise EvidenceError('Private image cannot be opened.')
    height, width = frame.shape[:2]
    mask = _np.zeros((height, width), dtype=_np.uint8)
    for pixel in selected['mask']:
        if not isinstance(pixel, tuple) or len(pixel) != 2 or any(type(v) is not int for v in pixel):
            raise EvidenceError('Mask coordinates must be integer pixels.')
        x, y = pixel
        if not (0 <= x < width and 0 <= y < height):
            raise EvidenceError('Mask coordinate is outside the image frame.')
        mask[y, x] = 255
    ys, xs = _np.nonzero(mask)
    x1, y1 = max(0, int(xs.min()) - padding), max(0, int(ys.min()) - padding)
    x2, y2 = min(width, int(xs.max()) + 1 + padding), min(height, int(ys.max()) + 1 + padding)
    crop = frame[y1:y2, x1:x2]
    encoded = []
    for extension, array in (('.jpg', frame), ('.jpg', crop), ('.png', mask)):
        ok, enc = cv2.imencode(extension, array)
        if not ok:
            raise EvidenceError('Evidence could not be encoded.')
        encoded.append(enc.tobytes())
    return Evidence(track_id, 0, 0.0, *encoded, (x1, y1, x2, y2), ())
