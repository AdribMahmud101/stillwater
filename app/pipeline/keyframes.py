"""Sample scout keyframes from a privacy-processed, constant-frame-rate video.

Privacy processing is a caller prerequisite; this module does not detect or
remove people. Frame data is returned in memory and is never written to disk.
"""
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class KeyframeError(RuntimeError):
    pass


@dataclass(frozen=True)
class Keyframe:
    frame_index: int
    timestamp_seconds: float
    blur_score: float
    jpeg_bytes: bytes


def blur_score(frame: np.ndarray) -> float:
    if not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim not in (2, 3):
        raise ValueError('A nonempty grayscale or BGR frame is required.')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_keyframes(path: Path, *, count: int = 8, blur_min: float = 100.0) -> list[Keyframe]:
    """Sample up to count evenly spaced frames, returning only sharp frames.

    Timestamps assume constant FPS; call on normalized privacy output. An empty
    list means no sampled frame passed the sharpness threshold, not that the
    video contains no objects. Missing or undecodable frames raise an error.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1 or not isinstance(blur_min, (int, float)) or not math.isfinite(blur_min) or blur_min < 0:
        raise ValueError('count must be positive and blur_min nonnegative.')
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise KeyframeError('Video cannot be opened.')
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if total <= 0 or not math.isfinite(fps) or fps <= 0:
            raise KeyframeError('Video has no valid frames or frame rate.')
        actual_count = min(count, total)
        if actual_count == 1:
            indices = {total // 2}
        else:
            indices = {i * (total - 1) // (actual_count - 1) for i in range(actual_count)}
        results = []
        for index in range(max(indices) + 1):
            ok, frame = cap.read()
            if not ok:
                raise KeyframeError('Video ended before all sampled frames could be read.')
            if index not in indices:
                continue
            score = blur_score(frame)
            if score < blur_min:
                continue
            ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise KeyframeError('A sampled frame could not be encoded.')
            results.append(Keyframe(index, index / fps, score, encoded.tobytes()))
        return results
    except cv2.error as exc:
        raise KeyframeError('Keyframe extraction failed.') from exc
    finally:
        cap.release()
