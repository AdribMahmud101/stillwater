"""Fail-closed video masking; caller must supply an explicit mask for every frame.

A zero mask means the upstream detector positively reported no person in that
frame. This module cannot establish detector completeness or mask accuracy.
Never treat a missing frame or detector failure as an empty mask.
"""
import math
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class PrivacyError(RuntimeError):
    """Masking could not be completed safely."""


@dataclass(frozen=True)
class PrivacyResult:
    path: Path
    frame_count: int


def blur_video(source: Path, destination: Path, masks: Iterable[np.ndarray], *, kernel: int = 51) -> PrivacyResult:
    """Apply per-frame binary person masks and publish only after full validation.

    Masks must be in decoded frame order, match its dimensions, and contain only
    binary values. No audio is copied. Do not serve input or temporary files.
    """
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve() or destination.exists():
        raise PrivacyError('Destination must be a new file distinct from the source.')
    if kernel < 51 or kernel % 2 != 1:
        raise PrivacyError('Blur kernel must be odd and at least 51.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(source))
    writer = None
    tmp_path = None
    try:
        if not cap.isOpened():
            raise PrivacyError('Video cannot be opened.')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            raise PrivacyError('Video has invalid dimensions or frame rate.')
        fd, tmp = tempfile.mkstemp(prefix='.privacy-', suffix='.mp4', dir=destination.parent)
        os.close(fd)
        tmp_path = Path(tmp)
        writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not writer.isOpened():
            raise PrivacyError('Privacy output cannot be encoded.')
        mask_provider = masks if callable(masks) else None
        iterator = None if mask_provider else iter(masks)
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            try:
                mask = mask_provider(frame) if mask_provider else next(iterator)
            except StopIteration as exc:
                raise PrivacyError('Person mask missing for a frame.') from exc
            if not isinstance(mask, np.ndarray) or mask.shape != frame.shape[:2] or mask.dtype not in (np.dtype('bool'), np.dtype('uint8')) or not np.isin(mask, [0, 1, 255]).all():
                raise PrivacyError('Person mask is malformed.')
            mask = cv2.dilate((mask != 0).astype(np.uint8), np.ones((5, 5), np.uint8))
            blurred = cv2.GaussianBlur(frame, (kernel, kernel), 0)
            frame[mask != 0] = blurred[mask != 0]
            writer.write(frame)
            count += 1
        if count == 0:
            raise PrivacyError('Video contains no readable frames.')
        if iterator is not None:
            try:
                next(iterator)
            except StopIteration:
                pass
            else:
                raise PrivacyError('More person masks than video frames.')
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
                raise PrivacyError('Privacy output could not be verified.')
        finally:
            verify.release()
        # Atomic publication without replacing an existing file.
        os.link(tmp_path, destination)
        return PrivacyResult(destination, count)
    except PrivacyError:
        raise
    except Exception as exc:
        raise PrivacyError('Privacy processing failed; no output was published.') from exc
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
