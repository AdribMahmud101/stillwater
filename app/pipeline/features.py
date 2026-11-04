"""Rescan matching features derived from an RGB crop and clip time.

Normalized time approximates walk position only if both clips follow a similar
path and pace. Coarse color histograms are sensitive to light and similar
objects. Neither feature proves identity; evaluate on labelled footage.
"""
import math

import numpy as np


def extract_match_features(rgb_crop, timestamp_seconds, duration_seconds, mask=None):
    """Return compare_scans-compatible position and 4x4x4 RGB histogram.

    rgb_crop must be an HxWx3 uint8 RGB array (convert OpenCV BGR beforehand).
    mask, when provided, is an HxW boolean foreground segmentation mask.
    """
    for value in (timestamp_seconds, duration_seconds):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('time values must be finite numbers')
    if duration_seconds <= 0 or not 0 <= timestamp_seconds <= duration_seconds:
        raise ValueError('timestamp must be within a positive duration')
    if not isinstance(rgb_crop, np.ndarray) or rgb_crop.dtype != np.uint8 or rgb_crop.ndim != 3 or rgb_crop.shape[2] != 3 or not rgb_crop.size:
        raise ValueError('crop must be a nonempty uint8 RGB image')
    if mask is not None:
        if not isinstance(mask, np.ndarray) or mask.dtype != np.bool_ or mask.shape != rgb_crop.shape[:2] or not mask.any():
            raise ValueError('mask must be a nonempty boolean foreground matching the crop')
        pixels = rgb_crop[mask]
    else:
        pixels = rgb_crop.reshape(-1, 3)
    bins = pixels.astype(np.int32) // 64
    indices = bins[:, 0] * 16 + bins[:, 1] * 4 + bins[:, 2]
    hist = np.bincount(indices, minlength=64).astype(float)
    hist /= len(pixels)
    return {'position': timestamp_seconds / duration_seconds, 'histogram': hist.tolist()}
