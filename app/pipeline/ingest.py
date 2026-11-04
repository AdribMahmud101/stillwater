"""Validate and normalize uploads. Output is still private, unblurred media."""
from __future__ import annotations

import json
import math
import os
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class IngestError(ValueError):
    """The upload is invalid or media processing failed."""


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    duration: float
    codec: str


@dataclass(frozen=True)
class ImageInfo:
    width: int
    height: int
    suffix: str


@dataclass(frozen=True)
class IngestResult:
    path: Path
    source: VideoInfo
    trimmed: bool
    warnings: tuple[str, ...]


IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp'}
IMAGE_MAX_BYTES = 15 * 1024 * 1024
IMAGE_MAX_SIDE_UPLOAD = 4096
IMAGE_MAX_SIDE_NORMALIZED = 1280


def probe_video(path: Path) -> VideoInfo:
    """Read video metadata using ffprobe, rejecting malformed and audio-only files."""
    try:
        proc = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)],
            capture_output=True, text=True, timeout=30, check=True,
        )
        data = json.loads(proc.stdout)
        stream = next(stream for stream in data['streams'] if stream.get('codec_type') == 'video' and not stream.get('disposition', {}).get('attached_pic'))
        width, height = int(stream['width']), int(stream['height'])
        duration = float(data.get('format', {}).get('duration') or stream['duration'])
        if width <= 0 or height <= 0 or not math.isfinite(duration) or duration <= 0:
            raise ValueError('invalid video dimensions or duration')
        return VideoInfo(width, height, duration, str(stream.get('codec_name', 'unknown')))
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, StopIteration) as exc:
        raise IngestError('The file is not a readable video with a known duration.') from exc


def _exif_orientation(data: bytes) -> int:
    try:
        # Find Exif APP1 segment.
        pos = data.find(b'Exif\x00\x00')
        if pos < 0:
            return 1
        tiff = data[pos + 6:]
        if len(tiff) < 8:
            return 1
        endian = tiff[:2]
        if endian == b'II':
            bo = '<'
        elif endian == b'MM':
            bo = '>'
        else:
            return 1
        magic = struct.unpack(bo + 'H', tiff[2:4])[0]
        if magic != 42:
            return 1
        offset = struct.unpack(bo + 'I', tiff[4:8])[0]
        if offset >= len(tiff):
            return 1
        n = struct.unpack(bo + 'H', tiff[offset:offset + 2])[0]
        for i in range(n):
            entry = tiff[offset + 2 + i * 12: offset + 2 + (i + 1) * 12]
            if len(entry) < 12:
                break
            tag, typ, count = struct.unpack(bo + 'HHI', entry[:8])
            if tag == 0x0112 and count == 1:
                if typ == 3:
                    val = struct.unpack(bo + 'H', entry[8:10])[0]
                elif typ == 4:
                    val = struct.unpack(bo + 'I', entry[8:12])[0]
                else:
                    continue
                if 1 <= val <= 8:
                    return val
        return 1
    except Exception:
        return 1


def _apply_orientation(img, orientation: int):
    try:
        if orientation == 1:
            return img
        if orientation == 2:
            return cv2.flip(img, 1)
        if orientation == 3:
            return cv2.rotate(img, cv2.ROTATE_180)
        if orientation == 4:
            return cv2.flip(img, 0)
        if orientation == 5:
            return cv2.flip(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), 1)
        if orientation == 6:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if orientation == 7:
            return cv2.flip(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
        if orientation == 8:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except Exception:
        return img
    return img


def probe_image(path: Path) -> ImageInfo:
    """Validate an image upload: extension, size, decodable, max side."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        raise IngestError('Use a JPG, JPEG, PNG, or WebP image.')
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise IngestError('Image file is unavailable.') from exc
    if size == 0 or size > IMAGE_MAX_BYTES:
        raise IngestError('Image must be nonempty and at most 15 MB.')
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise IngestError('Image file is unavailable.') from exc
    if not data:
        raise IngestError('The file is not a readable image.')
    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.size == 0:
        raise IngestError('The file is not a readable image.')
    h, w = decoded.shape[:2]
    if w <= 0 or h <= 0 or max(w, h) > IMAGE_MAX_SIDE_UPLOAD:
        raise IngestError('Image dimensions are invalid.')
    return ImageInfo(w, h, suffix)


def ingest_image(source: Path, destination: Path, *, max_side: int = 1280, max_bytes: int = 15 * 1024 * 1024) -> ImageInfo:
    """Decode, orient, downscale and re-encode an image to clean JPEG."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise IngestError('Input and output paths must differ.')
    if source.suffix.lower() not in IMAGE_SUFFIXES:
        raise IngestError('Use a JPG, JPEG, PNG, or WebP image.')
    try:
        size = source.stat().st_size
    except OSError as exc:
        raise IngestError('Image file is unavailable.') from exc
    if size == 0 or size > max_bytes:
        raise IngestError('Image must be nonempty and at most 15 MB.')
    try:
        data = source.read_bytes()
    except OSError as exc:
        raise IngestError('Image file is unavailable.') from exc
    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.size == 0:
        raise IngestError('The file is not a readable image of the claimed type.')
    h, w = decoded.shape[:2]
    if w <= 0 or h <= 0 or max(w, h) > IMAGE_MAX_SIDE_UPLOAD:
        raise IngestError('Image dimensions are invalid.')
    orientation = _exif_orientation(data)
    decoded = _apply_orientation(decoded, orientation)
    h, w = decoded.shape[:2]
    scale = min(1.0, max_side / max(w, h))
    if scale < 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        decoded = cv2.resize(decoded, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = decoded.shape[:2]
    ok, encoded = cv2.imencode('.jpg', decoded, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise IngestError('Image normalization failed.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix='.ingest-', suffix='.jpg', dir=destination.parent)
    os.close(fd)
    try:
        Path(tmp_name).write_bytes(encoded.tobytes())
        check = cv2.imdecode(np.frombuffer(Path(tmp_name).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if check is None or check.size == 0:
            raise IngestError('Image normalization failed.')
        os.replace(tmp_name, destination)
    finally:
        Path(tmp_name).unlink(missing_ok=True)
    return ImageInfo(w, h, source.suffix.lower())


def ingest_video(source: Path, destination: Path, *, max_seconds: int = 45,
                 max_side: int = 1280, fps: int = 10,
                 max_bytes: int = 200 * 1024 * 1024,
                 upload_max_seconds: int = 60) -> IngestResult:
    """Validate an upload and write H.264 without audio, before privacy processing.

    Callers must keep both files private and delete raw media after successful
    privacy processing. This function does not apply privacy masks.
    """
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise IngestError('Input and output paths must differ.')
    if source.suffix.lower() not in {'.mp4', '.mov', '.webm'}:
        raise IngestError('Use an MP4, MOV, or WebM video.')
    if min(max_seconds, max_side, fps, max_bytes, upload_max_seconds) <= 0 or max_side < 2:
        raise ValueError('Ingest limits must be positive and max_side at least 2.')
    try:
        size = source.stat().st_size
    except OSError as exc:
        raise IngestError('Video file is unavailable.') from exc
    if size == 0 or size > max_bytes:
        raise IngestError('Video must be nonempty and at most 200 MB.')
    info = probe_video(source)
    if info.duration > upload_max_seconds:
        raise IngestError(f'Video must be at most {upload_max_seconds} seconds.')
    trimmed = info.duration > max_seconds
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix='.ingest-', suffix='.mp4', dir=destination.parent)
    os.close(fd)
    try:
        # Force divisible dimensions for yuv420p while preserving aspect ratio.
        scale = f'scale=w=trunc(iw*min(1\\,{max_side}/max(iw\\,ih))/2)*2:h=trunc(ih*min(1\\,{max_side}/max(iw\\,ih))/2)*2'
        subprocess.run(
            ['ffmpeg', '-v', 'error', '-y', '-i', str(source), '-t', str(max_seconds),
             '-vf', scale, '-r', str(fps), '-an', '-c:v', 'libx264', '-preset', 'veryfast',
             '-pix_fmt', 'yuv420p', '-movflags', '+faststart', tmp_name],
            capture_output=True, timeout=300, check=True,
        )
        probe_video(Path(tmp_name))
        os.replace(tmp_name, destination)
    except (OSError, subprocess.SubprocessError) as exc:
        raise IngestError('Video normalization failed.') from exc
    finally:
        Path(tmp_name).unlink(missing_ok=True)
    warnings = (f'Video was trimmed to {max_seconds} seconds.',) if trimmed else ()
    return IngestResult(destination, info, trimmed, warnings)
