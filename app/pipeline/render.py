"""Render recorded track masks on privacy-filtered constant-FPS video.

The caller must verify that the input is private and every track refers to this
exact video. This renderer cannot validate detector accuracy or privacy coverage.
"""
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class RenderError(RuntimeError):
    pass


@dataclass(frozen=True)
class RenderResult:
    path: Path
    frame_count: int
    timeline: list[dict]


# BGR representation of the handover's risk palette.
_COLORS = {'high': (38, 38, 220), 'medium': (6, 119, 217),
           'low': (74, 163, 22), 'review': (237, 58, 124)}


def _draw_overlay(frame, width: int, height: int, entries) -> np.ndarray:
    for ident, concept, risk, pixels in sorted(entries, key=lambda e: e[0]):
        mask = np.zeros((height, width), dtype=np.uint8)
        for x, y in pixels:
            mask[y, x] = 255
        color = _COLORS[risk]
        overlay = frame.copy()
        overlay[mask != 0] = color
        frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, color, 2)
        ys, xs = np.nonzero(mask)
        label = f'{ident} {concept} {"NEEDS REVIEW" if risk == "review" else risk.upper()}'
        label = label.encode('ascii', errors='replace').decode('ascii')[:60]
        x, y = int(xs.min()), max(14, int(ys.min()) - 5)
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def render_image(private_image: Path, destination: Path, tracks: list[dict]) -> RenderResult:
    """Render masks on a single private image; output is a clean JPEG."""
    source, dest = Path(private_image), Path(destination)
    if source.resolve() == dest.resolve() or dest.exists():
        raise RenderError('Destination must be a new file distinct from the input.')
    if not isinstance(tracks, list):
        raise RenderError('Tracks must be a list.')
    data = source.read_bytes()
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.size == 0:
        raise RenderError('Private image cannot be opened.')
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        raise RenderError('Image metadata is invalid.')
    entries = []
    seen = set()
    for track in tracks:
        if not isinstance(track, dict):
            raise RenderError('Invalid track.')
        ident, concept, risk, frames = (track.get(k) for k in ('track_id', 'concept', 'risk', 'frames'))
        if not isinstance(ident, str) or not ident or ident in seen or not isinstance(concept, str) or not concept or risk not in _COLORS or not isinstance(frames, list) or not frames:
            raise RenderError('Invalid track metadata.')
        seen.add(ident)
        for item in frames:
            if not isinstance(item, dict):
                raise RenderError('Invalid track frame.')
            index, pixels = item.get('frame'), item.get('mask')
            if type(index) is not int or index != 0 or not isinstance(pixels, (set, frozenset)) or not pixels:
                raise RenderError('Image tracks must reference frame 0.')
            for pixel in pixels:
                if not isinstance(pixel, tuple) or len(pixel) != 2 or any(type(v) is not int for v in pixel) or not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
                    raise RenderError('Mask is outside the frame.')
            entries.append((ident, concept, risk, pixels))
    frame = _draw_overlay(frame, width, height, entries)
    ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RenderError('Annotated image could not be encoded.')
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.render-', suffix='.jpg', dir=dest.parent)
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        tmp_path.write_bytes(encoded.tobytes())
        check = cv2.imdecode(np.frombuffer(tmp_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if check is None or check.size == 0:
            raise RenderError('Annotated image could not be verified.')
        os.link(tmp_path, dest)
    finally:
        tmp_path.unlink(missing_ok=True)
    return RenderResult(dest, 1, [])


def render_video(private_video: Path, destination: Path, tracks: list[dict]) -> RenderResult:
    """Render only provided per-frame mask pixels and label them with risk.

    Track schema: track_id, concept, risk and frames containing frame, mask.
    Masks are sets of (x,y) pixel coordinates in the input video's timebase.
    Destination must not exist; output is published only after verification.
    """
    source, dest = Path(private_video), Path(destination)
    if source.resolve() == dest.resolve() or dest.exists():
        raise RenderError('Destination must be a new file distinct from the input.')
    if not isinstance(tracks, list):
        raise RenderError('Tracks must be a list.')
    cap = cv2.VideoCapture(str(source))
    writer, tmp_path, encoded_path = None, None, None
    try:
        if not cap.isOpened():
            raise RenderError('Private video cannot be opened.')
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
        if width <= 0 or height <= 0 or width % 2 or height % 2 or total <= 0 or not math.isfinite(fps) or fps <= 0:
            raise RenderError('Video metadata is invalid.')
        by_frame, timeline, seen = {}, [], set()
        for track in tracks:
            if not isinstance(track, dict):
                raise RenderError('Invalid track.')
            ident, concept, risk, frames = (track.get(k) for k in ('track_id', 'concept', 'risk', 'frames'))
            if not isinstance(ident, str) or not ident or ident in seen or not isinstance(concept, str) or not concept or risk not in _COLORS or not isinstance(frames, list) or not frames:
                raise RenderError('Invalid track metadata.')
            seen.add(ident)
            indices = set()
            for item in frames:
                if not isinstance(item, dict):
                    raise RenderError('Invalid track frame.')
                index, pixels = item.get('frame'), item.get('mask')
                if type(index) is not int or not 0 <= index < total or index in indices or not isinstance(pixels, (set, frozenset)) or not pixels:
                    raise RenderError('Missing or duplicate frame or mask.')
                indices.add(index)
                for pixel in pixels:
                    if not isinstance(pixel, tuple) or len(pixel) != 2 or any(type(v) is not int for v in pixel) or not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
                        raise RenderError('Mask is outside the frame.')
                by_frame.setdefault(index, []).append((ident, concept, risk, pixels))
            first = min(indices)
            timeline.append({'track_id': ident, 'risk': risk, 'frame_index': first, 'timestamp_seconds': first / fps})
        dest.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix='.render-', suffix='.mp4', dir=dest.parent)
        os.close(fd)
        tmp_path = Path(tmp)
        writer = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not writer.isOpened():
            raise RenderError('Video encoder could not start.')
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = _draw_overlay(frame, width, height, by_frame.get(count, []))
            writer.write(frame)
            count += 1
        if count != total or any(index >= count for index in by_frame):
            raise RenderError('Video frames are missing or undecodable.')
        writer.release()
        writer = None
        # OpenCV's MP4V encoder is not widely playable in browsers. Transcode
        # the private intermediate to H.264/yuv420p before publication.
        fd, encoded = tempfile.mkstemp(prefix='.render-', suffix='.mp4', dir=dest.parent)
        os.close(fd)
        encoded_path = Path(encoded)
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(tmp_path), '-an',
                        '-c:v', 'libx264', '-preset', 'veryfast', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', str(encoded_path)],
                       capture_output=True, timeout=300, check=True)
        verify = cv2.VideoCapture(str(encoded_path))
        try:
            written = 0
            while verify.read()[0]:
                written += 1
            if written != count:
                raise RenderError('Rendered video could not be verified.')
        finally:
            verify.release()
        os.link(encoded_path, dest)
        return RenderResult(dest, count, sorted(timeline, key=lambda item: (item['frame_index'], item['track_id'])))
    except RenderError:
        raise
    except Exception as exc:
        raise RenderError('Video rendering failed; no output was published.') from exc
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        if encoded_path is not None:
            encoded_path.unlink(missing_ok=True)
