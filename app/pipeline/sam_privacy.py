"""Hosted SAM person-mask adapter, requiring a completed segmentation stream.

Unblurred video is uploaded to Meta for segmentation; deletion is attempted
promptly, but a network failure may leave the upload until its one-hour expiry.
Meta documents omitted frames as frames without detected objects, but the
reported processed-frame count is a billing count, not a coverage guarantee.
This adapter is disabled pending separately validated frame coverage evidence.
Detector recall cannot guarantee every person is detected or fully masked.
See https://dev.meta.ai/docs/sam/reading-segmentation and
https://dev.meta.ai/docs/api-reference/responses/schemas.
"""
from pathlib import Path

import cv2
import numpy as np

from app.pipeline.masks import to_dedup_tracks
from app.pipeline.privacy import PrivacyError, blur_video
from app.pipeline.sam_parse import parse_sam_video


def _video_dimensions(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise PrivacyError('Cannot read source video.')
        count = 0
        width = height = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if width is None:
                width, height = w, h
            if (w, h) != (width, height):
                raise PrivacyError('Video frame dimensions changed.')
            count += 1
        if not count:
            raise PrivacyError('Video has no readable frames.')
        return width, height, count
    finally:
        cap.release()


def sam_privacy(source, destination, client):
    """Fail closed until per-frame coverage has been independently validated."""
    raise PrivacyError('SAM video frame coverage is unverified; privacy output disabled.')


def _sam_privacy_unverified(source, destination, client):
    """Upload raw normalized video, segment people, delete upload, blur locally.

    The client must return only a successfully completed stream and raise on
    failures. A deletion failure also stops publication of privacy output.
    """
    source, destination = Path(source), Path(destination)
    width, height, count = _video_dimensions(source)
    file_id = client.upload_video(source.read_bytes())
    try:
        result = client.segment_video_file_with_usage(file_id, 'person')
        if not isinstance(result, dict) or not isinstance(result.get('text'), str):
            raise ValueError('SAM result is invalid')  # noqa: TRY004
        tracks = parse_sam_video(result['text'], 'person')
        decoded = to_dedup_tracks(tracks, width, height)
        by_frame = {}
        for track in decoded:
            for frame in track['frames']:
                index = frame['frame']
                if type(index) is not int or not 0 <= index < count:
                    raise ValueError('SAM frame index out of bounds')
                by_frame.setdefault(index, set()).update(frame['mask'])
    finally:
        client.delete_file(file_id)

    def frame_masks():
        for index in range(count):
            mask = np.zeros((height, width), dtype=np.uint8)
            for x, y in by_frame.get(index, ()):
                mask[y, x] = 1
            yield mask
    return blur_video(source, destination, frame_masks())
