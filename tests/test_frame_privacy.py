import unittest
from unittest.mock import patch
import numpy as np
from app.pipeline.frame_privacy import frame_masks, FramePrivacyError


class FramePrivacyTests(unittest.TestCase):
    def setUp(self): self.frames = [np.zeros((4, 5, 3), dtype=np.uint8) for _ in range(3)]

    @patch('app.pipeline.frame_privacy.to_dedup_tracks')
    @patch('app.pipeline.frame_privacy.parse_sam_video')
    def test_each_frame_requested_and_masks_merged(self, parse, decode):
        calls = []
        def detector(image): calls.append(image); return {'completed': True, 'text': ''}
        parse.return_value = []
        decode.return_value = [{'frames': [{'frame': 0, 'mask': {(1, 2)}}]}]
        masks = list(frame_masks(self.frames, detector))
        self.assertEqual(len(calls), 3)
        self.assertTrue(all(x.startswith(b'\xff\xd8') for x in calls))
        self.assertTrue(all(mask[2, 1] == 1 for mask in masks))

    def test_failed_or_incomplete_results_fail_closed(self):
        for result in [None, {}, {'completed': False, 'text': ''}, {'completed': True, 'text': None}]:
            with self.subTest(result=result), self.assertRaises(FramePrivacyError):
                list(frame_masks(self.frames, lambda _: result))

    @patch('app.pipeline.frame_privacy.to_dedup_tracks', return_value=[{'frames': [{'frame': 1, 'mask': set()}]}])
    @patch('app.pipeline.frame_privacy.parse_sam_video', return_value=[])
    def test_wrong_frame_index_is_rejected(self, parse, decode):
        with self.assertRaises(FramePrivacyError):
            list(frame_masks(self.frames, lambda _: {'completed': True, 'text': ''}))

    def _make_video(self, path, n=10):
        import cv2
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (64, 48))
        for i in range(n):
            w.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
        w.release()

    @patch('app.pipeline.frame_privacy._mask_from_result', side_effect=lambda t, w, h: np.zeros((h, w), dtype=np.uint8))
    def test_parallel_blur_sends_every_frame_once_and_writes_all(self, _m):
        import os, tempfile, threading, cv2
        from pathlib import Path
        from app.pipeline.frame_privacy import blur_per_frame
        os.environ['PRIVACY_WORKERS'] = '4'
        lock, sent = threading.Lock(), []
        def detector(jpeg):
            with lock:
                sent.append(jpeg)
            return {'completed': True, 'text': ''}
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d) / 's.mp4', Path(d) / 'o.mp4'
            self._make_video(src)
            blur_per_frame(src, dst, detector)
            cap = cv2.VideoCapture(str(dst)); n = 0
            while cap.read()[0]: n += 1
        self.assertEqual(len(sent), 10)
        self.assertEqual(n, 10)

    def test_detector_failure_publishes_nothing(self):
        import os, tempfile
        from pathlib import Path
        from app.pipeline.frame_privacy import blur_per_frame
        os.environ['PRIVACY_WORKERS'] = '4'
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d) / 's.mp4', Path(d) / 'o.mp4'
            self._make_video(src)
            with self.assertRaises(FramePrivacyError):
                blur_per_frame(src, dst, lambda _: {'completed': False, 'text': ''})
            self.assertFalse(dst.exists())
            self.assertEqual([p.name for p in Path(d).iterdir() if p.name.startswith('.privacy-')], [])
