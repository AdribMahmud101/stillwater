import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.pipeline.privacy import PrivacyError, blur_video


class PrivacyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'raw.mp4'
        writer = cv2.VideoWriter(str(self.source), cv2.VideoWriter_fourcc(*'mp4v'), 10, (80, 80))
        self.assertTrue(writer.isOpened())
        rng = np.random.default_rng(4)
        for _ in range(3):
            writer.write(rng.integers(0, 255, (80, 80, 3), dtype=np.uint8))
        writer.release()

    def test_blurs_masks_and_writes_all_frames(self):
        masks = [np.ones((80, 80), dtype=np.uint8) for _ in range(3)]
        result = blur_video(self.source, self.root / 'private.mp4', masks)
        self.assertEqual(result.frame_count, 3)
        cap1, cap2 = cv2.VideoCapture(str(self.source)), cv2.VideoCapture(str(result.path))
        ok1, raw = cap1.read()
        ok2, private = cap2.read()
        cap1.release(); cap2.release()
        self.assertTrue(ok1 and ok2)
        self.assertLess(float(private.std()), float(raw.std()) / 2)

    def test_missing_extra_or_malformed_masks_never_publish(self):
        mask = np.zeros((80, 80), dtype=np.uint8)
        for masks in ([mask], [mask] * 4, [mask, np.zeros((2, 2), dtype=np.uint8), mask], [mask, None, mask]):
            out = self.root / 'private.mp4'
            with self.subTest(count=len(masks)), self.assertRaises(PrivacyError):
                blur_video(self.source, out, masks)
            self.assertFalse(out.exists())
            self.assertFalse(list(self.root.glob('.privacy-*')))

    def test_empty_mask_is_explicit_and_existing_output_is_not_overwritten(self):
        masks = [np.zeros((80, 80), dtype=np.uint8) for _ in range(3)]
        out = self.root / 'private.mp4'
        out.write_bytes(b'existing')
        with self.assertRaises(PrivacyError):
            blur_video(self.source, out, masks)
        self.assertEqual(out.read_bytes(), b'existing')
