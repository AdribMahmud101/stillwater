import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.pipeline.keyframes import KeyframeError, blur_score, extract_keyframes


class KeyframeTests(unittest.TestCase):
    def test_blur_score_distinguishes_flat_and_sharp(self):
        flat = np.full((80, 80, 3), 128, np.uint8)
        sharp = flat.copy()
        sharp[:, ::2] = 0
        self.assertEqual(blur_score(flat), 0)
        self.assertGreater(blur_score(sharp), 100)

    def test_samples_evenly_and_rejects_blurred_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'private.mp4'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (80, 80))
            self.assertTrue(writer.isOpened())
            pattern = np.zeros((80, 80, 3), np.uint8)
            pattern[:, ::4] = 255
            for i in range(10):
                writer.write(np.full_like(pattern, 100) if i == 0 else pattern)
            writer.release()
            result = extract_keyframes(path, count=3, blur_min=10)
            self.assertEqual([f.frame_index for f in result], [4, 9])
            self.assertAlmostEqual(result[0].timestamp_seconds, 0.4)
            self.assertGreater(len(result[0].jpeg_bytes), 0)

    def test_empty_or_missing_video_and_invalid_options_fail(self):
        with self.assertRaises(KeyframeError):
            extract_keyframes(Path('/missing/input.mp4'))
        with self.assertRaises(ValueError):
            extract_keyframes(Path('/missing/input.mp4'), count=0)
