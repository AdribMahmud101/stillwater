import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.pipeline.evidence import EvidenceError, extract_evidence


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.video = Path(self.temp.name) / 'private.mp4'
        out = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*'mp4v'), 10, (64, 48))
        self.assertTrue(out.isOpened())
        for n in range(3):
            image = np.full((48, 64, 3), n * 80, dtype=np.uint8)
            out.write(image)
        out.release()

    def test_uses_highest_score_and_returns_real_frame_and_mask(self):
        track = {'track_id': 'bucket-1', 'frames': [
            {'frame': 0, 'mask': {(2, 2)}, 'score': 0.4},
            {'frame': 2, 'mask': {(4, 5), (5, 5), (4, 6)}, 'score': 0.9},
        ]}
        evidence = extract_evidence(self.video, track)
        self.assertEqual(evidence.frame_index, 2)
        self.assertAlmostEqual(evidence.timestamp_seconds, 0.2)
        full = cv2.imdecode(np.frombuffer(evidence.frame_jpeg, np.uint8), cv2.IMREAD_COLOR)
        mask = cv2.imdecode(np.frombuffer(evidence.mask_png, np.uint8), cv2.IMREAD_GRAYSCALE)
        self.assertGreater(full.mean(), 100)
        self.assertEqual(mask.shape, (48, 64))
        self.assertEqual(np.count_nonzero(mask), 3)
        self.assertGreater(len(evidence.crop_jpeg), 0)

    def test_nonexistent_frame_fails(self):
        track = {'track_id': 'x', 'frames': [{'frame': 99, 'mask': {(0, 0)}, 'score': 0.9}]}
        with self.assertRaises(EvidenceError):
            extract_evidence(self.video, track)

    def test_empty_or_out_of_bounds_mask_fails(self):
        for mask in (set(), {(64, 0)}, {(-1, 0)}):
            with self.subTest(mask=mask), self.assertRaises(EvidenceError):
                extract_evidence(self.video, {'track_id': 'x', 'frames': [{'frame': 1, 'mask': mask, 'score': 0.9}]})
