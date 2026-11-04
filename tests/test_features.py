import unittest

import numpy as np

from app.pipeline.features import extract_match_features


class FeatureTests(unittest.TestCase):
    def test_histogram_normalizes_and_position_is_fraction_of_duration(self):
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        rgb[:, :, 0] = 255
        result = extract_match_features(rgb, 5, 20)
        self.assertEqual(result['position'], .25)
        self.assertAlmostEqual(sum(result['histogram']), 1.)
        self.assertEqual(len(result['histogram']), 64)

    def test_mask_excludes_background(self):
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        rgb[0, 0] = (255, 0, 0)
        mask = np.array([[True, False], [False, False]])
        selected = extract_match_features(rgb, 0, 1, mask=mask)
        red = extract_match_features(np.array([[[255, 0, 0]]], dtype=np.uint8), 0, 1)
        self.assertEqual(selected['histogram'], red['histogram'])

    def test_rejects_bad_inputs(self):
        image = np.zeros((2, 2, 3), dtype=np.uint8)
        for time, duration in [(1, 0), (-1, 5), (6, 5), (float('nan'), 5)]:
            with self.subTest(time=time, duration=duration), self.assertRaises(ValueError):
                extract_match_features(image, time, duration)
        with self.assertRaises(ValueError): extract_match_features(image, 0, 1, mask=np.zeros((2, 2), dtype=bool))
        with self.assertRaises(ValueError): extract_match_features(np.zeros((2, 2), dtype=np.uint8), 0, 1)
        with self.assertRaises(ValueError): extract_match_features(image.astype(float), 0, 1)
