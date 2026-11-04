import unittest
from pathlib import Path

from app.pipeline.masks import to_dedup_tracks
from app.pipeline.sam_parse import parse_sam_video


class MaskTests(unittest.TestCase):
    def test_real_synthetic_video_masks_are_full_frame(self):
        raw = Path('tests/fixtures/sam_synthetic_video.txt').read_text()
        tracks = parse_sam_video(raw, 'bucket')
        result = to_dedup_tracks(tracks, 128, 128)
        self.assertEqual(len(result), 1)
        self.assertGreater(len(result[0]['frames']), 1)
        pixels = result[0]['frames'][0]['mask']
        self.assertTrue(pixels)
        self.assertTrue(all(0 <= x < 128 and 0 <= y < 128 for x, y in pixels))
        self.assertIsNone(result[0]['frames'][0]['score'])

    def test_rejects_out_of_bounds_and_malformed_masks(self):
        track = {'track_id': '0', 'concept': 'bucket', 'frames': [{'frame_index': 0, 'mask_offset': [127, 127], 'mask_rle': {'size': [2, 2], 'counts': '4'}, 'score': None}]}
        with self.assertRaises(ValueError): to_dedup_tracks([track], 128, 128)
        track['frames'][0]['mask_offset'] = [0, 0]
        track['frames'][0]['mask_rle']['counts'] = 'invalid!'
        with self.assertRaises(ValueError): to_dedup_tracks([track], 128, 128)
