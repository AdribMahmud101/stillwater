import unittest
from unittest.mock import patch

from app.pipeline.sam_track import track_video_proxy


class ProxyHoldTests(unittest.TestCase):
    def run_proxy(self, step, frames, count=6):
        track = {'track_id': 'bottle:1', 'concept': 'bottle', 'frames': frames}
        with patch('app.pipeline.sam_track.probe_video_bytes', return_value=(8, 8, 10, count)), \
             patch('app.pipeline.sam_track.make_proxy_video', return_value=(b'proxy', 4, 4, step)), \
             patch('app.pipeline.sam_track._track_bytes_direct', return_value=[track]):
            return track_video_proxy(None, b'video', ['bottle'], 8, 8,
                                     min_track_frames=1, det_min=0)

    @staticmethod
    def frame(index, score):
        return {'frame': index, 'score': score, 'mask': {(index, 0)},
                'box': [index, 0, index + 1, 1], 'mask_offset': [index, 0],
                'mask_rle': {'size': [1, 1], 'counts': '01'}}

    def test_step_two_holds_scaled_masks_and_scores_through_last_frame(self):
        tracks = self.run_proxy(2, [self.frame(i, s) for i, s in enumerate((0.4, 0.6, 0.8))])
        self.assertEqual(len(tracks), 1)
        frames = tracks[0]['frames']
        self.assertEqual([f['frame'] for f in frames], list(range(6)))
        self.assertEqual([f['score'] for f in frames], [0.4, 0.4, 0.6, 0.6, 0.8, 0.8])
        self.assertEqual([f['mask'] for f in frames],
                         [{(x, y) for x in (i * 2, i * 2 + 1) for y in (0, 1)}
                          for i in (0, 0, 1, 1, 2, 2)])
        self.assertEqual(frames[-1]['mask_rle'], frames[-2]['mask_rle'])

    def test_step_one_keeps_original_frame_indices(self):
        tracks = self.run_proxy(1, [self.frame(0, 0.4), self.frame(2, 0.8)])
        self.assertEqual([f['frame'] for f in tracks[0]['frames']], [0, 2])

    def test_sparse_track_has_bounded_nearest_holds_and_video_limit(self):
        tracks = self.run_proxy(3, [self.frame(1, 0.4), self.frame(3, 0.8)], count=11)
        frames = tracks[0]['frames']
        self.assertEqual([f['frame'] for f in frames], [1, 2, 3, 4, 5, 7, 8, 9, 10])
        self.assertEqual([f['score'] for f in frames], [0.4] * 5 + [0.8] * 4)


if __name__ == '__main__':
    unittest.main()
