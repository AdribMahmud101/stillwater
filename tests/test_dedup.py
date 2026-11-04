import time
import unittest

from app.pipeline.dedup import deduplicate_tracks


def rect(x, y, width, height):
    return frozenset((i, j) for i in range(x, x + width) for j in range(y, y + height))


def track(id, concept, mask, count=3, score=.8):
    return {'track_id': id, 'concept': concept, 'frames': [
        {'frame': i, 'mask': mask, 'score': score} for i in range(count)]}


class DedupTests(unittest.TestCase):
    def test_cross_concept_overlap_merges_and_preserves_aliases(self):
        a = track('a', 'bucket', rect(0, 0, 10, 10), score=.9)
        b = track('b', 'container', rect(1, 0, 10, 10), score=.7)
        result = deduplicate_tracks([a, b])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['concept'], 'bucket')
        self.assertEqual(result[0]['aliases'], ['container'])
        self.assertEqual(set(result[0]['source_track_ids']), {'a', 'b'})

    def test_distinct_and_same_concept_tracks_remain_separate(self):
        a = track('a', 'bucket', rect(0, 0, 10, 10))
        b = track('b', 'container', rect(7, 0, 10, 10))
        c = track('c', 'bucket', rect(0, 0, 10, 10))
        self.assertEqual(len(deduplicate_tracks([a, b, c])), 3)

    def test_merge_needs_at_least_half_shared_frames(self):
        a = track('a', 'bucket', rect(0, 0, 10, 10), 4)
        b = track('b', 'pot', rect(30, 30, 10, 10), 4)
        b['frames'][0]['mask'] = a['frames'][0]['mask']
        self.assertEqual(len(deduplicate_tracks([a, b])), 2)
        b['frames'][1]['mask'] = a['frames'][1]['mask']
        self.assertEqual(len(deduplicate_tracks([a, b])), 1)

    def test_empty_short_low_score_and_disjoint_frames(self):
        self.assertEqual(deduplicate_tracks([]), [])
        self.assertEqual(deduplicate_tracks([track('a', 'bucket', rect(0, 0, 1, 1), 1)]), [])
        self.assertEqual(deduplicate_tracks([track('a', 'bucket', rect(0, 0, 1, 1), score=.34)]), [])
        a = track('a', 'bucket', rect(0, 0, 1, 1))
        b = track('b', 'pot', rect(0, 0, 1, 1))
        for frame in b['frames']:
            frame['frame'] += 10
        self.assertEqual(len(deduplicate_tracks([a, b])), 2)

    def test_many_tracks_are_fast(self):
        tracks = [track(str(i), 'bucket', rect(i, 0, 2, 2)) for i in range(100)]
        start = time.monotonic()
        self.assertEqual(len(deduplicate_tracks(tracks)), 100)
        self.assertLess(time.monotonic() - start, 1)
