"""Adversarial tests for the red-bucket (roof1) accuracy fix.

Rule table: strict agreement uses risk R1..R7; soft single-witness water
uses R9=HIGH (one witness holds=yes+water=yes+covered=no, other not
contradicting); soft open-container without confirmed water uses R8=medium.
Direct yes-vs-no contradiction stays REVIEW. All-unsure stays REVIEW.
Never 'low' by default.
"""
import unittest
from types import SimpleNamespace

from app.pipeline.decide import Witness, _soft_decision, aggregate_witness_frames, decide_track


def _attr(hw='unsure', cov='unsure', wv='unsure', rain='unsure', ori='unsure'):
    return {'holds_water': hw, 'covered': cov, 'water_visible': wv,
            'exposed_to_rain': rain, 'orientation': ori,
            'object_label': 'red bucket', 'evidence': 'dark liquid inside'}


TRACK = {'track_id': 'bucket:0', 'concept': 'bucket', 'frames': [{'score': 0.9}]}


def _ev(ncrops=3):
    crops = [b'crop%d' % i for i in range(ncrops)]
    return SimpleNamespace(crop_jpeg=crops[0], frame_jpeg=b'frame', mask_png=b'm',
                           timestamp_seconds=0.0, frame_index=0, alt_crops=tuple(crops[1:]))


class RedBucketTests(unittest.TestCase):
    def test_redbucket_a_unsure_b_yes_is_high_r9(self):
        # A: blurry -> all unsure (but evidence text mentions dark liquid).
        # B: clear view -> holds=yes, water=yes, covered=no => HIGH R9.
        a = Witness('pa', 'ma', lambda img, mime: _attr())
        b = Witness('pb', 'mb', lambda img, mime: _attr('yes', 'no', 'yes', 'yes', 'upright'))
        out = decide_track('v.mp4', TRACK, a, b, extractor=lambda v, t: _ev(1))
        self.assertEqual((out['risk'], out['rule_id']), ('high', 'R9'))
        self.assertIn('witness_a_frames', out)
        self.assertIn('witness_b_support', out)

    def test_contradiction_stays_review(self):
        a = Witness('pa', 'ma', lambda img, mime: _attr('yes', 'no', 'yes', 'yes', 'upright'))
        b = Witness('pb', 'mb', lambda img, mime: _attr('no', 'no', 'no', 'unsure', 'upright'))
        out = decide_track('v.mp4', TRACK, a, b, extractor=lambda v, t: _ev(1))
        self.assertEqual(out['risk'], 'review')

    def test_blurry_all_unsure_stays_review(self):
        a = Witness('pa', 'ma', lambda img, mime: _attr())
        b = Witness('pb', 'mb', lambda img, mime: _attr())
        out = decide_track('v.mp4', TRACK, a, b, extractor=lambda v, t: _ev(1))
        self.assertEqual(out['risk'], 'review')

    def test_never_low_by_default(self):
        for aa, bb in [(_attr(), _attr()),
                       (_attr('yes', 'no', 'unsure'), _attr()),
                       (_attr('yes', 'no', 'yes'), _attr('no', 'no', 'no'))]:
            a = Witness('pa', 'ma', lambda img, mime, _v=dict(aa): dict(_v))
            b = Witness('pb', 'mb', lambda img, mime, _v=dict(bb): dict(_v))
            out = decide_track('v.mp4', TRACK, a, b, extractor=lambda v, t: _ev(1))
            self.assertNotEqual(out['risk'], 'low')

    def test_multiframe_yes_wins_and_audit_trail(self):
        frames = [_attr(), _attr('unsure', 'unsure', 'unsure'), _attr('yes', 'no', 'yes', 'yes', 'upright')]
        agg, support = aggregate_witness_frames(frames)
        self.assertEqual(agg['holds_water'], 'yes')
        self.assertEqual(agg['water_visible'], 'yes')
        self.assertEqual(support, 2)
        # unsure stays unsure with no conclusive frame
        agg2, sup2 = aggregate_witness_frames([_attr(), _attr()])
        self.assertEqual(agg2['holds_water'], 'unsure')
        self.assertIsNone(sup2)

    def test_soft_r9_direct(self):
        a = _attr()
        b = _attr('yes', 'no', 'yes', 'yes', 'upright')
        self.assertEqual(_soft_decision(a, b, {}, 0.9, ('review', 'R1')), ('high', 'R9'))
        c = _attr('no', 'no', 'no', 'unsure', 'upright')
        self.assertEqual(_soft_decision(b, c, {}, 0.9, ('review', 'R1')), ('review', 'R1'))


if __name__ == '__main__':
    unittest.main()
