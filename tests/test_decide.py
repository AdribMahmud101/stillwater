import unittest
from types import SimpleNamespace

from app.pipeline.decide import DecisionError, Witness, decide_track

ATTR = {'holds_water':'yes','covered':'no','water_visible':'yes','exposed_to_rain':'yes','orientation':'upright','object_label':'bucket','evidence':'Open bucket.'}
TRACK = {'track_id':'bucket:0', 'concept':'bucket', 'frames':[{'score':.9}]}
EVIDENCE = SimpleNamespace(crop_jpeg=b'crop', frame_jpeg=b'frame', mask_png=b'mask', timestamp_seconds=1.5, frame_index=15)

class DecisionTest(unittest.TestCase):
    def test_agreement_produces_rule_and_order(self):
        a=Witness('baseten','spark',lambda image,mime: ATTR.copy())
        b=Witness('modal','glimmer',lambda image,mime: ATTR.copy())
        out=decide_track('private.mp4',TRACK,a,b,extractor=lambda video,track:EVIDENCE)
        self.assertEqual(out['checks']['a']['label'], 'spark')
        self.assertEqual(out['checks']['b']['holds_water'], 'yes')
        self.assertNotIn('evidence', out['checks']['a'])
        self.assertEqual(out['risk'],'high')
        self.assertEqual(out['rule_id'],'R3')
        self.assertEqual(out['work_order'].action_key,'empty_scrub')
        self.assertIs(out['evidence'], EVIDENCE)

    def test_duplicate_provider_or_model_rejected(self):
        a=Witness('same','spark',lambda image,mime: ATTR)
        for b in (Witness('same','glimmer',lambda image,mime: ATTR), Witness('other','spark',lambda image,mime: ATTR)):
            with self.assertRaises(DecisionError): decide_track('private.mp4',TRACK,a,b,extractor=lambda *a:EVIDENCE)

    def test_disagreement_forces_review(self):
        a=Witness('a','spark',lambda image,mime: ATTR.copy())
        b=Witness('b','glimmer',lambda image,mime: {**ATTR, 'covered':'yes'})
        out=decide_track('private.mp4',TRACK,a,b,extractor=lambda *a:EVIDENCE)
        self.assertEqual(out['risk'], 'review')
        self.assertEqual(out['work_order'].action_key,'inspect')


class SoftDecisionTests(unittest.TestCase):
    A = {'holds_water': 'yes', 'covered': 'no', 'water_visible': 'no', 'exposed_to_rain': 'unsure',
         'orientation': 'upright', 'object_label': 'bucket', 'evidence': 'x'}

    def soft(self, a, b, score=0.9):
        from app.pipeline.decide import _soft_decision
        return _soft_decision(a, b, {}, score, ('review', 'R1'))

    def test_one_definite_one_unsure_is_medium_r8(self):
        b = {**self.A, 'holds_water': 'unsure'}
        self.assertEqual(self.soft(self.A, b), ('medium', 'R8'))

    def test_contradiction_stays_review(self):
        b = {**self.A, 'holds_water': 'no'}
        self.assertEqual(self.soft(self.A, b), ('review', 'R1'))
        b = {**self.A, 'covered': 'yes'}
        self.assertEqual(self.soft(self.A, b), ('review', 'R1'))

    def test_both_unsure_or_low_detection_or_inverted_stay_review(self):
        u = {**self.A, 'holds_water': 'unsure'}
        self.assertEqual(self.soft(u, u), ('review', 'R1'))
        self.assertEqual(self.soft(self.A, {**self.A, 'holds_water': 'unsure'}, score=0.3), ('review', 'R1'))
        self.assertEqual(self.soft(self.A, {**self.A, 'orientation': 'inverted'}), ('review', 'R1'))

    def test_soft_rule_never_returns_high_or_low(self):
        for hw in ('yes', 'no', 'unsure'):
            for cv in ('yes', 'no', 'unsure'):
                a = {**self.A, 'holds_water': hw, 'covered': cv}
                b = {**self.A, 'holds_water': 'unsure', 'covered': 'unsure'}
                self.assertIn(self.soft(a, b)[0], ('medium', 'review'))
