import json
import unittest

from app.pipeline.verify import merge_witnesses, parse_witness

BASE = {'holds_water': 'yes', 'covered': 'no', 'water_visible': 'yes',
        'orientation': 'upright', 'exposed_to_rain': 'yes',
        'object_label': 'red bucket', 'evidence': 'Open container'}


class VerifyTests(unittest.TestCase):
    def test_valid_json(self):
        attrs, valid = parse_witness(json.dumps(BASE))
        self.assertTrue(valid)
        self.assertEqual(attrs, BASE)

    def test_malformed_and_missing_never_agree(self):
        for raw in ['prose', '{}', json.dumps({**BASE, 'covered': 'perhaps'}),
                    json.dumps({**BASE, 'extra': 'x'}), json.dumps({**BASE, 'object_label': 'four words in label'})]:
            with self.subTest(raw=raw):
                _attrs, valid = parse_witness(raw)
                self.assertFalse(valid)
                merged, agreement = merge_witnesses(raw, json.dumps(BASE))
                self.assertFalse(agreement)
                self.assertEqual(merged['holds_water'], 'unsure')

    def test_agreement_and_disagreement_merge(self):
        merged, agreement = merge_witnesses(json.dumps(BASE), json.dumps(BASE))
        self.assertTrue(agreement)
        self.assertEqual(merged, BASE)
        other = {**BASE, 'covered': 'yes', 'water_visible': 'no', 'evidence': 'Different'}
        merged, agreement = merge_witnesses(json.dumps(BASE), json.dumps(other))
        self.assertFalse(agreement)
        self.assertEqual(merged['covered'], 'unsure')
        self.assertEqual(merged['water_visible'], 'unsure')
        self.assertEqual(merged['evidence'], '')

    def test_unsure_critical_attribute_is_disagreement(self):
        raw = json.dumps({**BASE, 'covered': 'unsure'})
        self.assertFalse(merge_witnesses(raw, raw)[1])
