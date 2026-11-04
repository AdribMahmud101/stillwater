import itertools
import re
import unittest

from app.pipeline.risk import assign_risk

BASE = {'holds_water': 'yes', 'covered': 'no', 'water_visible': 'no',
        'orientation': 'upright', 'exposed_to_rain': 'no'}


class RiskTests(unittest.TestCase):
    def risk(self, changes=None, score=0.5, agreement=True):
        return assign_risk({**BASE, **(changes or {})}, score, agreement)

    def test_ordered_rules(self):
        cases = [({}, ('medium', 'R5')),
                 ({'water_visible': 'yes'}, ('high', 'R3')),
                 ({'exposed_to_rain': 'yes'}, ('high', 'R4')),
                 ({'covered': 'yes'}, ('low', 'R6')),
                 ({'holds_water': 'no'}, ('low', 'R2')),
                 ({'orientation': 'inverted'}, ('low', 'R2')),
                 ({'holds_water': 'no', 'water_visible': 'yes'}, ('review', 'R1b')),
                 ({'orientation': 'inverted', 'water_visible': 'yes'}, ('review', 'R1b')),
                 ({'holds_water': 'unsure'}, ('review', 'R7'))]
        for changes, expected in cases:
            with self.subTest(changes=changes):
                self.assertEqual(self.risk(changes), expected)
        self.assertEqual(self.risk(score=0.4999), ('review', 'R0'))
        self.assertEqual(self.risk(score=0.5, agreement=False), ('review', 'R1'))
        self.assertEqual(self.risk({'covered': 'yes', 'water_visible': 'yes'}), ('low', 'R6'))

    def test_invalid_fields_and_scores_never_create_confidence(self):
        for attrs in [None, {}, {**BASE, 'holds_water': 'maybe'}, {**BASE, 'extra': 1}]:
            self.assertEqual(assign_risk(attrs, .8, True)[0], 'review')
        for score in [None, float('nan'), float('inf'), -1, 1.1, 10**1000, '0.9', True]:
            self.assertEqual(assign_risk(BASE, score, True)[0], 'review')
        for agreement in [None, 1, 'yes']:
            self.assertEqual(assign_risk(BASE, .8, agreement)[0], 'review')

    def test_exhaustive_combinations_are_total(self):
        tri = ('yes', 'no', 'unsure')
        for values in itertools.product(tri, tri, tri, ('upright', 'inverted', 'on_side', 'unsure'), tri):
            attrs = dict(zip(BASE, values))
            risk, rule = assign_risk(attrs, .8, True)
            self.assertIn(risk, {'high', 'medium', 'low', 'review'})
            self.assertIn(rule, {'R0', 'R1', 'R1b', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7'})
            self.assertTrue(re.match(r'^R\d+[a-z]?$', rule))

    def test_monotonicity_of_classified_results(self):
        # Review is uncertainty, not a severity on the Low/Medium/High scale.
        order = {'low': 0, 'medium': 1, 'high': 2}
        tri = ('yes', 'no', 'unsure')
        for values in itertools.product(tri, tri, tri, ('upright', 'inverted', 'on_side', 'unsure'), tri):
            attrs = dict(zip(BASE, values))
            uncovered = assign_risk({**attrs, 'covered': 'no'}, .8, True)[0]
            covered = assign_risk({**attrs, 'covered': 'yes'}, .8, True)[0]
            if uncovered in order and covered in order:
                self.assertLessEqual(order[covered], order[uncovered], attrs)
            dry = assign_risk({**attrs, 'water_visible': 'no'}, .8, True)[0]
            wet = assign_risk({**attrs, 'water_visible': 'yes'}, .8, True)[0]
            if dry in order and wet in order:
                self.assertGreaterEqual(order[wet], order[dry], attrs)

    def test_uncertain_and_contradictory_inputs(self):
        attrs = {field: 'unsure' for field in BASE}
        self.assertEqual(assign_risk(attrs, .8, True), ('review', 'R7'))
        self.assertEqual(self.risk({'water_visible': 'yes'}, agreement=False), ('review', 'R1'))
        self.assertEqual(self.risk({'holds_water': 'no', 'water_visible': 'yes'}), ('review', 'R1b'))

    def test_puddle_with_visible_water_is_high(self):
        # Standing-water source: holds water, uncovered, water visible -> HIGH.
        for label in ('puddle', 'standing water', 'drain', 'ditch'):
            attrs = {**BASE, 'holds_water': 'yes', 'covered': 'no', 'water_visible': 'yes',
                     'object_label': label}
            risk, rule = assign_risk(attrs, 0.8, True)
            with self.subTest(label=label):
                self.assertEqual(risk, 'high')
                self.assertTrue(re.match(r'^R\d+[a-z]?$', rule))
                self.assertEqual((risk, rule), ('high', 'R3'))

    def test_covered_tarp_with_water_follows_existing_covered_logic(self):
        attrs = {**BASE, 'holds_water': 'yes', 'covered': 'yes', 'water_visible': 'yes',
                 'object_label': 'tarp'}
        risk, rule = assign_risk(attrs, 0.8, True)
        self.assertIn(risk, ('low', 'review'))
        self.assertEqual((risk, rule), ('low', 'R6'))

    def test_contradictory_or_unsure_water_source_is_review(self):
        self.assertEqual(self.risk({'holds_water': 'no', 'water_visible': 'yes'}), ('review', 'R1b'))
        self.assertEqual(self.risk({'holds_water': 'unsure', 'water_visible': 'yes'}), ('review', 'R7'))
        all_unsure = {f: 'unsure' for f in BASE}
        self.assertEqual(assign_risk(all_unsure, 0.8, True)[0], 'review')
        self.assertEqual(assign_risk({**BASE, 'water_visible': 'yes'}, 0.8, False)[0], 'review')

    def test_never_low_by_default(self):
        # Any unsure holds_water / covered / contradictory input must not be low.
        for changes in ({'holds_water': 'unsure'}, {'covered': 'unsure'},
                        {'holds_water': 'unsure', 'water_visible': 'unsure'},
                        {'holds_water': 'no', 'water_visible': 'yes'}):
            with self.subTest(changes=changes):
                self.assertNotEqual(self.risk(changes)[0], 'low')
        # Low requires an affirmative holds==no/inverted or holds==yes+covered==yes.
        tri = ('yes', 'no', 'unsure')
        for values in itertools.product(tri, tri, tri, ('upright', 'inverted', 'on_side', 'unsure'), tri):
            attrs = dict(zip(BASE, values))
            risk, _ = assign_risk(attrs, 0.8, True)
            if risk == 'low':
                self.assertTrue(attrs['holds_water'] == 'no' or attrs['orientation'] == 'inverted'
                                or (attrs['holds_water'] == 'yes' and attrs['covered'] == 'yes'))


if __name__ == '__main__':
    unittest.main()
