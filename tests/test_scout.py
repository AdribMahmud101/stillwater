import json
import unittest

from app.pipeline.scout import BASE_CONCEPTS, scout_concepts, validate_scout


def response(items):
    return json.dumps({'scene': {'setting': 'rooftop', 'notes': 'roof'}, 'extra_concepts': [{'noun': x, 'why': 'holds rain'} for x in items]})


class ScoutTests(unittest.TestCase):
    def test_base_concepts_include_water_sources(self):
        self.assertEqual(tuple(BASE_CONCEPTS), ('tire', 'bucket', 'flower pot', 'plastic container', 'plastic drum', 'puddle', 'standing water', 'drain'))
        self.assertNotIn('bottle', BASE_CONCEPTS)

    def test_normalizes_filters_deduplicates_and_limits(self):
        raw = response(['  Coconut Shell ', 'COCONUT SHELL', 'bucket', 'water tank', 'plant saucer', 'tarp', 'wheelbarrow', 'bottle', 'pond', 'ditch'])
        result = scout_concepts(raw)
        self.assertEqual(result, list(BASE_CONCEPTS) + ['coconut shell', 'water tank'])
        self.assertLessEqual(len(result), 10)

    def test_water_source_nouns_are_accepted(self):
        raw = response(['pond', 'ditch'])
        result = scout_concepts(raw)
        self.assertEqual(result, list(BASE_CONCEPTS) + ['pond', 'ditch'])
        # bottle is no longer base but remains an allowed extra noun
        raw2 = response(['bottle', 'pond'])
        result2 = scout_concepts(raw2)
        self.assertEqual(result2, list(BASE_CONCEPTS) + ['bottle', 'pond'])

    def test_extra_cap_keeps_total_at_ten(self):
        raw = response(['pond', 'ditch', 'gutter', 'coconut shell', 'water tank'])
        result = scout_concepts(raw)
        self.assertEqual(len(result), 10)
        self.assertLessEqual(len(result), 10)

    def test_invalid_responses_fall_back_and_can_report_error_for_retry(self):
        for raw in ['not json', '{}', json.dumps({'scene': {}, 'extra_concepts': []}),
                    json.dumps({'scene': {'setting': 'rooftop', 'notes': 'x'}, 'extra_concepts': [{'noun': 'tire'}]})]:
            with self.subTest(raw=raw):
                self.assertEqual(scout_concepts(raw), list(BASE_CONCEPTS))
                with self.assertRaises(ValueError): validate_scout(raw)

    def test_rejects_bad_scene_and_excessive_fields(self):
        raw = {'scene': {'setting': 'moon', 'notes': 'x'}, 'extra_concepts': []}
        with self.assertRaises(ValueError): validate_scout(raw)
        raw['scene']['setting'] = 'yard'
        raw['scene']['notes'] = 'x' * 121
        with self.assertRaises(ValueError): validate_scout(raw)
