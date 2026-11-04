import json
import unittest

from app.pipeline.witness import describe_container
from app.pipeline import witness

VALID = {'holds_water': 'yes', 'covered': 'no', 'water_visible': 'unsure', 'exposed_to_rain': 'yes', 'orientation': 'upright', 'object_label': 'bucket', 'evidence': 'Open bucket.'}

class FakeClient:
    def __init__(self, responses): self.responses, self.calls = list(responses), []
    def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        value = self.responses.pop(0)
        if isinstance(value, Exception): raise value
        return {'choices': [{'message': {'content': value}}]}

class WitnessTest(unittest.TestCase):
    def test_prompt_covers_water_sources(self):
        self.assertIn('container or water source', witness._PROMPT)

    def test_valid_description_and_embedded_image(self):
        client = FakeClient([json.dumps(VALID)])
        self.assertEqual(describe_container(client, b'image', 'image/jpeg'), VALID)
        content = client.calls[0][0][1]['content']
        self.assertTrue(content[1]['image_url']['url'].startswith('data:image/jpeg;base64,'))
        self.assertEqual(len(client.calls), 1)

    def test_invalid_is_retried_and_normalized_to_unsure(self):
        client = FakeClient(['not json', json.dumps({**VALID, 'covered': 'perhaps'})])
        result = describe_container(client, b'image', 'image/jpeg')
        self.assertEqual(result['covered'], 'unsure')
        self.assertEqual(result['holds_water'], 'unsure')
        self.assertEqual(len(client.calls), 2)
        self.assertNotEqual(client.calls[0], client.calls[1])

    def test_provider_failure_returns_unsure(self):
        client = FakeClient([RuntimeError('failed'), RuntimeError('failed')])
        self.assertEqual(describe_container(client, b'image', 'image/jpeg')['orientation'], 'unsure')

    def test_budget_accounting_failure_is_not_silently_retried(self):
        from app.pipeline.budget_runner import BudgetAccountingError
        client = FakeClient([BudgetAccountingError('usage unavailable', 'reservation-1')])
        with self.assertRaises(BudgetAccountingError):
            describe_container(client, b'image', 'image/jpeg')
        self.assertEqual(len(client.calls), 1)
