import json
import unittest
from types import SimpleNamespace
from app.pipeline.scout_client import scout_scene
from app.pipeline import scout_client

VALID = {'scene': {'setting':'rooftop','notes':'Outdoor'}, 'extra_concepts':[{'noun':'plant saucer','why':'collects water'}]}
class Client:
    def __init__(self, answers): self.answers=list(answers); self.calls=[]
    def chat(self, messages, **opts):
        self.calls.append((messages,opts))
        answer=self.answers.pop(0)
        if isinstance(answer, Exception): raise answer
        return {'choices':[{'message':{'content':answer}}]}

class ScoutClientTests(unittest.TestCase):
    def test_valid_response_and_private_jpeg(self):
        client=Client([json.dumps(VALID)])
        self.assertEqual(scout_scene(client,[SimpleNamespace(jpeg_bytes=b'image')]),VALID)
        parts=client.calls[0][0][1]['content']
        self.assertTrue(parts[1]['image_url']['url'].startswith('data:image/jpeg;base64,'))

    def test_prompt_asks_for_water_sources(self):
        text = scout_client._PROMPT.lower()
        self.assertIn('water source', text)
        self.assertIn('puddle', text)

    def test_retry_then_fallback_to_none(self):
        client=Client(['malformed','{}'])
        self.assertIsNone(scout_scene(client,[SimpleNamespace(jpeg_bytes=b'image')]))
        self.assertEqual(len(client.calls),2)
        self.assertNotEqual(client.calls[0],client.calls[1])

    def test_no_frames_means_no_provider_request(self):
        client=Client([])
        self.assertIsNone(scout_scene(client,[]))
        self.assertEqual(client.calls,[])

    def test_budget_accounting_failure_does_not_fall_back(self):
        from app.pipeline.budget_runner import BudgetAccountingError
        client=Client([BudgetAccountingError('usage unavailable', 'reservation-1')])
        with self.assertRaises(BudgetAccountingError):
            scout_scene(client,[SimpleNamespace(jpeg_bytes=b'image')])
        self.assertEqual(len(client.calls),1)
