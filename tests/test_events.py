import unittest
from app.events import make_event, EventError

SCAN = '1a845e51-9b5e-4e03-bd79-dbb90ac7f5ed'


class EventTests(unittest.TestCase):
    def test_fact_only_event_preserves_reported_values(self):
        event = make_event(SCAN, 'track', 'done', model='sam-3.1', provider='meta', latency_ms=82, cost_usd='0.001', tokens_in=3)
        self.assertEqual(event['latency_ms'], 82)
        self.assertEqual(event['cost_usd'], '0.001')
        self.assertNotIn('tokens_out', event)
        self.assertNotIn('payload', event)

    def test_raw_messages_and_unknown_fields_are_not_accepted(self):
        for changes in ({'message': 'secret'}, {'payload': {'api_key': 'key'}}, {'model': 'secret-key'}, {'provider': 'http://secret'}, {'latency_ms': -1}, {'cost_usd': 'NaN'}):
            with self.subTest(changes=changes), self.assertRaises(EventError):
                make_event(SCAN, 'track', 'done', **changes)

    def test_untrusted_identifiers_and_event_names_are_rejected(self):
        for scan, stage, kind in [('key', 'track', 'done'), (SCAN, 'raw text', 'done'), (SCAN, 'track', 'my secret')]:
            with self.subTest(scan=scan, stage=stage, kind=kind), self.assertRaises(EventError):
                make_event(scan, stage, kind)

    def test_values_must_be_finite_and_real(self):
        for changes in ({'latency_ms': True}, {'gpu_seconds': float('nan')}, {'tokens_in': 2.5}, {'cost_usd': '-1'}):
            with self.subTest(changes=changes), self.assertRaises(EventError):
                make_event(SCAN, 'track', 'done', **changes)
