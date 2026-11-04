import json
import unittest
from app.event_store import EventStore, EventStoreError
from app.events import make_event

SCAN = '1a845e51-9b5e-4e03-bd79-dbb90ac7f5ed'
OTHER = '2a845e51-9b5e-4e03-bd79-dbb90ac7f5ed'


class EventStoreTests(unittest.TestCase):
    def test_stream_existing_events_and_close(self):
        store = EventStore(max_events=4, max_scans=2)
        store.publish(make_event(SCAN, 'track', 'start'))
        store.publish(make_event(SCAN, 'track', 'done'))
        store.close(SCAN)
        chunks = list(store.stream(SCAN, heartbeat_seconds=.01))
        self.assertEqual(len(chunks), 2)
        data = json.loads(chunks[0].split('data: ',1)[1].strip())
        self.assertEqual(data['kind'], 'start')
        self.assertNotIn('payload', data)

    def test_rejects_raw_fields_and_limits(self):
        store = EventStore(max_events=1, max_scans=1)
        event = make_event(SCAN, 'privacy', 'error')
        with self.assertRaises(EventStoreError): store.publish({**event, 'payload': {'key': 'secret'}})
        store.publish(event)
        with self.assertRaises(EventStoreError): store.publish(make_event(OTHER, 'track', 'start'))
        store.close(SCAN)
        with self.assertRaises(EventStoreError): store.publish(event)

    def test_bounded_history_signals_gap(self):
        store = EventStore(max_events=1)
        store.publish(make_event(SCAN, 'track', 'start'))
        store.publish(make_event(SCAN, 'track', 'done'))
        store.close(SCAN)
        chunks = list(store.stream(SCAN, after=0, heartbeat_seconds=.01))
        self.assertIn('event: gap', chunks[0])
        self.assertIn('"kind":"done"', chunks[1])

    def test_heartbeat_then_later_event(self):
        store = EventStore()
        stream = store.stream(SCAN, heartbeat_seconds=.01)
        self.assertEqual(next(stream), ': heartbeat\n\n')
        store.publish(make_event(SCAN, 'ingest', 'start'))
        self.assertIn('event: stage', next(stream))
        stream.close()
