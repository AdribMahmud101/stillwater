import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from app.events import EventError, make_event
from app.event_store import EventStore
from app.pipeline.frame_privacy import blur_per_frame
from app.pipeline.sam_track import track_video

SCAN = '1a845e51-9b5e-4e03-bd79-dbb90ac7f5ed'

class ProgressTests(unittest.TestCase):
    def test_event_validation_and_store(self):
        event = make_event(SCAN, 'privacy', 'progress', done=1, total=2, unit='frames', fps=30.0)
        store = EventStore()
        store.publish(event)
        self.assertIn('"done":1', next(store.stream(SCAN)))
        for fields in ({'done': -1}, {'done': True}, {'total': 0}, {'total': True},
                       {'done': 3, 'total': 2}, {'unit': 'seconds'}, {'fps': 0}):
            with self.subTest(fields=fields), self.assertRaises(EventError):
                make_event(SCAN, 'track', 'progress', **fields)

    @patch('app.pipeline.frame_privacy._mask_from_result', side_effect=lambda t, w, h: np.zeros((h, w), dtype=np.uint8))
    def test_privacy_progress(self, _mask):
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d)/'input.mp4', Path(d)/'output.mp4'
            writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*'mp4v'), 10, (64, 48))
            for _ in range(5):
                writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
            writer.release()
            updates = []
            blur_per_frame(src, dst, lambda _: {'completed': True, 'text': ''}, progress=lambda d, t: updates.append((d, t)))
            self.assertEqual(updates[-1], (5, 5))
            self.assertEqual([x[0] for x in updates], sorted(x[0] for x in updates))
            self.assertTrue(all(t == 5 for _, t in updates))

    def test_tracking_progress(self):
        class Client:
            def upload_video(self, video): return 'id'
            def segment_video_file(self, file_id, concept): return 'empty'
            def delete_file(self, file_id): pass
        updates = []
        with patch('app.pipeline.sam_track.proxy_enabled', return_value=False), patch('app.pipeline.sam_track.parse_sam_video', return_value=[]):
            track_video(Client(), b'video', ['a', 'b', 'c'], 10, 10, progress=lambda d, t: updates.append((d, t)))
        self.assertEqual(updates, [(1, 3), (2, 3), (3, 3)])

class ProductionProgressTests(unittest.TestCase):
    def test_throttling_and_callback_failures(self):
        from decimal import Decimal
        from test_production import FakeLedger, _make_settings
        from app.pipeline.production import build_runner
        from app.pipeline.providers import get_current_progress
        import uuid

        with tempfile.TemporaryDirectory() as d, patch.dict('os.environ', {'SCANNING_ENABLED': '1', 'MODEL_API_KEY': 'key', 'BASETEN_API_KEY': 'key'}):
            source = Path(d) / 'source.mp4'
            source.write_bytes(b'video')
            def copy(src, dst, *args): Path(dst).write_bytes(b'video')
            def privacy(src, dst, *args):
                from app.pipeline.providers import get_current_scan_meter
                copy(src, dst)
                get_current_scan_meter().add(Decimal("0.01"))
            def track(private, concepts):
                callback = get_current_progress()
                for i in range(100): callback('track', i, 100, 'concepts')
                return []
            clients = {'ledger': FakeLedger(), 'sam_client': object(), 'witness_a': object(),
                       'witness_b': object(), 'ingest': copy, 'privacy': privacy,
                       'scout': lambda _: ['box'], 'track': track, 'verify': lambda *_: [],
                       'act': lambda _: [], 'upper_bound_usd': Decimal('1'), 'cost_usd': Decimal('0.1')}
            runner = build_runner(_make_settings(), clients=clients)
            events = []
            with patch('app.pipeline.production.time.monotonic', return_value=10):
                result = runner(source, {'id': str(uuid.uuid4())}, event_callback=events.append)
            self.assertTrue(result['privacy_verified'])
            self.assertEqual(len([e for e in events if e['stage'] == 'track' and e['kind'] == 'progress']), 1)
            def broken(_): raise ValueError('subscriber failed')
            with self.assertLogs('app.pipeline.production', level='ERROR'):
                result = runner(source, {'id': str(uuid.uuid4())}, event_callback=broken)
            self.assertTrue(result['privacy_verified'])
