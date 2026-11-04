import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.pipeline.decide import Witness
from app.pipeline.runner import RunnerError, run_scan

ATTR={'holds_water':'yes','covered':'no','water_visible':'yes','exposed_to_rain':'yes','orientation':'upright','object_label':'bucket','evidence':'open bucket'}
TRACK={'track_id':'0','concept':'bucket','frames':[{'score':.9}]}
EVIDENCE=SimpleNamespace(crop_jpeg=b'crop')

class RunnerTests(unittest.TestCase):
    def test_private_file_used_and_temporary_files_removed(self):
        seen=[]
        def ingest(src,dst): dst.write_bytes(b'normalized')
        def privacy(src,dst,client): dst.write_bytes(b'private')
        def track(client,data,concepts,width,height):
            self.assertEqual(data,b'private'); seen.append(concepts); return [TRACK]
        def render(src, dst, tracks):
            self.assertEqual(src.read_bytes(), b'private')
            self.assertEqual(tracks[0]['risk'], 'high')
            self.assertEqual(tracks[0]['track_id'], TRACK['track_id'])
            dst.write_bytes(b'annotated')
            return SimpleNamespace(path=dst, timeline=[{'track_id': '0', 'frame_index': 0}])
        a=Witness('a','m1',lambda img,mime:ATTR)
        b=Witness('b','m2',lambda img,mime:ATTR)
        with tempfile.TemporaryDirectory() as d:
            raw=Path(d)/'input.mp4'; raw.write_bytes(b'raw')
            result=run_scan(raw, object(),a,b,ingest=ingest,privacy=privacy,track=track,probe=lambda path:SimpleNamespace(width=128,height=128),extractor=lambda *args:EVIDENCE, renderer=render, temp_root=Path(d))
            self.assertEqual(result['private_video'],b'private')
            self.assertEqual(result['annotated_video'], b'annotated')
            self.assertEqual(result['timeline'][0]['track_id'], '0')
            self.assertEqual(result['containers'][0]['risk'],'high')
            self.assertEqual(len(list(Path(d).iterdir())),1)
            self.assertIn('bucket',seen[0])

    def test_privacy_failure_stops_all_later_stages(self):
        calls=[]
        def ingest(src,dst): dst.write_bytes(b'normalized')
        def privacy(*args): raise RuntimeError('secret')
        with tempfile.TemporaryDirectory() as d:
            raw=Path(d)/'input.mp4'; raw.write_bytes(b'raw')
            with self.assertRaises(RunnerError) as exc:
                run_scan(raw, object(),object(),object(),ingest=ingest,privacy=privacy,track=lambda *a:calls.append('track'),temp_root=Path(d))
            self.assertNotIn('secret',str(exc.exception)); self.assertFalse(calls)
            self.assertEqual(len(list(Path(d).iterdir())),1)

    def test_mismatched_track_decision_fails_before_rendering(self):
        calls = []
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d) / 'input.mp4'
            raw.write_bytes(b'raw')
            def copy(src, dst, *args): dst.write_bytes(b'private')
            with patch('app.pipeline.runner.decide_track', return_value={'track_id': 'wrong', 'risk': 'high', 'work_order': None}), self.assertRaises(RunnerError):
                run_scan(raw, object(), object(), object(), ingest=copy, privacy=copy,
                         track=lambda *args: [TRACK], probe=lambda p: SimpleNamespace(width=128, height=128),
                         renderer=lambda *args: calls.append('render'), temp_root=Path(d))
            self.assertEqual(calls, [])

    def test_trace_has_actual_stage_transitions_only(self):
        events = []
        scan_id = '1a845e51-9b5e-4e03-bd79-dbb90ac7f5ed'
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d) / 'input.mp4'; raw.write_bytes(b'raw')
            def copy(src, dst, *args): dst.write_bytes(b'private')
            def render(src, dst, tracks):
                dst.write_bytes(b'rendered'); return SimpleNamespace(path=dst, timeline=[])
            run_scan(raw, object(), object(), object(), ingest=copy, privacy=copy,
                     track=lambda *args: [], probe=lambda p: SimpleNamespace(width=128,height=128),
                     renderer=render, temp_root=Path(d), scan_id=scan_id, event_callback=events.append)
        self.assertEqual([(e['stage'],e['kind']) for e in events],
                         [(s,k) for s in ('ingest','privacy','scout','track','verify','render') for k in ('start','done')])
        self.assertTrue(all(set(e) == {'scan_id','stage','kind','created_at'} for e in events))

    def test_trace_failure_reports_stage_without_error_text(self):
        events = []; scan_id = '1a845e51-9b5e-4e03-bd79-dbb90ac7f5ed'
        with tempfile.TemporaryDirectory() as d:
            raw = Path(d) / 'input.mp4'; raw.write_bytes(b'raw')
            def copy(src, dst): dst.write_bytes(b'private')
            def fail(*args): raise ValueError('secret content')
            with self.assertRaises(RunnerError):
                run_scan(raw, object(), object(), object(), ingest=copy, privacy=fail,
                         temp_root=Path(d), scan_id=scan_id, event_callback=events.append)
        self.assertEqual([(e['stage'],e['kind']) for e in events], [('ingest','start'),('ingest','done'),('privacy','start'),('privacy','error')])
        self.assertNotIn('secret', str(events))

    def test_budget_failure_preserves_reservation_without_exposing_detail(self):
        from app.pipeline.budget_runner import BudgetAccountingError
        events=[]
        with tempfile.TemporaryDirectory() as d:
            raw=Path(d)/'input.mp4'; raw.write_bytes(b'raw')
            def ingest(src,dst): dst.write_bytes(b'normalized')
            def privacy(*args): raise BudgetAccountingError('secret provider detail','a0c3ddf1-3bd1-4afb-9bcc-43dd44a1b8a4')
            with self.assertRaises(RunnerError) as caught:
                run_scan(raw,object(),object(),object(),ingest=ingest,privacy=privacy,temp_root=Path(d), scan_id='1a845e51-9b5e-4e03-bd79-dbb90ac7f5ed', event_callback=events.append)
            self.assertEqual(caught.exception.reservation_id,'a0c3ddf1-3bd1-4afb-9bcc-43dd44a1b8a4')
            self.assertNotIn('secret',str(caught.exception))
            self.assertNotIn('secret',str(events))
            self.assertEqual(len(list(Path(d).iterdir())),1)
