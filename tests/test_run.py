import tempfile
import unittest
from pathlib import Path

from app.pipeline.run import PipelineError, run_pipeline


class FakeStages:
    def __init__(self): self.calls = []
    def ingest(self, raw): self.calls.append('ingest'); return raw
    def privacy(self, normalized):
        self.calls.append('privacy')
        p = normalized.parent / 'private.mp4'; p.write_bytes(b'blurred'); return p
    def scout(self, private): self.calls.append(('scout', private)); return ['bucket']
    def track(self, private, concepts): self.calls.append(('track', private)); return [{'id': 'a'}]
    def verify(self, private, tracks): self.calls.append(('verify', private)); return [{'id': 'a', 'risk': 'review'}]
    def act(self, records): self.calls.append('act'); return []


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.raw = Path(self.tmp.name) / 'raw.mp4'; self.raw.write_bytes(b'raw')
        self.stages = FakeStages()

    def test_privacy_precedes_all_downstream_video_access(self):
        result = run_pipeline(self.raw, self.stages)
        self.assertEqual(result['containers'][0]['risk'], 'review')
        self.assertEqual(self.stages.calls[:2], ['ingest', 'privacy'])
        for call in self.stages.calls[2:5]: self.assertEqual(call[1].name, 'private.mp4')

    def test_privacy_failure_does_not_run_models_or_actions(self):
        def fail(_): raise RuntimeError('failure')
        self.stages.privacy = fail
        with self.assertRaises(PipelineError): run_pipeline(self.raw, self.stages)
        self.assertEqual(self.stages.calls, ['ingest'])

    def test_rejects_raw_or_missing_as_private(self):
        for returned in (self.raw, self.raw.parent / 'missing'):
            self.stages.calls.clear(); self.stages.privacy = lambda _, returned=returned: returned
            with self.assertRaises(PipelineError): run_pipeline(self.raw, self.stages)
            self.assertEqual(self.stages.calls, ['ingest'])
