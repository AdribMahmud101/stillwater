import re
import unittest
from pathlib import Path

from app.pipeline.sam_track import TrackingError, track_video

FIXTURE = Path('tests/fixtures/sam_synthetic_video.txt').read_text()
SCORED = re.sub(r'(;h=128)(\|>)', r'\1;c=0.9\2', FIXTURE)

class FakeSAM:
    def __init__(self, raw=SCORED, fail=False): self.calls=[]; self.raw=raw; self.fail=fail
    def upload_video(self, video): self.calls.append('upload'); return 'file-123'
    def segment_video_file(self, file_id, concept):
        self.calls.append(concept)
        if self.fail: raise RuntimeError('request failed')
        return self.raw
    def delete_file(self, file_id): self.calls.append('delete')

class TrackingTest(unittest.TestCase):
    def test_one_upload_multiple_concepts_and_cleanup(self):
        client=FakeSAM()
        tracks=track_video(client, b'video', ['bucket', 'container'], 128, 128)
        self.assertEqual(client.calls, ['upload', 'bucket', 'container', 'delete'])
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]['aliases'], ['container'])
        self.assertGreater(len(tracks[0]['frames']), 2)

    def test_missing_scores_fail_closed_and_cleanup(self):
        client=FakeSAM(FIXTURE)
        with self.assertRaises(TrackingError): track_video(client, b'video', ['bucket'], 128, 128)
        self.assertEqual(client.calls[-1], 'delete')

    def test_provider_error_cleans_up(self):
        client=FakeSAM(fail=True)
        with self.assertRaises(RuntimeError): track_video(client, b'video', ['bucket'], 128, 128)
        self.assertEqual(client.calls[-1], 'delete')
