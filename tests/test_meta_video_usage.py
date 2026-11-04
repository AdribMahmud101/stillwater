import io
import json
import tempfile
import unittest
from pathlib import Path

from app.clients.cache import CallCache
from app.clients.meta import MetaSAMClient


class Response(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *args): self.close()


class MetaVideoUsageTests(unittest.TestCase):
    def test_completed_usage_cached_and_existing_method_preserved(self):
        calls = []
        def opener(request, timeout):
            calls.append(1)
            events = [{'type': 'response.output_text.delta', 'delta': 'result'},
                      {'type': 'response.completed', 'response': {'usage': {'video_frames_processed': 12}}}]
            return Response(b''.join(b'data: ' + json.dumps(event).encode() + b'\n\n' for event in events))
        with tempfile.TemporaryDirectory() as tmp:
            client = MetaSAMClient('secret', CallCache(Path(tmp)), opener=opener)
            self.assertEqual(client.segment_video_file_with_usage('file-123', 'person'), {'text': 'result', 'video_frames_processed': 12})
            self.assertEqual(client.segment_video_file('file-123', 'person'), 'result')
            self.assertEqual(len(calls), 1)

    def test_missing_count_remains_unknown(self):
        def opener(request, timeout): return Response(b'data: {"type":"response.completed","response":{"usage":{}}}\n\n')
        with tempfile.TemporaryDirectory() as tmp:
            client = MetaSAMClient('secret', CallCache(Path(tmp)), opener=opener)
            self.assertEqual(client.segment_video_file_with_usage('file-123', 'person'), {'text': '', 'video_frames_processed': None})
