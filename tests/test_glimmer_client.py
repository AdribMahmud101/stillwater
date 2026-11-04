import io
import json
import tempfile
import unittest
from pathlib import Path

from app.clients.cache import CallCache
from app.clients.glimmer import GlimmerClient, GlimmerError
from app.pipeline.witness import describe_container


class Response(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *args): self.close()

class GlimmerTest(unittest.TestCase):
    def test_chat_and_witness_retry_are_cached(self):
        requests=[]
        attrs = {'holds_water': 'yes', 'covered': 'no', 'water_visible': 'unsure', 'exposed_to_rain': 'yes', 'orientation': 'upright', 'object_label': 'bucket', 'evidence': 'Open bucket.'}
        def open_(request, timeout):
            self.assertEqual(timeout, 10)
            requests.append(json.loads(request.data))
            text = 'not json' if len(requests) == 1 else json.dumps(attrs)
            return Response(json.dumps({'choices': [{'message': {'content': text}}]}).encode())
        with tempfile.TemporaryDirectory() as d:
            client=GlimmerClient('https://example.com/v1', 'secret', CallCache(Path(d)), model='glimmer-test', opener=open_, timeout=10)
            self.assertEqual(describe_container(client, b'image', 'image/png'), attrs)
            self.assertEqual(describe_container(client, b'image', 'image/png'), attrs)
            self.assertEqual(len(requests), 2)

    def test_errors_do_not_leak_credentials(self):
        def open_(request, timeout): raise OSError('secret')
        with tempfile.TemporaryDirectory() as d:
            client=GlimmerClient('https://example.com/v1', 'secret', CallCache(Path(d)), model='glimmer-test', opener=open_)
            with self.assertRaises(GlimmerError) as caught: client.chat([{'role': 'user', 'content': 'hi'}])
            self.assertNotIn('secret', str(caught.exception))

    def test_endpoint_must_be_secure(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ValueError):
            GlimmerClient('http://example.com/v1', 'x', CallCache(Path(d)), model='glimmer-test')
