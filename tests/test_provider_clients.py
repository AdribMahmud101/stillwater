import io
import json
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError

from app.clients.baseten import BasetenClient, BasetenError
from app.clients.cache import CallCache
from app.clients.meta import MetaSAMClient, MetaSAMError


class Response(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *args): self.close()


class ClientTests(unittest.TestCase):
    def test_meta_stream_and_cache(self):
        calls = []
        def open_(request, timeout):
            calls.append(json.loads(request.data))
            return Response(b'data: {"type":"response.output_text.delta","delta":"box"}\n\ndata: {"type":"response.completed"}\n\n')
        with tempfile.TemporaryDirectory() as d:
            client = MetaSAMClient('secret', CallCache(Path(d)), opener=open_)
            self.assertEqual(client.segment_image(b'image', 'image/png', 'bucket'), 'box')
            self.assertEqual(client.segment_image(b'image', 'image/png', 'bucket'), 'box')
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]['model'], 'sam-3.1')
            self.assertTrue(calls[0]['stream'])
            self.assertEqual(calls[0]['metadata']['include_confidence'], 'true')
            self.assertEqual(calls[0]['input'][0]['content'][0]['text'], 'bucket')

    def test_meta_video_file_stream_returns_all_frame_records(self):
        calls = []
        def open_(request, timeout):
            calls.append(json.loads(request.data))
            return Response(b'data: {"type":"response.output_text.delta","delta":"<0f>first\\n"}\n\ndata: {"type":"response.output_text.delta","delta":"<1f>second"}\n\ndata: {"type":"response.completed"}\n\n')
        with tempfile.TemporaryDirectory() as d:
            client = MetaSAMClient('secret', CallCache(Path(d)), opener=open_)
            self.assertEqual(client.segment_video_file('file-abc', 'bucket'), '<0f>first\n<1f>second')
            self.assertEqual(client.segment_video_file('file-abc', 'bucket'), '<0f>first\n<1f>second')
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]['input'][0]['content'][1], {'type': 'input_video', 'file_id': 'file-abc'})

    def test_meta_video_upload_and_deletion(self):
        requests = []
        def open_(request, timeout):
            requests.append(request)
            if request.get_method() == 'POST':
                return Response(b'{"id":"file-123","status":"uploaded"}')
            return Response(b'{"id":"file-123","deleted":true}')
        with tempfile.TemporaryDirectory() as d:
            client = MetaSAMClient('secret', CallCache(Path(d)), opener=open_)
            self.assertEqual(client.upload_video(b'clip'), 'file-123')
            self.assertIn(b'name="purpose"', requests[0].data)
            self.assertIn(b'name="expires_after[seconds]"', requests[0].data)
            self.assertIn(b'3600', requests[0].data)
            client.delete_file('file-123')
            self.assertEqual(requests[1].get_method(), 'DELETE')
            self.assertTrue(requests[1].full_url.endswith('/files/file-123'))

    def test_meta_delete_requires_confirmation_and_safe_id(self):
        def open_(request, timeout): return Response(b'{"deleted":false}')
        with tempfile.TemporaryDirectory() as d:
            client = MetaSAMClient('secret', CallCache(Path(d)), opener=open_)
            with self.assertRaises(MetaSAMError): client.delete_file('file-123')
            with self.assertRaises(ValueError): client.delete_file('file-../bad')

    def test_meta_incomplete_stream_is_not_cached(self):
        calls = []
        def open_(request, timeout):
            calls.append(1)
            return Response(b'data: {"type":"response.output_text.delta","delta":"partial"}\n\n')
        with tempfile.TemporaryDirectory() as d:
            client = MetaSAMClient('secret', CallCache(Path(d)), opener=open_)
            for _ in range(2):
                with self.assertRaises(MetaSAMError): client.segment_image(b'i', 'image/png', 'bucket')
            self.assertEqual(len(calls), 2)

    def test_baseten_response_is_cached_and_error_not_cached(self):
        calls = []
        def open_(request, timeout):
            calls.append(json.loads(request.data))
            if len(calls) == 1: raise HTTPError(request.full_url, 429, 'rate', {}, None)
            return Response(b'{"choices":[{"message":{"content":"ok"}}],"usage":{"total_tokens":2}}')
        with tempfile.TemporaryDirectory() as d:
            client = BasetenClient('secret', CallCache(Path(d)), opener=open_)
            with self.assertRaises(BasetenError): client.chat([{'role': 'user', 'content': 'hi'}])
            a = client.chat([{'role': 'user', 'content': 'hi'}])
            self.assertEqual(a['choices'][0]['message']['content'], 'ok')
            self.assertEqual(client.chat([{'role': 'user', 'content': 'hi'}]), a)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[-1]['model'], 'meta-models/muse-spark-open-1026')

    def test_video_cache_uses_uploaded_content_not_transient_file_id(self):
        calls=[]
        def open_(request, timeout):
            calls.append(request)
            if request.full_url.endswith('/files'):
                return Response(json.dumps({'id':f'file-{len(calls)}'}).encode())
            if request.get_method() == 'DELETE':
                return Response(json.dumps({'id':request.full_url.rsplit('/',1)[-1],'deleted':True}).encode())
            return Response(b'data: {"type":"response.output_text.delta","delta":"output"}\n\ndata: {"type":"response.completed"}\n\n')
        with tempfile.TemporaryDirectory() as d:
            c=MetaSAMClient('secret', CallCache(Path(d)), opener=open_)
            first=c.upload_video(b'video')
            self.assertEqual(c.segment_video_file(first,'bucket'),'output')
            c.delete_file(first)
            second=c.upload_video(b'video')
            self.assertEqual(c.segment_video_file(second,'bucket'),'output')
            c.delete_file(second)
            self.assertEqual(sum(req.full_url.endswith('/responses') for req in calls), 1)
