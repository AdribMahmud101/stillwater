import tempfile
import unittest
from pathlib import Path
from app.clients.cache import CallCache
from app.clients.modal_glimmer import ModalGlimmerClient, ModalGlimmerError

class ModalGlimmerTests(unittest.TestCase):
    def test_remote_response_cached_with_duration(self):
        calls=[]
        def remote(payload):
            calls.append(payload)
            return {'response':{'choices':[{'message':{'content':'{}'}}], 'usage':{'total_tokens':5}}, 'inference_seconds':1.5}
        with tempfile.TemporaryDirectory() as d:
            client=ModalGlimmerClient(remote, CallCache(Path(d)))
            a=client.chat([{'role':'user','content':'hello'}],max_tokens=200)
            b=client.chat([{'role':'user','content':'hello'}],max_tokens=200)
            self.assertEqual(a['choices'][0]['message']['content'],'{}')
            self.assertEqual(a['_stillwater']['inference_seconds'],1.5)
            self.assertEqual(b['_stillwater']['inference_seconds'],0)
            self.assertTrue(b['_stillwater']['cached'])
            self.assertEqual(len(calls),1)
            self.assertEqual(calls[0]['model'],'muse-glimmer')

    def test_failed_calls_not_cached_and_errors_sanitized(self):
        calls=[]
        def remote(payload): calls.append(1); raise RuntimeError('secret')
        with tempfile.TemporaryDirectory() as d:
            client=ModalGlimmerClient(remote, CallCache(Path(d)))
            for _ in range(2):
                with self.assertRaises(ModalGlimmerError) as e: client.chat([{'role':'user','content':'hello'}],max_tokens=100)
                self.assertNotIn('secret',str(e.exception))
            self.assertEqual(len(calls),2)
