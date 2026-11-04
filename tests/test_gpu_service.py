import io
import unittest
from gpu_services import _request_glimmer, _validate_payload

class Response(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *args): self.close()

class ServiceTests(unittest.TestCase):
    def test_validation_rejects_wrong_model_or_unbounded_tokens(self):
        for payload in ({'model':'other','max_tokens':512,'messages':[{'role':'user','content':'x'}]}, {'model':'muse-glimmer','max_tokens':100000,'messages':[{'role':'user','content':'x'}]}):
            with self.assertRaises(ValueError): _validate_payload(payload)

    def test_local_inference_contract(self):
        calls=[]
        def open_(request, timeout):
            calls.append((request, timeout))
            return Response(b'{"choices":[{"message":{"content":"{}"}}]}')
        payload={'model':'muse-glimmer','max_tokens':512,'messages':[{'role':'user','content':'x'}]}
        result=_request_glimmer(payload, open_)
        self.assertEqual(result['choices'][0]['message']['content'], '{}')
        self.assertEqual(calls[0][0].full_url,'http://127.0.0.1:8080/v1/chat/completions')
        self.assertEqual(calls[0][1],240)
