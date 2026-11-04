"""Baseten chat completions adapter with a required response cache."""
import json
import urllib.error
import urllib.request


class BasetenError(RuntimeError):
    """The provider call failed or returned an invalid response."""


class BasetenClient:
    def __init__(self, api_key, cache, *, model='meta-models/muse-spark-open-1026', opener=None, timeout=120):
        if not api_key or cache is None:
            raise ValueError('API key and cache are required')
        self._key, self._cache, self.model = api_key, cache, model
        self._open = opener or urllib.request.urlopen
        self._timeout = timeout

    def chat(self, messages, **options):
        response, _ = self.chat_with_meta(messages, **options)
        return response

    def chat_with_meta(self, messages, **options):
        if not messages or 'model' in options or 'messages' in options or options.get('stream'):
            raise ValueError('messages are required; streaming is not supported')
        payload = {'model': self.model, 'messages': messages, **options}
        cached = self._cache.get('baseten', self.model, payload)
        if cached is not None:
            return cached, True
        request = urllib.request.Request('https://inference.baseten.co/v1/chat/completions',
            data=json.dumps(payload).encode(), headers={'Authorization': 'Bearer ' + self._key,
            'Content-Type': 'application/json'}, method='POST')
        try:
            with self._open(request, timeout=self._timeout) as response:
                result = json.load(response)
            if not isinstance(result, dict) or not isinstance(result.get('choices'), list) or not result['choices']:
                raise ValueError('missing choices')
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, TypeError):
            raise BasetenError('Baseten request or response failed') from None
        self._cache.put('baseten', self.model, payload, result)
        return result, False
