"""OpenAI-compatible Glimmer transport; deploy and model ID must be verified.

Callers must send privacy-processed images and reserve budget before use.
The shared witness parser in app.pipeline.witness handles validation and retry.
"""
import json
import math
import urllib.error
import urllib.parse
import urllib.request


class GlimmerError(RuntimeError):
    pass


class GlimmerClient:
    def __init__(self, base_url, api_key, cache, *, model, opener=None, timeout=120):
        url = urllib.parse.urlsplit(base_url)
        if (url.scheme != 'https' or not url.hostname or url.username or url.password
                or url.query or url.fragment or not api_key or cache is None or not model):
            raise ValueError('HTTPS endpoint, key, cache and model are required')
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be positive and finite')
        self._url = base_url.rstrip('/') + '/chat/completions'
        self._key, self._cache, self.model, self._timeout = api_key, cache, model, timeout
        self._open = opener or urllib.request.urlopen

    def chat(self, messages, **options):
        if not messages or 'model' in options or 'messages' in options or options.get('stream'):
            raise ValueError('messages required; streaming is unsupported')
        payload = {'model': self.model, 'messages': messages, **options}
        cache_key = {'endpoint': self._url, 'request': payload}
        cached = self._cache.get('glimmer', self.model, cache_key)
        if cached is not None:
            return cached
        request = urllib.request.Request(self._url, data=json.dumps(payload).encode(),
            headers={'Authorization': 'Bearer ' + self._key, 'Content-Type': 'application/json'}, method='POST')
        try:
            with self._open(request, timeout=self._timeout) as response:
                result = json.load(response)
            if not isinstance(result, dict) or not isinstance(result.get('choices'), list) or not result['choices']:
                raise ValueError('missing choices')
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, TypeError):
            raise GlimmerError('Glimmer request or response failed') from None
        self._cache.put('glimmer', self.model, cache_key, result)
        return result
