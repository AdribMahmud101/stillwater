"""Hosted SAM 3.1 Responses API adapter.

Returns the provider's segmentation grammar unchanged; callers must parse it
with Meta's SAM parser before using masks. Video uploads and deletion are
the responsibility of the caller.
"""
import base64
import hashlib
import json
import re
import secrets
import urllib.error
import urllib.request


class MetaSAMError(RuntimeError):
    """The provider call did not return a complete, valid response."""


class MetaSAMClient:
    def __init__(self, api_key, cache, *, opener=None, timeout=120):
        if not api_key or cache is None:
            raise ValueError('API key and cache are required')
        self._key, self._cache = api_key, cache
        self._open = opener or urllib.request.urlopen
        self._timeout = timeout
        self._upload_digests = {}

    @staticmethod
    def _valid_file_id(file_id):
        return isinstance(file_id, str) and re.fullmatch(r"file[-_][A-Za-z0-9_-]+", file_id) is not None

    def upload_video(self, video: bytes) -> str:
        """Upload MP4 with a one-hour expiry; caller must delete it promptly.

        Network failures can leave an uploaded file whose ID was not received;
        expiration limits retention even in that case. Video is buffered in memory.
        """
        if not isinstance(video, bytes) or not video or len(video) > 200_000_000:
            raise ValueError('video must be non-empty bytes up to 200 MB')
        boundary = 'stillwater-' + secrets.token_hex(16)
        def field(name, value):
            return (f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n').encode()
        body = (field('purpose', 'user_data') + field('expires_after[anchor]', 'created_at')
                + field('expires_after[seconds]', '3600')
                + (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="clip.mp4"\r\nContent-Type: video/mp4\r\n\r\n').encode()
                + video + f'\r\n--{boundary}--\r\n'.encode())
        request = urllib.request.Request('https://api.meta.ai/v1/files', data=body,
            headers={'Authorization': 'Bearer ' + self._key, 'Content-Type': 'multipart/form-data; boundary=' + boundary}, method='POST')
        try:
            with self._open(request, timeout=self._timeout) as response:
                result = json.load(response)
            file_id = result.get('id')
            if not self._valid_file_id(file_id):
                raise ValueError('invalid file ID')
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, TypeError, AttributeError):
            raise MetaSAMError('Meta upload failed; remote file may exist until expiry') from None
        self._upload_digests[file_id] = hashlib.sha256(video).hexdigest()
        return file_id

    def delete_file(self, file_id: str) -> None:
        """Delete a remote file and require the provider's deletion confirmation."""
        if not self._valid_file_id(file_id):
            raise ValueError('invalid file ID')
        request = urllib.request.Request('https://api.meta.ai/v1/files/' + file_id,
            headers={'Authorization': 'Bearer ' + self._key}, method='DELETE')
        try:
            with self._open(request, timeout=self._timeout) as response:
                result = json.load(response)
            if result.get('deleted') is not True or result.get('id') != file_id:
                raise ValueError('deletion not confirmed')
            self._upload_digests.pop(file_id, None)
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, TypeError, AttributeError):
            raise MetaSAMError('Meta file deletion not confirmed; file may remain until expiry') from None

    def segment_image(self, image: bytes, mime_type: str, prompt: str) -> str:
        if not image or mime_type not in ('image/png', 'image/jpeg', 'image/webp') or not prompt.strip():
            raise ValueError('image, supported MIME type and prompt are required')
        image_url = 'data:' + mime_type + ';base64,' + base64.b64encode(image).decode('ascii')
        return self._segment({'type': 'input_image', 'image_url': image_url}, prompt)

    def segment_video_file(self, file_id: str, prompt: str) -> str:
        """Segment an already uploaded file; caller owns upload and deletion."""
        if not self._valid_file_id(file_id) or not prompt.strip():
            raise ValueError('Meta file ID and prompt are required')
        return self._segment({'type': 'input_video', 'file_id': file_id}, prompt)

    def segment_video_file_with_usage(self, file_id: str, prompt: str) -> dict:
        """Return completed output and optional provider-reported frame count."""
        if not self._valid_file_id(file_id) or not prompt.strip():
            raise ValueError('Meta file ID and prompt are required')
        return self._segment({'type': 'input_video', 'file_id': file_id}, prompt, with_usage=True)

    def segment_video_file_with_meta(self, file_id: str, prompt: str) -> dict:
        """Return text, provider frame count and explicit cache-hit status."""
        if not self._valid_file_id(file_id) or not prompt.strip():
            raise ValueError('Meta file ID and prompt are required')
        return self._segment({'type': 'input_video', 'file_id': file_id}, prompt, with_usage=True, with_meta=True)

    def _segment(self, media, prompt, *, with_usage=False, with_meta=False):
        payload = {'model': 'sam-3.1', 'input': [{'type': 'message', 'role': 'user', 'content': [
            {'type': 'input_text', 'text': prompt}, media
        ]}], 'stream': True, 'metadata': {'include_confidence': 'true'}}
        # File IDs expire and differ on re-upload; use a digest only for IDs
        # this client itself uploaded, so the digest is bound to actual bytes.
        media_key = media
        if media.get('type') == 'input_video' and media.get('file_id') in self._upload_digests:
            media_key = {'type': 'input_video_sha256', 'sha256': self._upload_digests[media['file_id']]}
        cache_key = {'model': payload['model'], 'media': media_key, 'prompt': prompt,
                     'metadata': payload['metadata'], 'stream': payload['stream']}
        cached = self._cache.get('meta', 'sam-3.1', cache_key)
        if cached is not None:
            result = {'text': cached['text'], 'video_frames_processed': cached.get('video_frames_processed')}
            if with_meta:
                result['cached'] = True
            return result if with_usage else cached['text']
        request = urllib.request.Request('https://api.meta.ai/v1/responses',
            data=json.dumps(payload).encode(), headers={'Authorization': 'Bearer ' + self._key,
            'Content-Type': 'application/json', 'Accept': 'text/event-stream'}, method='POST')
        output, completed, frame_count = [], False, None
        try:
            with self._open(request, timeout=self._timeout) as response:
                for line in response:
                    if not line.startswith(b'data:'):
                        continue
                    raw = line[5:].strip()
                    if raw == b'[DONE]':
                        continue
                    event = json.loads(raw)
                    kind = event.get('type')
                    if kind == 'response.output_text.delta':
                        output.append(event.get('delta', ''))
                    elif kind == 'response.completed':
                        completed = True
                        response_data = event.get('response') or {}
                        usage = response_data.get('usage') or {}
                        value = usage.get('video_frames_processed')
                        frame_count = value if type(value) is int and value >= 0 else None
                    elif kind in ('response.failed', 'error'):
                        raise MetaSAMError('SAM response failed')
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, TypeError) as exc:
            import logging
            detail = type(exc).__name__
            if isinstance(exc, urllib.error.HTTPError):
                try:
                    body = exc.read(300).decode('utf-8', 'replace')
                except Exception:  # noqa: BLE001
                    body = ''
                detail = 'HTTP %s %s' % (exc.code, body)
            else:
                detail = '%s %s' % (type(exc).__name__, str(exc)[:160])
            logging.getLogger('stillwater').error('SAM request failed: %s', detail)
            err = MetaSAMError('SAM request or response failed')
            if isinstance(exc, urllib.error.HTTPError):
                err.status_code = exc.code
                err.transient = exc.code == 429 or 500 <= exc.code <= 599
            else:
                # network drops, resets and timeouts are retryable; parse errors are not
                err.transient = isinstance(exc, (urllib.error.URLError, TimeoutError, OSError))
            raise err from None
        if not completed:
            raise MetaSAMError('SAM stream ended before completion')
        text = ''.join(output)
        result = {'text': text, 'video_frames_processed': frame_count}
        self._cache.put('meta', 'sam-3.1', cache_key, result)
        if with_meta:
            result = {**result, 'cached': False}
        return result if with_usage else text
