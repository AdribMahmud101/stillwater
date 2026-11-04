"""Cached private Modal RPC transport for the shared witness interface.

RPC durations exclude cold start, idle GPU time and billing; they are not cost.
Callers must enforce budget and supply privacy-processed media.
"""
import concurrent.futures
import math
import os


def _glimmer_call_timeout_default() -> float:
    raw = os.environ.get("GLIMMER_CALL_TIMEOUT", "180")
    try:
        v = float(str(raw).strip())
    except Exception:
        return 180.0
    if not math.isfinite(v) or v <= 0:
        return 180.0
    return v


def _glimmer_warm_timeout_short_default() -> float:
    raw = os.environ.get("GLIMMER_WARM_SHORT_TIMEOUT", "30")
    try:
        v = float(str(raw).strip())
    except Exception:
        return 30.0
    if not math.isfinite(v) or v <= 0:
        return 30.0
    return v


class ModalGlimmerError(RuntimeError):
    pass


class ModalGlimmerClient:
    def __init__(self, remote_chat, cache, *, service_identity='stillwater-glimmer:GlimmerService', remote_warm=None):
        if not callable(remote_chat) or cache is None or not isinstance(service_identity, str) or not service_identity:
            raise ValueError('remote callable, cache and service identity are required')
        if remote_warm is not None and not callable(remote_warm):
            raise ValueError('remote_warm must be callable or None')
        self.remote_chat, self.cache, self.service_identity = remote_chat, cache, service_identity
        self._remote_warm = remote_warm
        self.model = 'muse-glimmer'

    def warm(self, timeout=None):
        """Cheap warm-up probe; never raises, never consumes scan budget.

        Returns True on ok, False on any failure/timeout. May record an
        estimated GPU cost note only (no budget accounting here).
        """
        fn = self._remote_warm
        if fn is None:
            # Try to derive a warm callable from a Modal-style remote_chat
            # object (tests inject fakes with .warm or _warm attributes).
            for attr in ("warm", "remote_warm", "_warm"):
                try:
                    cand = getattr(self.remote_chat, attr, None)
                except Exception:
                    cand = None
                if callable(cand):
                    fn = cand
                    break
            else:
                try:
                    owner = getattr(self.remote_chat, "__self__", None)
                    if owner is not None:
                        for attr in ("warm", "remote_warm"):
                            cand = getattr(owner, attr, None)
                            if callable(cand):
                                fn = cand
                                break
                except Exception:
                    fn = None
        if fn is None:
            return False
        if timeout is None:
            timeout = _glimmer_warm_timeout_short_default()
        try:
            timeout = float(timeout)
        except Exception:
            timeout = _glimmer_warm_timeout_short_default()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(fn)
                try:
                    res = fut.result(timeout=timeout)
                except BaseException:
                    try:
                        fut.cancel()
                    except Exception:
                        pass
                    return False
            # Accept {'ok': True}, 'ok', True as success.
            if res is True or res == 'ok':
                return True
            if isinstance(res, dict) and res.get('ok') is True:
                return True
            # Envelope with remote() style? Treat truthy dict ok.
            return bool(res) and (not isinstance(res, dict) or bool(res.get('ok', True)))
        except Exception:
            return False

    def chat(self, messages, **options):
        if not isinstance(messages, list) or not messages or 'model' in options or 'messages' in options or options.get('stream'):
            raise ValueError('messages required; streaming unsupported')
        payload = {'model': self.model, 'messages': messages, **options}
        key = {'service': self.service_identity, 'request': payload}
        response = self.cache.get('modal-glimmer', self.model, key)
        if response is not None:
            return {**response, '_stillwater': {'cached': True, 'inference_seconds': 0.0}}
        timeout = _glimmer_call_timeout_default()
        # Options may carry an explicit timeout override (tests).
        try:
            if 'timeout' in options and options['timeout'] is not None:
                timeout = float(options['timeout'])
        except Exception:
            timeout = _glimmer_call_timeout_default()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(self.remote_chat, payload)
                try:
                    envelope = fut.result(timeout=timeout)
                except concurrent.futures.TimeoutError:
                    try:
                        fut.cancel()
                    except Exception:
                        pass
                    raise ModalGlimmerError('Glimmer RPC timed out')
                except ModalGlimmerError:
                    raise
                except Exception as exc:
                    raise ModalGlimmerError('Glimmer RPC failed or returned invalid data (%s)' % type(exc).__name__) from None
            response = envelope['response']
            seconds = envelope['inference_seconds']
            if (not isinstance(response, dict) or not isinstance(response.get('choices'), list) or not response['choices']
                    or isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0):
                raise ValueError('invalid remote response')
        except ModalGlimmerError:
            raise
        except Exception as exc:
            raise ModalGlimmerError('Glimmer RPC failed or returned invalid data (%s)' % type(exc).__name__) from None
        self.cache.put('modal-glimmer', self.model, key, response)
        return {**response, '_stillwater': {'cached': False, 'inference_seconds': seconds}}
