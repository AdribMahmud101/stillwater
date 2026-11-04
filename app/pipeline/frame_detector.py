"""Budgeted per-image SAM detector for experimental privacy processing.

Caller supplies a current verified per-image cost and conservative upper bound.
Provider failures retain reservations for reconciliation. The injected client
must only return a string after the provider's response.completed event.
Not wired to public scanning; detection recall remains unverified.
"""
from decimal import Decimal
from hashlib import sha256
from app.clients.budgeted import budgeted_call


def make_frame_detector(client, cache, ledger, day_provider, estimate_usd, actual_usd):
    for value in (estimate_usd, actual_usd):
        if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
            raise ValueError('estimate and actual image cost must be explicit finite nonnegative Decimal values')
    if actual_usd > estimate_usd:
        raise ValueError('estimate must cover actual image cost')
    if not callable(day_provider):
        raise ValueError('day provider must be callable')

    def detect(image):
        if not isinstance(image, bytes) or not image:
            raise ValueError('a nonempty JPEG image is required')
        payload = {'sha256': sha256(image).hexdigest(), 'mime_type': 'image/jpeg', 'prompt': 'person'}
        def execute():
            text = client.segment_image(image, 'image/jpeg', 'person')
            if not isinstance(text, str):
                raise ValueError('provider did not return a completed image response')
            return {'completed': True, 'text': text}, actual_usd
        result, _ = budgeted_call(cache, ledger, day_provider(), 'meta', 'sam-3.1-image', payload, estimate_usd, execute)
        return result
    return detect
