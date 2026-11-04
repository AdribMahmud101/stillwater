"""Estimate model charges from provider counters and explicitly supplied rates.

These are estimates, not provider invoices. Rate source, pricing tiers, cached
input discounts, taxes and currency must be verified by the caller. GPU time,
uploads and any other fees are deliberately excluded.
"""
from decimal import Decimal


class UsageError(ValueError):
    pass


def _price(value):
    if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
        raise UsageError('an explicit finite nonnegative Decimal rate is required')
    return value


def _count(value):
    if type(value) is not int or value < 0:
        raise UsageError('a nonnegative integer usage count is required')
    return value


def _cached(value):
    if type(value) is not bool:
        raise UsageError('cache status must be explicitly known')
    return value


def baseten_usage_estimate(response, *, input_per_million: Decimal, output_per_million: Decimal, cached: bool) -> Decimal:
    """Estimate using total input and output counts; reasoning is included in output."""
    input_rate, output_rate = _price(input_per_million), _price(output_per_million)
    if _cached(cached):
        return Decimal(0)
    if not isinstance(response, dict) or not isinstance(response.get('usage'), dict):
        raise UsageError('Baseten usage is unavailable')
    usage = response['usage']
    inp, out = _count(usage.get('prompt_tokens')), _count(usage.get('completion_tokens'))
    return (Decimal(inp) * input_rate + Decimal(out) * output_rate) / Decimal(1_000_000)


def sam_video_estimate(result, *, per_frame: Decimal, minimum_frames: int, cached: bool) -> Decimal:
    """Estimate from provider processed-frame count and caller-specified minimum."""
    rate, minimum = _price(per_frame), _count(minimum_frames)
    if _cached(cached):
        return Decimal(0)
    if not isinstance(result, dict):
        raise UsageError('SAM usage is unavailable')
    frames = _count(result.get('video_frames_processed'))
    return Decimal(max(frames, minimum)) * rate
