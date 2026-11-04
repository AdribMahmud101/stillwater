"""Optional reservation wrapper for GPU RPCs with verified billed duration.

The wrapped RPC must supply an authoritative, attributable billed duration in
`billed_gpu_seconds`. Inference time and locally observed wall-clock time are
not substitutes. This wrapper is not connected to production.
"""
from decimal import Decimal

from app.pipeline.budget_runner import run_budgeted


class BudgetedGPU:
    def __init__(self, remote, ledger, *, day, upper_bound, usd_per_gpu_hour):
        if not callable(remote):
            raise ValueError('remote must be callable')
        if (not isinstance(usd_per_gpu_hour, Decimal) or not usd_per_gpu_hour.is_finite()
                or usd_per_gpu_hour <= 0):
            raise ValueError('a verified positive Decimal hourly rate is required')
        self.remote, self.ledger, self.day = remote, ledger, day
        self.upper_bound, self.rate = upper_bound, usd_per_gpu_hour

    def __call__(self, payload):
        def invoke():
            result = self.remote(payload)
            if not isinstance(result, dict):
                raise ValueError('GPU result is invalid')
            seconds = result.get('billed_gpu_seconds')
            if isinstance(seconds, bool) or not isinstance(seconds, (int, Decimal)):
                raise ValueError('authoritative billed GPU seconds are unavailable')
            seconds = Decimal(seconds)
            if not seconds.is_finite() or seconds <= 0:
                raise ValueError('authoritative billed GPU seconds are unavailable')
            cost = seconds * self.rate / Decimal(3600)
            return {'response': result, 'cost_usd': cost}
        measured = run_budgeted(self.ledger, self.day, self.upper_bound, invoke)
        return {**measured['response'], 'estimated_gpu_cost_usd': str(measured['cost_usd'])}
