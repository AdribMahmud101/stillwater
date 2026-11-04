"""Serialize identical calls, cache their results, and reserve before spending.

Reservations survive uncertain failures until reconciled manually. This protects
against assuming a timed-out provider did not bill, but requires an operator to
reconcile the ledger. File locks coordinate processes sharing this filesystem;
they are not distributed locks across machines or independent volumes.
"""
import fcntl
from decimal import Decimal


def budgeted_call(cache, ledger, day, provider, model, payload, estimated_usd, execute):
    """Return (response, cache_hit); execute returns (JSON response, actual USD Decimal).

    Estimate must be a defensible upper bound; actual overages are recorded, but
    cannot retroactively prevent spending. Never include secrets in payload.
    """
    if not isinstance(estimated_usd, Decimal) or not estimated_usd.is_finite() or estimated_usd < 0:
        raise ValueError('estimated cost must be a finite, nonnegative Decimal')
    lock_path = cache._path(provider, model, payload).with_suffix('.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cached = cache.get(provider, model, payload)
        if isinstance(cached, dict) and cached.get('_budgeted_cache') == 1 and 'response' in cached:
            return cached['response'], True
        reservation = ledger.reserve(day, estimated_usd)
        response, actual_usd = execute()
        if not isinstance(actual_usd, Decimal) or not actual_usd.is_finite() or actual_usd < 0:
            raise ValueError('actual cost is unknown; reservation retained for reconciliation')
        ledger.settle(reservation, actual_usd)
        cache.put(provider, model, payload, {'_budgeted_cache': 1, 'response': response})
        return response, False
