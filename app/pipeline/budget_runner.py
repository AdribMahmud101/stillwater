"""Conservative whole-scan admission; does not establish provider costs.

Caller must calculate a defensible maximum before starting and report actual
measured charges. Uncertain runs keep their reservation for reconciliation.
"""
from datetime import date
from decimal import Decimal


class BudgetAccountingError(RuntimeError):
    def __init__(self, message, reservation_id):
        super().__init__(message)
        self.reservation_id = reservation_id


def run_budgeted(ledger, day: str, upper_bound_usd: Decimal, runner, *args, measured_spend=None, **kwargs):
    """Atomically reserve a maximum, run, and settle only known in-bound cost.

    Does not cancel requests that exceed their bound. A scan failure, absent or
    excessive measured cost, or failed settlement retains the reservation,
    except when ``measured_spend`` is provided: on scan failure the
    reservation is settled at ``min(measured_spend(), reservation)`` when the
    callable returns a finite non-negative Decimal, releasing the rest. The
    error is still raised as BudgetAccountingError. If ``measured_spend`` is
    missing, not callable, raises, or returns an invalid value, the old
    conservative behavior (retain reservation) is kept. Never settles below
    what was measured (except clamping to the reservation) and never raises
    the daily cap (settlement only moves/shrinks the reservation).
    """
    try:
        if not isinstance(day, str) or date.fromisoformat(day).isoformat() != day:
            raise ValueError('invalid accounting date')
    except (TypeError, ValueError):
        raise ValueError('invalid accounting date') from None
    if not isinstance(upper_bound_usd, Decimal) or not upper_bound_usd.is_finite() or upper_bound_usd <= 0:
        raise ValueError('a finite positive Decimal upper bound is required')
    if not callable(runner):
        raise ValueError('runner must be callable')  # noqa: TRY004
    if measured_spend is not None and not callable(measured_spend):
        raise ValueError('measured_spend must be a zero-arg callable or None')
    reservation = ledger.reserve(day, upper_bound_usd)
    try:
        result = runner(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        import logging
        logging.getLogger('stillwater').error('scan stage failed: %s: %s', type(exc).__name__, str(exc)[:300])
        if measured_spend is not None:
            try:
                m = measured_spend()
            except Exception:
                m = None
            if isinstance(m, Decimal) and m.is_finite() and m >= 0:
                settle_at = m if m <= upper_bound_usd else upper_bound_usd
                try:
                    ledger.settle(reservation, settle_at)
                    raise BudgetAccountingError('scan failed; settled at measured spend', reservation) from None
                except BudgetAccountingError:
                    raise
                except Exception:
                    pass
        raise BudgetAccountingError('scan failed; reservation retained for reconciliation', reservation) from None
    actual = result.get('cost_usd') if isinstance(result, dict) else None
    if not isinstance(actual, Decimal) or not actual.is_finite() or actual < 0 or actual > upper_bound_usd:
        raise BudgetAccountingError('scan cost unavailable or exceeds reservation; reconcile manually', reservation)
    try:
        ledger.settle(reservation, actual)
    except Exception:  # noqa: BLE001
        raise BudgetAccountingError('settlement failed; reconcile reservation manually', reservation) from None
    return result
