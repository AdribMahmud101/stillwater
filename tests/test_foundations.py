import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from app.clients.cache import CallCache
from app.clients.costs import BudgetExceeded, CostLedger


class FoundationsTest(unittest.TestCase):
    def test_cache_is_keyed_by_provider_model_and_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = CallCache(Path(tmp))
            cache.put('meta', 'sam', {'prompt': 'bucket'}, {'result': 1})
            self.assertEqual(cache.get('meta', 'sam', {'prompt': 'bucket'}), {'result': 1})
            self.assertIsNone(cache.get('meta', 'sam', {'prompt': 'tire'}))
            self.assertIsNone(cache.get('other', 'sam', {'prompt': 'bucket'}))

    def test_budget_counts_day_and_rejects_over_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = CostLedger(Path(tmp) / 'costs.sqlite3', Decimal('1.00'))
            ledger.record('2026-11-03', Decimal('0.70'))
            ledger.record('2026-11-02', Decimal(10))
            ledger.check('2026-11-03', Decimal('0.30'))
            with self.assertRaises(BudgetExceeded):
                ledger.check('2026-11-03', Decimal('0.31'))

class ReservationTests(unittest.TestCase):
    def test_reservations_count_toward_budget_and_can_be_settled(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = CostLedger(Path(tmp) / 'costs.sqlite3', Decimal('1.00'))
            reservation = ledger.reserve('2026-11-03', Decimal('0.70'))
            with self.assertRaises(BudgetExceeded):
                ledger.reserve('2026-11-03', Decimal('0.31'))
            ledger.settle(reservation, Decimal('0.40'))
            ledger.reserve('2026-11-03', Decimal('0.60'))
