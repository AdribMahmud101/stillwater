import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

from app.clients.costs import BudgetExceeded, CostLedger
from app.pipeline.budget_runner import BudgetAccountingError, run_budgeted


class BudgetRunnerTests(unittest.TestCase):
    def test_running_scan_blocks_concurrent_admission(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=CostLedger(Path(d)/'costs.sqlite3',Decimal(1))
            started, release=threading.Event(),threading.Event()
            def runner():
                started.set(); release.wait(5); return {'cost_usd': Decimal('0.40')}
            with ThreadPoolExecutor(max_workers=2) as pool:
                future=pool.submit(run_budgeted,ledger,'2026-11-03',Decimal('0.75'),runner)
                self.assertTrue(started.wait(5))
                with self.assertRaises(BudgetExceeded): run_budgeted(ledger,'2026-11-03',Decimal('0.30'),lambda:None)
                release.set(); self.assertEqual(future.result()['cost_usd'],Decimal('0.40'))
            run_budgeted(ledger,'2026-11-03',Decimal('0.60'),lambda:{'cost_usd':Decimal(0)})

    def test_failure_and_missing_cost_keep_reservation(self):
        for runner in (lambda: (_ for _ in ()).throw(RuntimeError()), dict, lambda:{'cost_usd': Decimal('1.1')}):
            with tempfile.TemporaryDirectory() as d:
                ledger=CostLedger(Path(d)/'costs.sqlite3',Decimal(1))
                with self.assertRaises((RuntimeError,BudgetAccountingError)): run_budgeted(ledger,'2026-11-03',Decimal(1),runner)
                with self.assertRaises(BudgetExceeded): ledger.reserve('2026-11-03',Decimal('0.01'))

    def test_ledger_rejects_nonfinite_and_inconsistent_values(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'costs.sqlite3'
            with self.assertRaises(ValueError): CostLedger(path,Decimal('Infinity'))
            ledger=CostLedger(path,Decimal('1'))
            for value in (Decimal('NaN'), Decimal('Infinity'), Decimal('-1')):
                with self.assertRaises(ValueError): ledger.record('2026-11-03',value)
            reservation=ledger.reserve('2026-11-03',Decimal('.25'))
            with self.assertRaises(ValueError): ledger.settle(reservation,Decimal('.26'))
            with self.assertRaises(BudgetExceeded): ledger.reserve('2026-11-03',Decimal('.76'))
