import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from app.clients.costs import CostLedger, BudgetExceeded
from app.clients.gpu_budget import BudgetedGPU
from app.pipeline.budget_runner import BudgetAccountingError

class GPUBudgetTests(unittest.TestCase):
    def test_explicit_billed_seconds_only(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=CostLedger(Path(d)/'costs.sqlite',Decimal('10'))
            client=BudgetedGPU(lambda p:{'response':{'choices':[]},'inference_seconds':1,'billed_gpu_seconds':120},ledger,day='2026-11-03',upper_bound=Decimal('1'),usd_per_gpu_hour=Decimal('3'))
            result=client({'model':'test'})
            self.assertEqual(result['estimated_gpu_cost_usd'],str(Decimal('0.1')))

    def test_missing_actual_billed_duration_retains_reservation(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=CostLedger(Path(d)/'costs.sqlite',Decimal('1'))
            client=BudgetedGPU(lambda p:{'response':{},'inference_seconds':1},ledger,day='2026-11-03',upper_bound=Decimal('1'),usd_per_gpu_hour=Decimal('1'))
            with self.assertRaises(BudgetAccountingError): client({})
            with self.assertRaises(BudgetExceeded): ledger.reserve('2026-11-03',Decimal('.01'))

    def test_negative_nonfinite_or_missing_rate_rejected_before_call(self):
        for rate in (None,Decimal('-1'),Decimal('NaN')):
            with tempfile.TemporaryDirectory() as d:
                ledger=CostLedger(Path(d)/'costs.sqlite',Decimal('1'))
                calls=[]
                with self.assertRaises(ValueError): BudgetedGPU(lambda p:calls.append(1),ledger,day='2026-11-03',upper_bound=Decimal('1'),usd_per_gpu_hour=rate)
                self.assertEqual(calls,[])
