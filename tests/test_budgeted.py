import sqlite3
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from app.clients.budgeted import budgeted_call
from app.clients.cache import CallCache
from app.clients.costs import BudgetExceeded, CostLedger


class BudgetedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.cache = CallCache(root / 'cache')
        self.ledger = CostLedger(root / 'costs.sqlite3', Decimal(1))
        self.addCleanup(self.tmp.cleanup)

    def call(self, execute):
        return budgeted_call(self.cache, self.ledger, '2026-11-03', 'meta', 'model', {'x': 1}, Decimal('.8'), execute)

    def test_cache_hit_does_not_call_or_charge_again(self):
        calls = []
        def request():
            calls.append(1)
            return {'text': 'ok'}, Decimal('.3')
        self.assertEqual(self.call(request), ({'text': 'ok'}, False))
        self.assertEqual(self.call(request), ({'text': 'ok'}, True))
        self.assertEqual(len(calls), 1)
        with sqlite3.connect(self.ledger.path) as db:
            self.assertEqual(db.execute('select usd from costs').fetchall(), [('0.3',)])

    def test_uncertain_failure_keeps_reservation(self):
        def request(): raise TimeoutError('unknown provider charge')
        with self.assertRaises(TimeoutError): self.call(request)
        with self.assertRaises(BudgetExceeded): self.call(request)

    def test_missing_usage_keeps_reservation(self):
        with self.assertRaises(ValueError): self.call(lambda: ({'text': 'ok'}, None))
        with self.assertRaises(BudgetExceeded): self.call(lambda: ({}, Decimal('.1')))

    def test_unaffordable_does_not_invoke_request(self):
        self.ledger.record('2026-11-03', Decimal('.5'))
        calls = []
        with self.assertRaises(BudgetExceeded): self.call(lambda: calls.append(1))
        self.assertEqual(calls, [])
