import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from app.clients.cache import CallCache
from app.clients.costs import BudgetExceeded, CostLedger
from app.pipeline.frame_detector import make_frame_detector


class Client:
    def __init__(self): self.calls = []
    def segment_image(self, image, mime, prompt):
        self.calls.append((image, mime, prompt)); return ''


class DetectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.cache = CallCache(root / 'cache'); self.ledger = CostLedger(root / 'costs', Decimal('1'))
        self.client = Client()

    def make(self, estimate=Decimal('.6'), actual=Decimal('.4')):
        return make_frame_detector(self.client, self.cache, self.ledger, lambda: '2026-11-03', estimate, actual)

    def test_each_image_is_cached_and_charged_once(self):
        detect = self.make()
        self.assertEqual(detect(b'image'), {'text': '', 'completed': True})
        self.assertEqual(detect(b'image'), {'text': '', 'completed': True})
        self.assertEqual(len(self.client.calls), 1)
        self.assertEqual(self.client.calls[0], (b'image', 'image/jpeg', 'person'))
        detect(b'another')
        with self.assertRaises(BudgetExceeded): detect(b'third')

    def test_missing_cost_not_allowed(self):
        for actual in [None, Decimal('NaN'), Decimal('-1'), '.2']:
            with self.subTest(actual=actual), self.assertRaises(ValueError): self.make(actual=actual)

    def test_failed_provider_keeps_reservation(self):
        def fail(*args): raise TimeoutError('unknown billing')
        self.client.segment_image = fail
        detect = self.make()
        with self.assertRaises(TimeoutError): detect(b'image')
        with self.assertRaises(BudgetExceeded): detect(b'image')

    def test_invalid_image_does_not_spend(self):
        with self.assertRaises(ValueError): self.make()(b'')
        self.assertEqual(self.client.calls, [])
