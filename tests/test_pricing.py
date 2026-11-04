import unittest
from decimal import Decimal
from app.clients.pricing import token_cost, unit_cost


class PricingTests(unittest.TestCase):
    def test_token_cost_uses_explicit_rates(self):
        self.assertEqual(token_cost(1_000_000, 500_000, Decimal('1.25'), Decimal('4.25')), Decimal('3.375'))

    def test_unit_cost_and_invalid_input(self):
        self.assertEqual(unit_cost(2, Decimal('2.50'), 1000), Decimal('0.005'))
        with self.assertRaises(ValueError):
            token_cost(-1, 2, Decimal('1'), Decimal('2'))
        with self.assertRaises(ValueError):
            unit_cost(1, Decimal('NaN'), 1000)
