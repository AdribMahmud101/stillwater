import unittest
from datetime import date

from app.pipeline.orders import create_work_order


class OrderTests(unittest.TestCase):
    def test_all_risk_levels_have_bilingual_actions(self):
        for risk in ('high', 'medium', 'low', 'review'):
            with self.subTest(risk=risk):
                order = create_work_order('container-1', risk)
                self.assertTrue(order.action_en)
                self.assertTrue(order.action_bn)
                self.assertIsNone(order.due_date)
                self.assertIsNone(order.assignee)
                self.assertEqual(order.container_ids, ('container-1',))

    def test_tire_action_and_explicit_assignment(self):
        order = create_work_order('one', 'high', object_label='tyre', due_date=date(2026, 11, 5), assignee='Worker')
        self.assertIn('Tyres', order.action_en)
        self.assertIn('টায়ার', order.action_bn)
        self.assertEqual(order.due_date, date(2026, 11, 5))
        self.assertEqual(order.assignee, 'Worker')

    def test_invalid_inputs_fail(self):
        for container_id, risk in [('', 'high'), ('one', 'unknown'), ('  ', 'low')]:
            with self.subTest(risk=risk), self.assertRaises(ValueError):
                create_work_order(container_id, risk)
        with self.assertRaises(ValueError):
            create_work_order('one', 'low', due_date='tomorrow')
