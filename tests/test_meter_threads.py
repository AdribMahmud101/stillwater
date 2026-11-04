import contextvars
import unittest
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

from app.pipeline import providers as P


class MeterThreadTests(unittest.TestCase):
    def test_meter_visible_in_worker_threads_via_copied_context(self):
        meter = P.ScanMeter(Decimal('1.00'))
        P.set_current_scan_meter(meter)
        try:
            with ThreadPoolExecutor(4) as ex:
                futs = [ex.submit(contextvars.copy_context().run,
                                  lambda: P.get_current_scan_meter().add(Decimal('0.01'))) for _ in range(8)]
                [f.result() for f in futs]
        finally:
            P.clear_current_scan_meter()
        self.assertEqual(meter.count, 8)
        self.assertEqual(meter.total(), Decimal('0.08'))


if __name__ == '__main__':
    unittest.main()
