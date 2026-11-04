import tempfile
import unittest
from pathlib import Path

from app.store import Store


class SetLocationTests(unittest.TestCase):
    def test_sets_area_and_fills_missing_coordinates_without_overwriting_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / 's.sqlite3')
            a = store.create_scan('old name')['id']
            b = store.create_scan('typed label', lat=1.5, lng=2.5)['id']
            store.set_location(a, 'Demo area', 23.8, 90.4)
            store.set_location(b, 'Demo area 2', 23.8, 90.4)
            rows = {r['id']: r for r in store.list_scans()}
            self.assertEqual((rows[a]['area'], rows[a]['lat'], rows[a]['lng']), ('Demo area', 23.8, 90.4))
            self.assertEqual((rows[b]['area'], rows[b]['lat'], rows[b]['lng']), ('Demo area 2', 1.5, 2.5))

    def test_rejects_bad_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / 's.sqlite3')
            a = store.create_scan('x')['id']
            with self.assertRaises(ValueError):
                store.set_location(a, ' ', 1, 1)
            with self.assertRaises(ValueError):
                store.set_location(a, 'ok', 95, 1)


if __name__ == '__main__':
    unittest.main()
