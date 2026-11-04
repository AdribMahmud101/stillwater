import tempfile
import unittest
from pathlib import Path

from app.store import Store


class StoreTests(unittest.TestCase):
    def test_create_and_update_scan(self):
        with tempfile.TemporaryDirectory() as temp:
            store = Store(Path(temp) / 'data.sqlite3')
            scan = store.create_scan('Mirpur', 23.8, 90.4)
            self.assertEqual(scan['status'], 'queued')
            self.assertEqual(store.get_scan(scan['id'])['area'], 'Mirpur')
            store.set_status(scan['id'], 'running')
            self.assertEqual(store.get_scan(scan['id'])['status'], 'running')
            self.assertIsNone(store.get_scan('missing'))

    def test_reject_invalid_coordinates(self):
        with tempfile.TemporaryDirectory() as temp:
            store = Store(Path(temp) / 'data.sqlite3')
            with self.assertRaises(ValueError):
                store.create_scan('X', 91, 0)
