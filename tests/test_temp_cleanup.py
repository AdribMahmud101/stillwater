import tempfile
import unittest
from pathlib import Path
from app.temp_cleanup import managed_temp, cleanup_abandoned, TempCleanupError


class TempCleanupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'private'; self.root.mkdir(mode=0o700)

    def test_active_directory_not_removed_and_closed_directory_removed(self):
        with managed_temp(self.root) as directory:
            (directory / 'upload.mp4').write_bytes(b'private')
            self.assertEqual(cleanup_abandoned(self.root, older_than_seconds=0), 0)
            self.assertTrue(directory.exists())
        self.assertEqual(cleanup_abandoned(self.root, older_than_seconds=0), 1)
        self.assertFalse(directory.exists())

    def test_unrelated_and_unmarked_directories_untouched(self):
        unrelated = self.root / 'documents'; unrelated.mkdir()
        unmarked = self.root / 'scan-unmarked'; unmarked.mkdir()
        self.assertEqual(cleanup_abandoned(self.root, older_than_seconds=0), 0)
        self.assertTrue(unrelated.exists() and unmarked.exists())

    def test_recent_managed_directory_is_retained(self):
        with managed_temp(self.root) as directory: pass
        self.assertEqual(cleanup_abandoned(self.root, older_than_seconds=3600), 0)
        self.assertTrue(directory.exists())

    def test_insecure_root_refused(self):
        self.root.chmod(0o755)
        with self.assertRaises(TempCleanupError): cleanup_abandoned(self.root, older_than_seconds=0)
        with self.assertRaises(TempCleanupError):
            with managed_temp(self.root): pass
