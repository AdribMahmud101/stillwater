import os
import tempfile
import unittest
from pathlib import Path
from app.clients.cache import CallCache, CacheError

class CachePrivacyTests(unittest.TestCase):
    def test_new_cache_directory_and_files_are_private(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'cache'
            cache=CallCache(path)
            cache.put('provider','model',{'image':'sensitive'}, {'text':'private'})
            self.assertEqual(path.stat().st_mode & 0o777, 0o700)
            entry=next(path.iterdir())
            self.assertEqual(entry.stat().st_mode & 0o777, 0o600)
            self.assertNotIn('sensitive',entry.name)
            self.assertEqual(cache.get('provider','model',{'image':'sensitive'}),{'text':'private'})

    def test_insecure_existing_directory_is_not_modified(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'shared'; path.mkdir(mode=0o755); os.chmod(path,0o755)
            with self.assertRaises(CacheError): CallCache(path)
            self.assertEqual(path.stat().st_mode & 0o777, 0o755)

    def test_corrupt_cache_error_does_not_expose_contents(self):
        with tempfile.TemporaryDirectory() as d:
            cache=CallCache(Path(d))
            cache.put('p','m',{}, {'value':1})
            next(Path(d).iterdir()).write_text('secret-model-output')
            with self.assertRaises(CacheError) as caught: cache.get('p','m',{})
            self.assertNotIn('secret-model-output',str(caught.exception))
