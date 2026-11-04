"""Private content-addressed cache for JSON model inputs and responses.

Cache entries can contain sensitive derived data. The owner must restrict parent
paths, manage retention and never place API secrets in payloads or responses.
"""
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path


class CacheError(RuntimeError):
    pass


class CallCache:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        try:
            self.directory.mkdir(parents=True, mode=0o700, exist_ok=True)
            info = self.directory.lstat()
            if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid()
                    or info.st_mode & 0o077):
                raise CacheError('Cache directory must be owner-only and owned by the current user')
        except OSError:
            raise CacheError('Cache directory cannot be initialized') from None

    def _path(self, provider, model, payload):
        try:
            encoded = json.dumps([provider, model, payload], sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
        except (TypeError, ValueError):
            raise CacheError('Cache input cannot be encoded') from None
        return self.directory / (hashlib.sha256(encoded).hexdigest() + '.json')

    def get(self, provider, model, payload):
        path = self._path(provider, model, payload)
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            with os.fdopen(fd, 'r') as f:
                info = os.fstat(f.fileno())
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
                    raise CacheError('Cache entry is not private')
                return json.load(f)
        except FileNotFoundError:
            return None
        except (OSError, ValueError, UnicodeError):
            raise CacheError('Cache entry cannot be read') from None

    def put(self, provider, model, payload, response):
        path = self._path(provider, model, payload)
        name = None
        try:
            fd, name = tempfile.mkstemp(dir=self.directory, prefix='.tmp-')
            with os.fdopen(fd, 'w') as f:
                json.dump(response, f, ensure_ascii=False)
            os.replace(name, path)
        except (OSError, TypeError, ValueError):
            raise CacheError('Cache entry cannot be written') from None
        finally:
            if name is not None:
                try:
                    os.unlink(name)
                except FileNotFoundError:
                    pass
