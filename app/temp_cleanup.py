"""Opt-in abandoned private temp cleanup with an advisory active-scan lock.

Only directories created by managed_temp are eligible. All scan workers must
keep the lock held for their entire use of the directory; otherwise cleanup
cannot distinguish an active worker from an abandoned one. Not yet wired to API.
"""
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import shutil
import stat
import tempfile
import time

MARKER = '.stillwater-managed'
LOCK = '.stillwater-lock'
SIGNATURE = b'stillwater-private-scan-v1\n'


class TempCleanupError(RuntimeError):
    pass


def _check_root(root):
    root = Path(root)
    try:
        info = root.lstat()
    except OSError as exc:
        raise TempCleanupError('private root unavailable') from exc
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
        raise TempCleanupError('private root must be owned by this user with mode 0700 or stricter')
    return root


@contextmanager
def managed_temp(root):
    """Create a private managed directory and hold its lock while in use.

    On exit it becomes eligible for cleanup; callers may remove it immediately.
    """
    root = _check_root(root)
    directory = Path(tempfile.mkdtemp(prefix='stillwater-scan-', dir=root))
    fd = os.open(directory / LOCK, os.O_CREAT | os.O_EXCL | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        marker = os.open(directory / MARKER, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        with os.fdopen(marker, 'wb') as output:
            output.write(SIGNATURE)
        yield directory
    finally:
        os.close(fd)


def cleanup_abandoned(root, *, older_than_seconds=3600):
    """Delete stale, marked, unlocked, owned directories; skip all others."""
    root = _check_root(root)
    if isinstance(older_than_seconds, bool) or not isinstance(older_than_seconds, (int, float)) or not 0 <= older_than_seconds < float('inf'):
        raise ValueError('age must be finite and nonnegative')
    removed = 0
    for path in root.iterdir():
        if not path.name.startswith('stillwater-scan-'):
            continue
        fd = None
        try:
            info = path.lstat()
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
                continue
            marker_path = path / MARKER
            marker_info = marker_path.lstat()
            if not stat.S_ISREG(marker_info.st_mode) or marker_info.st_uid != os.geteuid() or time.time() - marker_info.st_mtime < older_than_seconds:
                continue
            marker_fd = os.open(marker_path, os.O_RDONLY | os.O_NOFOLLOW)
            try:
                if os.read(marker_fd, len(SIGNATURE) + 1) != SIGNATURE:
                    continue
            finally:
                os.close(marker_fd)
            fd = os.open(path / LOCK, os.O_RDWR | os.O_NOFOLLOW)
            lock_info = os.fstat(fd)
            if not stat.S_ISREG(lock_info.st_mode) or lock_info.st_uid != os.geteuid():
                continue
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            shutil.rmtree(path)
            removed += 1
        except (FileNotFoundError, PermissionError, OSError):
            continue
        finally:
            if fd is not None:
                os.close(fd)
    return removed
