"""Persistent scan metadata; raw uploads are never stored here."""
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path


class Store:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS scans (id TEXT PRIMARY KEY, area TEXT NOT NULL, lat REAL, lng REAL, status TEXT NOT NULL, created_at TEXT NOT NULL)')
            self._ensure_columns(db)

    def _ensure_columns(self, db):
        try:
            cols = {row['name'] for row in db.execute('PRAGMA table_info(scans)').fetchall()}
        except Exception:
            cols = set()
        if 'media_type' not in cols:
            db.execute("ALTER TABLE scans ADD COLUMN media_type TEXT")
        if 'duration_seconds' not in cols:
            db.execute("ALTER TABLE scans ADD COLUMN duration_seconds REAL")

    def _ensure_columns_connect(self):
        with self._connect() as db:
            self._ensure_columns(db)

    def fail_interrupted(self):
        """Mark scans left 'running'/'queued' by a stopped process as failed (call once at startup)."""
        self._ensure_columns_connect()
        with self._connect() as db:
            db.execute("UPDATE scans SET status = 'failed' WHERE status IN ('running', 'queued')")

    def _connect(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def _normalize_media_type(media_type):
        if media_type is None:
            return None
        if isinstance(media_type, str) and media_type in ('image', 'video'):
            return media_type
        return None

    @staticmethod
    def _normalize_duration(duration_seconds):
        if duration_seconds is None:
            return None
        try:
            v = float(duration_seconds)
        except (TypeError, ValueError):
            return None
        # Guard against NaN/inf; store null instead.
        import math as _math
        if not _math.isfinite(v) or v < 0:
            return None
        return v

    def create_scan(self, area, lat=None, lng=None, media_type=None, duration_seconds=None):
        if not area or not area.strip():
            raise ValueError('area is required')
        if (lat is None) != (lng is None) or (lat is not None and not (-90 <= lat <= 90 and -180 <= lng <= 180)):
            raise ValueError('invalid coordinates')
        media_type = self._normalize_media_type(media_type)
        duration_seconds = self._normalize_duration(duration_seconds)
        scan_id = str(uuid.uuid4())
        self._ensure_columns_connect()
        with self._connect() as db:
            db.execute(
                'INSERT INTO scans (id, area, lat, lng, status, created_at, media_type, duration_seconds) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                (scan_id, area.strip(), lat, lng, 'queued', datetime.now(UTC).isoformat(), media_type, duration_seconds),
            )
        return self.get_scan(scan_id)

    def list_scans(self, limit=50):
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError('limit must be between 1 and 100')
        self._ensure_columns_connect()
        with self._connect() as db:
            rows = db.execute('SELECT id, area, lat, lng, status, created_at, media_type, duration_seconds FROM scans ORDER BY created_at DESC, id DESC LIMIT ?', (limit,)).fetchall()
        return [dict(row) for row in rows]

    def get_scan(self, scan_id):
        self._ensure_columns_connect()
        with self._connect() as db:
            row = db.execute('SELECT * FROM scans WHERE id = ?', (scan_id,)).fetchone()
        return dict(row) if row else None

    def set_status(self, scan_id, status):
        if status not in {'queued', 'running', 'complete', 'failed'}:
            raise ValueError('invalid status')
        with self._connect() as db:
            db.execute('UPDATE scans SET status = ? WHERE id = ?', (status, scan_id))

    def set_location(self, scan_id, area, lat, lng):
        if not isinstance(area, str) or not area.strip():
            raise ValueError('area is required')
        if isinstance(lat, bool) or isinstance(lng, bool) or not isinstance(lat, (int, float)) or not isinstance(lng, (int, float)) or not (-90 <= lat <= 90 and -180 <= lng <= 180):
            raise ValueError('invalid coordinates')
        with self._connect() as db:
            db.execute('UPDATE scans SET area = ?, lat = ?, lng = ? WHERE id = ? AND lat IS NULL', (area.strip(), lat, lng, scan_id))

    def set_media(self, scan_id, media_type=None, duration_seconds=None):
        media_type = self._normalize_media_type(media_type)
        duration_seconds = self._normalize_duration(duration_seconds)
        self._ensure_columns_connect()
        with self._connect() as db:
            if media_type is None and duration_seconds is None:
                return
            if media_type is not None and duration_seconds is not None:
                db.execute('UPDATE scans SET media_type = ?, duration_seconds = ? WHERE id = ?', (media_type, duration_seconds, scan_id))
            elif media_type is not None:
                db.execute('UPDATE scans SET media_type = ? WHERE id = ?', (media_type, scan_id))
            else:
                db.execute('UPDATE scans SET duration_seconds = ? WHERE id = ?', (duration_seconds, scan_id))
