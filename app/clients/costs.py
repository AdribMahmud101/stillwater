"""Local cost ledger with atomic per-database reservations."""
import sqlite3
import time
import uuid
from decimal import Decimal
from pathlib import Path


class BudgetExceeded(Exception):
    pass


def _ensure_created_at_column(db):
    try:
        cols = [r[1] for r in db.execute('PRAGMA table_info(reservations)').fetchall()]
    except Exception:
        cols = []
    if 'created_at' not in cols:
        now = time.time()
        try:
            # Default to current epoch so old rows read as stale.
            db.execute('ALTER TABLE reservations ADD COLUMN created_at REAL DEFAULT %r' % (float(now),))
        except Exception:
            pass


class CostLedger:
    def __init__(self, path: Path, daily_budget: Decimal):
        if not isinstance(daily_budget, Decimal) or not daily_budget.is_finite() or daily_budget < 0:
            raise ValueError('budget must be nonnegative')
        self.path, self.daily_budget = path, daily_budget
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE IF NOT EXISTS costs (day TEXT NOT NULL, usd TEXT NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS reservations (id TEXT PRIMARY KEY, day TEXT NOT NULL, usd TEXT NOT NULL)')
            _ensure_created_at_column(db)
            # Backfill any NULL created_at as very old (epoch 0) so they reconcile.
            try:
                db.execute('UPDATE reservations SET created_at = 0 WHERE created_at IS NULL')
            except Exception:
                pass

    def record(self, day: str, usd: Decimal):
        if not isinstance(usd, Decimal) or not usd.is_finite() or usd < 0:
            raise ValueError('cost must be nonnegative')
        with sqlite3.connect(self.path) as db:
            db.execute('INSERT INTO costs VALUES (?, ?)', (day, str(usd)))

    def check(self, day: str, estimated_usd: Decimal):
        if not isinstance(estimated_usd, Decimal) or not estimated_usd.is_finite() or estimated_usd < 0:
            raise ValueError('estimate must be nonnegative')
        with sqlite3.connect(self.path) as db:
            total = sum((Decimal(row[0]) for row in db.execute('SELECT usd FROM costs WHERE day = ? UNION ALL SELECT usd FROM reservations WHERE day = ?', (day, day))), Decimal(0))
        if total + estimated_usd > self.daily_budget:
            raise BudgetExceeded('daily budget exceeded')

    def reserve(self, day: str, estimated_usd: Decimal):
        if not isinstance(estimated_usd, Decimal) or not estimated_usd.is_finite() or estimated_usd < 0:
            raise ValueError('estimate must be finite and nonnegative')
        with sqlite3.connect(self.path, isolation_level=None) as db:
            db.execute('BEGIN IMMEDIATE')
            try:
                _ensure_created_at_column(db)
                total = sum((Decimal(row[0]) for row in db.execute('SELECT usd FROM costs WHERE day = ? UNION ALL SELECT usd FROM reservations WHERE day = ?', (day, day))), Decimal(0))
                if total + estimated_usd > self.daily_budget:
                    db.rollback()
                    raise BudgetExceeded('daily budget exceeded')
                reservation = str(uuid.uuid4())
                now = time.time()
                try:
                    db.execute('INSERT INTO reservations VALUES (?, ?, ?, ?)', (reservation, day, str(estimated_usd), float(now)))
                except Exception:
                    # DB without created_at column (race): fall back to 3-col insert.
                    db.execute('INSERT INTO reservations VALUES (?, ?, ?)', (reservation, day, str(estimated_usd)))
                db.commit()
            except BudgetExceeded:
                raise
            except Exception:
                try:
                    db.rollback()
                except Exception:
                    pass
                raise
        return reservation

    def settle(self, reservation: str, actual_usd: Decimal):
        if not isinstance(actual_usd, Decimal) or not actual_usd.is_finite() or actual_usd < 0:
            raise ValueError('cost must be finite and nonnegative')
        with sqlite3.connect(self.path, isolation_level=None) as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT day, usd FROM reservations WHERE id = ?', (reservation,)).fetchone()
            if row is None:
                db.rollback()
                raise ValueError('unknown reservation')
            if actual_usd > Decimal(row[1]):
                db.rollback()
                raise ValueError('actual cost exceeds reservation; reconcile manually')
            db.execute('INSERT INTO costs VALUES (?, ?)', (row[0], str(actual_usd)))
            db.execute('DELETE FROM reservations WHERE id = ?', (reservation,))
            db.commit()

    def reconcile_stale(self, max_age_seconds, settle_usd=Decimal('0.35')):
        """Settle reservations older than max_age_seconds at settle_usd.

        Never settles above the reservation amount. Returns number reconciled.
        Safe on databases lacking the created_at column (migrates them).
        """
        try:
            age = float(max_age_seconds)
        except Exception:
            raise ValueError('max_age_seconds must be a non-negative number')
        import math as _math
        if not _math.isfinite(age) or age < 0:
            raise ValueError('max_age_seconds must be a finite non-negative number')
        if not isinstance(settle_usd, Decimal) or not settle_usd.is_finite() or settle_usd < 0:
            raise ValueError('settle_usd must be a finite non-negative Decimal')
        cutoff = time.time() - age
        count = 0
        with sqlite3.connect(self.path, isolation_level=None) as db:
            db.execute('BEGIN IMMEDIATE')
            try:
                _ensure_created_at_column(db)
                try:
                    rows = db.execute('SELECT id, day, usd, created_at FROM reservations').fetchall()
                except Exception:
                    rows = [(r[0], r[1], r[2], 0) for r in db.execute('SELECT id, day, usd FROM reservations').fetchall()]
                for rid, day, usd_s, created in rows:
                    try:
                        c = float(created) if created is not None else 0.0
                    except Exception:
                        c = 0.0
                    if c >= cutoff:
                        continue
                    try:
                        reserved = Decimal(str(usd_s))
                    except Exception:
                        continue
                    if not reserved.is_finite() or reserved < 0:
                        continue
                    actual = settle_usd if settle_usd <= reserved else reserved
                    try:
                        db.execute('INSERT INTO costs VALUES (?, ?)', (day, str(actual)))
                        db.execute('DELETE FROM reservations WHERE id = ?', (rid,))
                        count += 1
                    except Exception:
                        continue
                db.commit()
            except Exception:
                try:
                    db.rollback()
                except Exception:
                    pass
                raise
        return count
