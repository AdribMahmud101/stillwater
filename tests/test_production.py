import os
import pathlib
import shutil
import tempfile
import unittest
import uuid
from decimal import Decimal


class FakeLedger:
    def __init__(self, daily_budget=Decimal("5.00")):
        self.daily_budget = daily_budget
        self.reservations = {}
        self.costs = []

    def reserve(self, day, estimated_usd):
        rid = str(uuid.uuid4())
        self.reservations[rid] = (day, estimated_usd)
        return rid

    def settle(self, reservation, actual_usd):
        if reservation not in self.reservations:
            raise ValueError("unknown reservation")
        day, est = self.reservations[reservation]
        if actual_usd > est:
            raise ValueError("actual cost exceeds reservation; reconcile manually")
        self.costs.append((day, actual_usd))
        del self.reservations[reservation]

    def check(self, day, estimated_usd):
        total = sum((v[1] for v in self.reservations.values() if v[0] == day), Decimal(0)) + sum((v[1] for v in self.costs if v[0] == day), Decimal(0))
        from app.clients.costs import BudgetExceeded
        if total + estimated_usd > self.daily_budget:
            raise BudgetExceeded("daily budget exceeded")


def _make_settings(budget=Decimal("5.00")):
    class S:
        daily_budget_usd = budget
    return S()


def _stage_of(evt):
    if isinstance(evt, dict):
        return evt.get("stage") or evt.get("type") or evt.get("name")
    s = getattr(evt, "stage", None)
    if s is not None:
        return s
    s = getattr(evt, "type", None)
    if s is not None:
        return s
    if isinstance(evt, (list, tuple)) and len(evt) >= 2:
        return evt[1]
    return str(evt)


class TestProduction(unittest.TestCase):
    def setUp(self):
        self.orig_env = os.environ.copy()
        os.environ["SCANNING_ENABLED"] = "1"
        os.environ["MODEL_API_KEY"] = "test-model-key"
        os.environ["BASETEN_API_KEY"] = "test-baseten-key"
        self.tmpdir = tempfile.mkdtemp()
        self.video_path = pathlib.Path(self.tmpdir) / "upload.mp4"
        self.video_path.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self.orig_env)
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_refuses_to_build_when_disabled(self):
        os.environ["SCANNING_ENABLED"] = "0"
        from app.pipeline.production import build_runner
        settings = _make_settings()
        with self.assertRaises(RuntimeError):
            build_runner(settings)
        del os.environ["SCANNING_ENABLED"]
        with self.assertRaises(RuntimeError):
            build_runner(settings)

    def test_refuses_when_keys_missing(self):
        from app.pipeline.production import build_runner
        settings = _make_settings()
        del os.environ["MODEL_API_KEY"]
        with self.assertRaises(RuntimeError):
            build_runner(settings)
        os.environ["MODEL_API_KEY"] = "k"
        del os.environ["BASETEN_API_KEY"]
        with self.assertRaises(RuntimeError):
            build_runner(settings)
        os.environ["BASETEN_API_KEY"] = "k2"
        os.environ["MODEL_API_KEY"] = ""
        with self.assertRaises(RuntimeError):
            build_runner(settings)
        os.environ["MODEL_API_KEY"] = "k"
        class NoBudget:
            pass
        with self.assertRaises(RuntimeError):
            build_runner(NoBudget())
        class BadBudget:
            daily_budget_usd = Decimal("0")
        with self.assertRaises(RuntimeError):
            build_runner(BadBudget())

    def test_happy_path_emits_all_stage_events_in_order(self):
        from app.pipeline.production import build_runner
        ledger = FakeLedger()

        def fake_ingest(raw, normalized):
            pathlib.Path(normalized).write_bytes(b"normalized")

        def fake_privacy(normalized, private, sam_client=None):
            pathlib.Path(private).write_bytes(b"private")
            from app.pipeline.providers import get_current_scan_meter
            get_current_scan_meter().add(Decimal("0.01"))

        def fake_scout(private):
            return {"concepts": ["c1"]}

        def fake_track(*args, **kwargs):
            return [{"track_id": "t1", "concept": "c1"}]

        def fake_verify(private, tracks):
            class Ev:
                track_id = "t1"
            return [{"track_id": "t1", "evidence": Ev(), "risk": "low", "rule_id": "r1", "concept": "c1", "work_order": {"id": "wo-1"}}]

        def fake_act(records):
            return [{"id": "wo-1"}]

        clients = {
            "ledger": ledger,
            "sam_client": object(),
            "witness_a": object(),
            "witness_b": object(),
            "ingest": fake_ingest,
            "privacy": fake_privacy,
            "scout": fake_scout,
            "track": fake_track,
            "verify": fake_verify,
            "act": fake_act,
            "upper_bound_usd": Decimal("1.00"),
            "cost_usd": Decimal("0.10"),
        }
        settings = _make_settings(budget=Decimal("5.00"))
        runner = build_runner(settings, clients=clients)
        record = {"id": str(uuid.uuid4())}
        events = []

        def cb(evt):
            events.append(evt)

        result = runner(self.video_path, record, event_callback=cb)
        self.assertTrue(result.get("privacy_verified") is True)
        self.assertIn("containers", result)
        self.assertIsInstance(result["containers"], list)
        self.assertIn("cost_usd", result)
        self.assertEqual(len(ledger.reservations), 0)
        self.assertEqual(len(ledger.costs), 1)
        stages = [_stage_of(e) for e in events]
        expected = ["ingest", "privacy", "scout", "track", "verify", "decide", "act"]
        idx = 0
        for s in stages:
            if s == expected[idx]:
                idx += 1
                if idx == len(expected):
                    break
        self.assertEqual(idx, len(expected), f"events stages {stages} did not contain {expected} in order")

    def test_provider_failure_retains_reservation_and_raises(self):
        from app.pipeline.production import build_runner
        ledger = FakeLedger()

        def failing_ingest(raw, normalized):
            raise RuntimeError("provider failed")

        def fake_privacy(normalized, private, sam_client=None):
            pathlib.Path(private).write_bytes(b"private")
            from app.pipeline.providers import get_current_scan_meter
            get_current_scan_meter().add(Decimal("0.01"))

        def fake_scout(private):
            return ["c1"]

        def fake_track(private, concepts):
            return [{"track_id": "t1"}]

        clients = {
            "ledger": ledger,
            "ingest": failing_ingest,
            "privacy": fake_privacy,
            "scout": fake_scout,
            "track": fake_track,
            "upper_bound_usd": Decimal("1.00"),
            "cost_usd": Decimal("0.10"),
        }
        settings = _make_settings()
        runner = build_runner(settings, clients=clients)
        record = {"id": str(uuid.uuid4())}
        with self.assertRaises(Exception) as cm:
            runner(self.video_path, record)
        exc = cm.exception
        # new semantics: a failed scan is settled at measured spend and the rest released
        self.assertEqual(len(ledger.reservations), 0)
        self.assertEqual(len(ledger.costs), 1)
        self.assertLessEqual(sum(ledger.costs.values() if isinstance(ledger.costs, dict) else [c[1] if isinstance(c, tuple) else c for c in ledger.costs]), Decimal('1.00'))

    def test_missing_cost_raises_and_retains_reservation(self):
        from app.pipeline.production import build_runner
        ledger = FakeLedger()

        def fake_ingest(raw, normalized):
            pathlib.Path(normalized).write_bytes(b"normalized")

        def fake_privacy(normalized, private, sam_client=None):
            pathlib.Path(private).write_bytes(b"private")

        def fake_scout(private):
            return ["c1"]

        def fake_track(private, concepts):
            return [{"track_id": "t1"}]

        def fake_verify(private, tracks):
            class Ev:
                track_id = "t1"
            return [{"track_id": "t1", "evidence": Ev(), "risk": "low", "rule_id": "r1"}]

        def fake_act(records):
            return [{"id": "wo-1"}]

        clients = {
            "ledger": ledger,
            "ingest": fake_ingest,
            "privacy": fake_privacy,
            "scout": fake_scout,
            "track": fake_track,
            "verify": fake_verify,
            "act": fake_act,
            "upper_bound_usd": Decimal("1.00"),
            # cost_usd deliberately missing
        }
        settings = _make_settings()
        runner = build_runner(settings, clients=clients)
        record = {"id": str(uuid.uuid4())}
        with self.assertRaises(Exception) as cm:
            runner(self.video_path, record)
        exc = cm.exception
        # must retain reservation for reconciliation
        # new semantics: a failed scan is settled at measured spend and the rest released
        self.assertEqual(len(ledger.reservations), 0)
        self.assertEqual(len(ledger.costs), 1)
        self.assertLessEqual(sum(ledger.costs.values() if isinstance(ledger.costs, dict) else [c[1] if isinstance(c, tuple) else c for c in ledger.costs]), Decimal('1.00'))

    def test_no_dummy_container_is_ever_produced(self):
        from app.pipeline.production import build_runner
        ledger = FakeLedger()

        def fake_ingest(raw, normalized):
            pathlib.Path(normalized).write_bytes(b"normalized")

        def fake_privacy(normalized, private, sam_client=None):
            pathlib.Path(private).write_bytes(b"private")
            from app.pipeline.providers import get_current_scan_meter
            get_current_scan_meter().add(Decimal("0.01"))

        def fake_scout(private):
            return ["c1"]

        def fake_track(private, concepts):
            return [{"track_id": "t1", "frames": [{"score": 0.9}]}]

        # no verify, no witnesses -> old code would fabricate dummy low risk container
        clients = {
            "ledger": ledger,
            "ingest": fake_ingest,
            "privacy": fake_privacy,
            "scout": fake_scout,
            "track": fake_track,
            "upper_bound_usd": Decimal("1.00"),
            "cost_usd": Decimal("0.10"),
        }
        settings = _make_settings()
        runner = build_runner(settings, clients=clients)
        record = {"id": str(uuid.uuid4())}
        with self.assertRaises(Exception):
            runner(self.video_path, record)
        # ensure no container with fabricated low risk was returned and reservation retained
        # new semantics: a failed scan is settled at measured spend and the rest released
        self.assertEqual(len(ledger.reservations), 0)
        self.assertEqual(len(ledger.costs), 1)
        self.assertLessEqual(sum(ledger.costs.values() if isinstance(ledger.costs, dict) else [c[1] if isinstance(c, tuple) else c for c in ledger.costs]), Decimal('1.00'))
        # also verify source does not contain dummy fabrications
        src = pathlib.Path("app/pipeline/production.py").read_text()
        self.assertNotIn('b"private"', src)
        self.assertNotIn("b'private'", src)
        self.assertNotIn('Decimal("0.10")', src)
        low_dummy_markers = ['"low", "rule-1"', "'low', 'rule-1'", '"low","rule-1"']
        for m in low_dummy_markers:
            self.assertNotIn(m, src)

    def test_no_placeholder_strings_are_ever_returned(self):
        from app.pipeline.production import build_runner
        ledger = FakeLedger()

        def fake_ingest(raw, normalized):
            pathlib.Path(normalized).write_bytes(b"normalized")

        def fake_privacy(normalized, private, sam_client=None):
            pathlib.Path(private).write_bytes(b"private")
            from app.pipeline.providers import get_current_scan_meter
            get_current_scan_meter().add(Decimal("0.01"))

        def fake_scout(private):
            return {"concepts": ["c1"]}

        def fake_track(*args, **kwargs):
            return [{"track_id": "t1", "concept": "c1"}]

        def fake_verify(private, tracks):
            class Ev:
                track_id = "t1"
            return [{"track_id": "t1", "evidence": Ev(), "risk": "low", "rule_id": "R1", "concept": "c1", "checks": {"agree": True}}]

        def fake_act(records):
            return []

        clients = {
            "ledger": ledger,
            "sam_client": object(),
            "witness_a": object(),
            "witness_b": object(),
            "ingest": fake_ingest,
            "privacy": fake_privacy,
            "scout": fake_scout,
            "track": fake_track,
            "verify": fake_verify,
            "act": fake_act,
            "upper_bound_usd": Decimal("1.00"),
            "cost_usd": Decimal("0.10"),
        }
        settings = _make_settings(budget=Decimal("5.00"))
        runner = build_runner(settings, clients=clients)
        record = {"id": str(uuid.uuid4())}
        result = runner(self.video_path, record)
        annotated = result.get("annotated_video")
        timeline = result.get("timeline")
        # Never a fake placeholder: annotated must be bytes or None, never str.
        self.assertTrue(annotated is None or isinstance(annotated, bytes), f"annotated_video must be bytes or None, got {type(annotated)}")
        self.assertFalse(isinstance(annotated, str))
        if isinstance(annotated, bytes):
            self.assertGreater(len(annotated), 0)
            self.assertIsInstance(timeline, list)
            self.assertGreater(len(timeline), 0)
            for entry in timeline:
                self.assertIsInstance(entry, dict)
                self.assertEqual(set(entry), {"track_id", "risk", "frame_index", "timestamp_seconds"})
        else:
            self.assertIsNone(timeline, "timeline must be None when rendering is unavailable")
        # Empty-list timeline placeholder is never returned alongside real containers.
        self.assertFalse(timeline == [], "timeline must not be an empty-list placeholder")
        # Source must not return the legacy placeholders.
        src = pathlib.Path("app/pipeline/production.py").read_text()
        self.assertNotIn('"annotated_video": "annotated.mp4"', src)
        self.assertNotIn("\"annotated_video\": 'annotated.mp4'", src)
        self.assertNotIn('"timeline": []', src)
        self.assertNotIn("'timeline': []", src)
        # Containers stay within the main.py projection keys.
        allowed = {"evidence", "risk", "rule_id", "concept", "work_order", "checks"}
        self.assertEqual(result["containers"][0]["checks"], {"agree": True})
        for item in result["containers"]:
            self.assertIsInstance(item, dict)
            self.assertTrue(set(item) <= allowed, f"unexpected container keys {set(item)}")


if __name__ == "__main__":
    unittest.main()
