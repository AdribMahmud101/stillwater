import base64
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path


class FakeLedger:
    def __init__(self, path, budget):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.daily_budget = budget
        import sqlite3

        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS costs (day TEXT NOT NULL, usd TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS reservations (id TEXT PRIMARY KEY, day TEXT NOT NULL, usd TEXT NOT NULL)")

    def record(self, day, usd):
        import sqlite3

        with sqlite3.connect(self.path) as db:
            db.execute("INSERT INTO costs VALUES (?, ?)", (day, str(usd)))


def _write_video(path: Path, n_frames: int = 3, width: int = 64, height: int = 64) -> int:
    import cv2
    import numpy as np

    path = Path(path)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (width, height))
    assert writer.isOpened(), "test video writer did not open"
    for i in range(n_frames):
        color = (i * 40) % 256
        frame = np.full((height, width, 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    import cv2 as _cv2

    cap = _cv2.VideoCapture(str(path))
    count = 0
    try:
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
    finally:
        cap.release()
    assert count == n_frames, (count, n_frames)
    return count


def _patch_parsers_empty():
    import app.pipeline.frame_privacy as fp

    orig_parse = fp.parse_sam_video
    orig_tracks = fp.to_dedup_tracks
    fp.parse_sam_video = lambda text, prompt: []
    fp.to_dedup_tracks = lambda parsed, w, h: []
    return orig_parse, orig_tracks


def _restore_parsers(orig):
    import app.pipeline.frame_privacy as fp

    fp.parse_sam_video, fp.to_dedup_tracks = orig


class FakeSAM:
    def __init__(self, text="dummy", fail_at=None):
        self.text = text
        self.fail_at = fail_at
        self.calls = []

    def segment_image(self, image: bytes, mime: str, prompt: str):
        self.calls.append((bytes(image), mime, prompt))
        assert mime == "image/jpeg"
        assert prompt == "person"
        assert isinstance(image, bytes) and image[:2] == b"\xff\xd8"
        if self.fail_at is not None and len(self.calls) > self.fail_at:
            raise RuntimeError("detector boom")
        return self.text


_VALID_ATTRS = {
    "holds_water": "no",
    "covered": "yes",
    "water_visible": "no",
    "exposed_to_rain": "yes",
    "orientation": "upright",
    "object_label": "blue bucket",
    "evidence": "round container",
}


def _choices(content: str):
    return {"choices": [{"message": {"content": content}}]}


class FakeModalGlimmer:
    """Injectable fake for the Modal Glimmer raw client (chat interface)."""

    def __init__(self, content, seconds=10.0):
        self.content = content
        self.seconds = seconds
        self.calls = []
        self.model = "muse-glimmer"

    def chat(self, messages, **options):
        self.calls.append((messages, options))
        resp = _choices(self.content)
        return {**resp, "_stillwater": {"cached": False, "inference_seconds": self.seconds}}


class FakeBaseten:
    def __init__(self, content):
        self.content = content
        self.calls = []
        self.model = "baseten"

    def chat_with_meta(self, messages, **options):
        self.calls.append((messages, options))
        return (_choices(self.content), False)


class FakeMetaSparkRaw:
    """Fake Meta Spark raw client speaking the chat-completions interface."""

    def __init__(self, content, usage=None):
        self.content = content
        self.usage = usage or {"input_tokens": 100, "output_tokens": 50}
        self.calls = []
        self.model = "muse-spark-1.3"

    def chat(self, messages, **options):
        self.calls.append((messages, options))
        return {"choices": [{"message": {"content": self.content}}],
                "usage": {"prompt_tokens": self.usage["input_tokens"],
                          "completion_tokens": self.usage["output_tokens"],
                          "input_tokens": self.usage["input_tokens"],
                          "output_tokens": self.usage["output_tokens"]}}


class _FakeHTTPResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _meta_opener(body: dict, seen: dict):
    def _open(req, timeout=None):
        seen["timeout"] = timeout
        seen["url"] = req.full_url
        seen["payload"] = json.loads(req.data.decode())
        seen["headers"] = dict(req.header_items())
        return _FakeHTTPResponse(json.dumps(body).encode())
    return _open


class TestProviders(unittest.TestCase):
    def test_make_clients_keys(self):
        from app.pipeline import providers as P

        with tempfile.TemporaryDirectory() as d:
            env = {
                "MODEL_API_KEY": "k1",
                "BASETEN_API_KEY": "k2",
                "DAILY_BUDGET_USD": "5.00",
            }
            clients = P.make_clients(env, data_dir=d)
        for key in ("ledger", "sam_client", "witness_a", "witness_b", "scout",
                    "ingest", "privacy", "track", "probe", "extractor",
                    "upper_bound_usd", "cost_usd"):
            self.assertIn(key, clients, key)
        for key in ("ingest", "privacy", "scout", "track"):
            self.assertTrue(callable(clients[key]), key)
        self.assertIsInstance(clients["upper_bound_usd"], Decimal)

    def test_scan_reservation_is_scan_max(self):
        from app.pipeline import providers as P

        with tempfile.TemporaryDirectory() as d:
            env = {"MODEL_API_KEY": "k", "BASETEN_API_KEY": "k2",
                   "DAILY_BUDGET_USD": "5.00", "SCAN_MAX_USD": "1.00"}
            clients = P.make_clients(env, data_dir=d)
            self.assertEqual(clients["upper_bound_usd"], Decimal("1.00"))
            self.assertLessEqual(clients["upper_bound_usd"], clients["ledger"].daily_budget)
        with tempfile.TemporaryDirectory() as d:
            env = {"MODEL_API_KEY": "k", "BASETEN_API_KEY": "k2",
                   "DAILY_BUDGET_USD": "5.00"}
            clients = P.make_clients(env, data_dir=d)
            self.assertEqual(clients["upper_bound_usd"], Decimal("1.00"))

    def test_cost_usd_is_measured_not_constant(self):
        from app.pipeline import providers as P

        with tempfile.TemporaryDirectory() as d:
            env = {"MODEL_API_KEY": "k", "BASETEN_API_KEY": "k2", "DAILY_BUDGET_USD": "5.00"}
            clients = P.make_clients(env, data_dir=d)
            ledger = clients["ledger"]
            cost = clients["cost_usd"]
            before = Decimal(str(cost))
            self.assertEqual(before, Decimal("0"))
            ledger.record("2026-01-01", Decimal("1.25"))
            ledger.record("2026-01-01", Decimal("0.75"))
            after = Decimal(str(cost))
            self.assertEqual(after, Decimal("2.00"))
            self.assertNotEqual(after, before)

    def test_missing_stage_raises_not_implemented_not_fake(self):
        from app.pipeline import providers as P

        with tempfile.TemporaryDirectory() as d:
            env = {"MODEL_API_KEY": "k", "BASETEN_API_KEY": "k2", "DAILY_BUDGET_USD": "5.00"}
            clients = P.make_clients(env, data_dir=d)
            # Privacy is fail-closed: missing input cannot produce output.
            with self.assertRaises(Exception):
                clients["privacy"](Path(d) / "a.mp4", Path(d) / "b.mp4", clients["sam_client"])
            # Track must attempt the real stage (probe before any faked
            # bytes) and raise rather than return fake tracks.
            missing = Path(d) / "missing.mp4"
            with self.assertRaises(Exception):
                clients["track"](missing, ["bucket"])
            # Even with a present file the stage must not fake tracks:
            # without network/valid media the real track_video fails closed.
            present = Path(d) / "present.mp4"
            present.write_bytes(b"\x00" * 1024)
            with self.assertRaises(Exception):
                clients["track"](present, ["bucket"])

    def test_make_runner_from_env_gating(self):
        from app.pipeline import providers as P

        self.assertIsNone(P.make_runner_from_env({}))
        self.assertIsNone(P.make_runner_from_env({"SCANNING_ENABLED": "0",
                                                   "MODEL_API_KEY": "a", "BASETEN_API_KEY": "b",
                                                   "DAILY_BUDGET_USD": "5.00"}))
        self.assertIsNone(P.make_runner_from_env({"SCANNING_ENABLED": "1",
                                                   "MODEL_API_KEY": "", "BASETEN_API_KEY": "b",
                                                   "DAILY_BUDGET_USD": "5.00"}))
        self.assertIsNone(P.make_runner_from_env({"SCANNING_ENABLED": "1",
                                                   "MODEL_API_KEY": "a", "BASETEN_API_KEY": "",
                                                   "DAILY_BUDGET_USD": "5.00"}))
        self.assertIsNone(P.make_runner_from_env({"SCANNING_ENABLED": "1",
                                                   "MODEL_API_KEY": "a", "BASETEN_API_KEY": "b"}))

    def test_measured_cost_sums_ledger(self):
        from app.pipeline.providers import MeasuredCost

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ledger.sqlite3"
            led = FakeLedger(p, Decimal("5.00"))
            m = MeasuredCost(led)
            self.assertEqual(Decimal(str(m)), Decimal("0"))
            led.record("2026-11-03", Decimal("0.10"))
            self.assertEqual(Decimal(str(m)), Decimal("0.10"))

    def test_privacy_sends_every_frame_once(self):
        from app.pipeline import providers as P

        with tempfile.TemporaryDirectory() as d:
            env = {"MODEL_API_KEY": "k", "BASETEN_API_KEY": "k2",
                   "DAILY_BUDGET_USD": "5.00", "SAM_PER_FRAME": "0.001"}
            clients = P.make_clients(env, data_dir=d)
            src = Path(d) / "in.mp4"
            dst = Path(d) / "out.mp4"
            n = _write_video(src, n_frames=3)
            fake = FakeSAM(text="dummy")
            orig = _patch_parsers_empty()
            try:
                clients["privacy"](src, dst, fake)
            finally:
                _restore_parsers(orig)
            self.assertEqual(len(fake.calls), n)
            for img, mime, prompt in fake.calls:
                self.assertEqual(mime, "image/jpeg")
                self.assertEqual(prompt, "person")
                self.assertTrue(img[:2] == b"\xff\xd8")
            self.assertTrue(dst.exists())
            # Measured cost counts every frame (ledger recorded, no nested reserve).
            self.assertEqual(Decimal(str(clients["cost_usd"])), Decimal("0.001") * n)

    def test_privacy_failure_raises_and_publishes_nothing(self):
        from app.pipeline import providers as P

        with tempfile.TemporaryDirectory() as d:
            env = {"MODEL_API_KEY": "k", "BASETEN_API_KEY": "k2",
                   "DAILY_BUDGET_USD": "5.00", "SAM_PER_FRAME": "0.001"}
            clients = P.make_clients(env, data_dir=d)
            src = Path(d) / "in.mp4"
            dst = Path(d) / "out.mp4"
            _write_video(src, n_frames=3)
            fake = FakeSAM(text="dummy", fail_at=0)
            orig = _patch_parsers_empty()
            try:
                with self.assertRaises(Exception):
                    clients["privacy"](src, dst, fake)
            finally:
                _restore_parsers(orig)
            self.assertFalse(dst.exists())

    def test_privacy_budget_exhaustion_raises_and_publishes_nothing(self):
        from app.pipeline import providers as P

        with tempfile.TemporaryDirectory() as d:
            env = {"MODEL_API_KEY": "k", "BASETEN_API_KEY": "k2",
                   "DAILY_BUDGET_USD": "0.001", "SAM_PER_FRAME": "0.01"}
            clients = P.make_clients(env, data_dir=d)
            src = Path(d) / "in.mp4"
            dst = Path(d) / "out.mp4"
            _write_video(src, n_frames=2)
            fake = FakeSAM(text="dummy")
            orig = _patch_parsers_empty()
            try:
                with self.assertRaises(Exception):
                    clients["privacy"](src, dst, fake)
            finally:
                _restore_parsers(orig)
            self.assertFalse(dst.exists())

    def test_scan_80_frames_plus_3_chats_succeeds(self):
        from app.clients.costs import CostLedger
        from app.pipeline import providers as P

        with tempfile.TemporaryDirectory() as d:
            env = {"MODEL_API_KEY": "k", "BASETEN_API_KEY": "k2",
                   "DAILY_BUDGET_USD": "5.00", "SCAN_MAX_USD": "1.00",
                   "SAM_PER_FRAME": "0.001",
                   "WITNESS_INPUT_PER_M": "1.0", "WITNESS_OUTPUT_PER_M": "1.0"}
            clients = P.make_clients(env, data_dir=d)
            ledger = clients["ledger"]
            self.assertIsInstance(ledger, CostLedger)
            meter = P.ScanMeter(Decimal("1.00"))
            P.set_current_scan_meter(meter)
            try:
                per_frame = Decimal("0.001")
                for _ in range(80):
                    meter.check(per_frame)
                    meter.add(per_frame)
                # 3 small chat calls: measured from tiny usage.
                for _ in range(3):
                    upper = P._chat_upper_bound([{"content": "hi"}], 64, Decimal("1.0"), Decimal("1.0"))
                    meter.check(upper)
                    meter.add(Decimal("0.0001"))
            finally:
                P.clear_current_scan_meter()
            total = meter.total()
            self.assertLessEqual(total, Decimal("1.00"))
            # Ledger untouched by inner calls; single scan reservation settles.
            from app.pipeline.budget_runner import run_budgeted
            day = "2026-11-03"
            result = run_budgeted(ledger, day, clients["upper_bound_usd"],
                                  lambda: {"cost_usd": total})
            self.assertEqual(result["cost_usd"], total)

    def test_scan_meter_exceeding_raises(self):
        from app.clients.costs import BudgetExceeded
        from app.pipeline import providers as P

        meter = P.ScanMeter(Decimal("1.00"))
        meter.check(Decimal("0.50"))
        meter.add(Decimal("0.50"))
        with self.assertRaises(BudgetExceeded):
            meter.check(Decimal("0.60"))
        # Fail closed: total unchanged.
        self.assertEqual(meter.total(), Decimal("0.50"))

    def test_two_scans_do_not_share_meters(self):
        from app.pipeline import providers as P

        m1 = P.ScanMeter(Decimal("1.00"))
        m2 = P.ScanMeter(Decimal("1.00"))
        self.assertIsNot(m1, m2)
        m1.add(Decimal("0.25"))
        self.assertEqual(m1.total(), Decimal("0.25"))
        self.assertEqual(m2.total(), Decimal("0"))
        # Thread-local isolation: setting one scan meter does not leak.
        P.set_current_scan_meter(m1)
        try:
            self.assertIs(P.get_current_scan_meter(), m1)
        finally:
            P.clear_current_scan_meter()
        self.assertIsNone(P.get_current_scan_meter())

    def test_witness_b_wired_from_modal_fake(self):
        from app.pipeline import providers as P

        with tempfile.TemporaryDirectory() as d:
            env = {"MODEL_API_KEY": "k", "BASETEN_API_KEY": "k2",
                   "DAILY_BUDGET_USD": "5.00", "WITNESS_B": "glimmer"}
            fake = FakeModalGlimmer(json.dumps(_VALID_ATTRS), seconds=10.0)
            clients = P.make_clients(env, data_dir=d, modal_glimmer_client=fake)
            wb = clients["witness_b"]
            # Wired witness is a real Witness object, not the fallback dict.
            self.assertFalse(isinstance(wb, dict))
            self.assertEqual(getattr(wb, "provider", None), "glimmer")
            out = wb.describe(b"\xff\xd8fake", "image/jpeg")
            self.assertEqual(out, _VALID_ATTRS)
            self.assertEqual(len(fake.calls), 1)

    def test_witness_b_metering_added(self):
        from app.pipeline import providers as P

        with tempfile.TemporaryDirectory() as d:
            env = {"MODEL_API_KEY": "k", "BASETEN_API_KEY": "k2",
                   "DAILY_BUDGET_USD": "5.00", "WITNESS_B": "glimmer",
                   "GLIMMER_MAX_CALL_USD": "0.05",
                   "GLIMMER_GPU_PER_SECOND": "0.000222"}
            fake = FakeModalGlimmer(json.dumps(_VALID_ATTRS), seconds=10.0)
            clients = P.make_clients(env, data_dir=d, modal_glimmer_client=fake)
            wb = clients["witness_b"]
            meter = P.ScanMeter(Decimal("1.00"))
            P.set_current_scan_meter(meter)
            try:
                wb.describe(b"\xff\xd8fake", "image/jpeg")
            finally:
                P.clear_current_scan_meter()
            expected = Decimal("10.0") * Decimal("0.000222")
            self.assertEqual(meter.total(), expected)
            self.assertGreater(meter.total(), Decimal("0"))

    def test_witness_b_invalid_json_gives_unsure(self):
        from app.pipeline import providers as P
        from app.pipeline.verify import merge_witnesses

        with tempfile.TemporaryDirectory() as d:
            env = {"MODEL_API_KEY": "k", "BASETEN_API_KEY": "k2",
                   "DAILY_BUDGET_USD": "5.00", "WITNESS_B": "glimmer"}
            fake = FakeModalGlimmer("not-json{{{", seconds=1.0)
            clients = P.make_clients(env, data_dir=d, modal_glimmer_client=fake)
            wb = clients["witness_b"]
            out = wb.describe(b"\xff\xd8fake", "image/jpeg")
            # Invalid JSON -> all unsure (human review), never agreement.
            for k in ("holds_water", "covered", "water_visible", "exposed_to_rain"):
                self.assertEqual(out[k], "unsure")
            self.assertEqual(out["orientation"], "unsure")
            merged, agree = merge_witnesses(_VALID_ATTRS, out)
            self.assertFalse(agree)
            self.assertEqual(merged["holds_water"], "unsure")

    def test_witness_b_missing_credentials_clear_error(self):
        import os
        from unittest import mock
        from app.pipeline import providers as P

        with tempfile.TemporaryDirectory() as d:
            env = {"MODEL_API_KEY": "k", "BASETEN_API_KEY": "k2",
                   "DAILY_BUDGET_USD": "5.00", "WITNESS_B": "glimmer"}
            fake_home = Path(d) / "nohome"
            fake_home.mkdir(parents=True, exist_ok=True)
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch("pathlib.Path.home", return_value=fake_home):
                    with self.assertRaises(RuntimeError) as cm:
                        P.make_clients(env, data_dir=d)
                    msg = str(cm.exception)
                    self.assertIn("witness_b", msg)
                    self.assertIn("Glimmer", msg)

    def test_meta_spark_translation(self):
        from app.pipeline.providers import MetaSparkChat

        data_url = "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8fake").decode()
        body = {"status": "completed",
                "output": [{"type": "message",
                            "content": [{"type": "output_text", "text": json.dumps(_VALID_ATTRS)}]}],
                "usage": {"input_tokens": 10, "output_tokens": 20}}
        seen: dict = {}
        client = MetaSparkChat("sek", opener=_meta_opener(body, seen), max_retries=0)
        messages = [{"role": "system", "content": "sys-prompt"},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Describe this container."},
                        {"type": "image_url", "image_url": {"url": data_url}}]}]
        out = client.chat(messages)
        self.assertEqual(out["choices"][0]["message"]["content"], json.dumps(_VALID_ATTRS))
        payload = seen["payload"]
        self.assertEqual(payload["model"], "muse-spark-1.3")
        self.assertGreaterEqual(payload["max_output_tokens"], 3000)
        self.assertEqual(payload["reasoning"], {"effort": "low"})
        self.assertEqual(payload["instructions"], "sys-prompt")
        user_item = [i for i in payload["input"] if i["role"] == "user"][0]
        kinds = [c["type"] for c in user_item["content"]]
        self.assertIn("input_text", kinds)
        self.assertIn("input_image", kinds)
        img = [c for c in user_item["content"] if c["type"] == "input_image"][0]
        self.assertTrue(img["image_url"].startswith("data:image/jpeg;base64,"))
        self.assertEqual(seen["timeout"], 120)
        self.assertEqual(seen["url"], "https://api.meta.ai/v1/responses")
        # usage mapped for metering.
        self.assertEqual(out["usage"]["input_tokens"], 10)
        self.assertEqual(out["usage"]["output_tokens"], 20)

    def test_meta_spark_incomplete_raises(self):
        from app.pipeline.providers import MetaSparkChat

        body = {"status": "incomplete", "output": [], "usage": {}}
        seen: dict = {}
        client = MetaSparkChat("sek", opener=_meta_opener(body, seen), max_retries=0)
        with self.assertRaises(RuntimeError):
            client.chat([{"role": "user", "content": "hi"}])

    def test_meta_spark_metering_from_usage(self):
        from app.pipeline import providers as P

        with tempfile.TemporaryDirectory() as d:
            env = {"MODEL_API_KEY": "k", "BASETEN_API_KEY": "k2",
                   "DAILY_BUDGET_USD": "5.00", "WITNESS_B": "meta",
                   "META_SPARK_INPUT_PER_M": "1.0", "META_SPARK_OUTPUT_PER_M": "3.0"}
            fake = FakeMetaSparkRaw(json.dumps(_VALID_ATTRS),
                                    usage={"input_tokens": 1000, "output_tokens": 2000})
            clients = P.make_clients(env, data_dir=d, meta_spark_client=fake)
            wb = clients["witness_b"]
            meter = P.ScanMeter(Decimal("1.00"))
            P.set_current_scan_meter(meter)
            try:
                out = wb.describe(b"\xff\xd8fake", "image/jpeg")
            finally:
                P.clear_current_scan_meter()
            self.assertEqual(out, _VALID_ATTRS)
            expected = Decimal("1000") * Decimal("1.0") / Decimal(1000000) + Decimal("2000") * Decimal("3.0") / Decimal(1000000)
            self.assertEqual(meter.total(), expected)

    def test_witness_b_default_is_meta_and_distinct(self):
        from app.pipeline import providers as P

        with tempfile.TemporaryDirectory() as d:
            env = {"MODEL_API_KEY": "k", "BASETEN_API_KEY": "k2",
                   "DAILY_BUDGET_USD": "5.00"}
            fake = FakeMetaSparkRaw(json.dumps(_VALID_ATTRS))
            clients = P.make_clients(env, data_dir=d, meta_spark_client=fake)
            wa, wb = clients["witness_a"], clients["witness_b"]
            self.assertEqual(wa.label, "Muse Spark Open")
            self.assertEqual(wb.label, "Muse Spark 1.3")
            self.assertEqual(getattr(wb, "provider", None), "meta")
            self.assertEqual(getattr(wb, "model", None), "muse-spark-1.3")
            self.assertNotEqual(getattr(wa, "provider", "").lower(),
                                getattr(wb, "provider", "").lower())
            self.assertNotEqual(getattr(wa, "model", "").lower(),
                                getattr(wb, "model", "").lower())
            self.assertIsNone(clients["prewarm"])

    def test_witness_b_switch_glimmer_keeps_prewarm(self):
        from app.pipeline import providers as P

        with tempfile.TemporaryDirectory() as d:
            env = {"MODEL_API_KEY": "k", "BASETEN_API_KEY": "k2",
                   "DAILY_BUDGET_USD": "5.00", "WITNESS_B": "glimmer",
                   "MODAL_TOKEN_ID": "tid", "MODAL_TOKEN_SECRET": "tsec"}
            fake = FakeModalGlimmer(json.dumps(_VALID_ATTRS))
            clients = P.make_clients(env, data_dir=d, modal_glimmer_client=fake)
            self.assertEqual(clients["witness_b"].label, "Muse Glimmer 30B")
            self.assertEqual(getattr(clients["witness_b"], "provider", None), "glimmer")
            # prewarm hook present only for glimmer with credentials.
            self.assertTrue(callable(clients["prewarm"]))
            with tempfile.TemporaryDirectory() as d2:
                env2 = dict(env, WITNESS_B="meta")
                c2 = P.make_clients(env2, data_dir=d2,
                                    meta_spark_client=FakeMetaSparkRaw(json.dumps(_VALID_ATTRS)))
                self.assertIsNone(c2["prewarm"])

    def test_witness_b_fail_closed_no_substitution(self):
        from app.pipeline import providers as P

        with tempfile.TemporaryDirectory() as d:
            env = {"MODEL_API_KEY": "", "BASETEN_API_KEY": "k2",
                   "DAILY_BUDGET_USD": "5.00", "WITNESS_B": "meta"}
            with self.assertRaises(RuntimeError) as cm:
                P.make_clients(env, data_dir=d)
            self.assertIn("witness_b", str(cm.exception).lower())


if __name__ == "__main__":
    unittest.main()
