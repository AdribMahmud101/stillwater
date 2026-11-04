import math
import os
import time
import urllib.error
import concurrent.futures
import unittest
from unittest.mock import patch

import app.pipeline.sam_track as sam_track_module
from app.pipeline.sam_track import track_video, TrackingError


def _fake_parse(raw, concept):
    return [{"track_id": f"id_{concept}", "concept": concept, "frames": [{"frame": 0, "score": 0.9, "box": [0, 0, 10, 10], "mask_rle": "rle", "mask_offset": [0, 0]}]}]


def _fake_convert(parsed, w, h):
    out = []
    for track in parsed:
        nt = dict(track)
        nfs = []
        for f in track["frames"]:
            nf = dict(f)
            nf["mask"] = {(0, 0)}
            nfs.append(nf)
        nt["frames"] = nfs
        out.append(nt)
    return out


def _fake_dedup(tracks, *args, **kwargs):
    return tracks


def sequential_reference(client, video, concepts, width, height, *, min_track_frames=3, det_min=0.35, merge_iou=0.5):
    file_id = client.upload_video(video)
    try:
        all_tracks = []
        for concept in concepts:
            raw = client.segment_video_file(file_id, concept)
            parsed = sam_track_module.parse_sam_video(raw, concept)
            for track in parsed:
                for frame in track["frames"]:
                    score = frame["score"]
                    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                        raise TrackingError('SAM returned missing or invalid confidence')
                track["track_id"] = f"{concept}:{track['track_id']}"
            converted = sam_track_module.to_dedup_tracks(parsed, width, height)
            for original, track in zip(parsed, converted):
                for source, frame in zip(original["frames"], track["frames"]):
                    frame.update({'box': source['box'], 'mask_rle': source['mask_rle'], 'mask_offset': source['mask_offset']})
            all_tracks.extend(converted)
        return sam_track_module.deduplicate_tracks(all_tracks, min_track_frames, det_min, merge_iou)
    finally:
        client.delete_file(file_id)


class TestSamTrackParallel(unittest.TestCase):
    def test_parallel_equals_sequential_with_out_of_order_delays(self):
        concepts = ["cat", "dog", "bird"]
        raws = {c: f"raw_{c}" for c in concepts}
        delays = {"cat": 0.04, "dog": 0.01, "bird": 0.02}

        class FakeClient:
            def __init__(self, r, d):
                self.raws = r
                self.delays = d
                self.deleted = []
            def upload_video(self, video):
                return "file-test123"
            def delete_file(self, fid):
                self.deleted.append(fid)
            def segment_video_file(self, fid, concept):
                time.sleep(self.delays.get(concept, 0))
                return self.raws[concept]

        with patch("app.pipeline.sam_track.parse_sam_video", side_effect=_fake_parse), \
             patch("app.pipeline.sam_track.to_dedup_tracks", side_effect=_fake_convert), \
             patch("app.pipeline.sam_track.deduplicate_tracks", side_effect=_fake_dedup):
            c1 = FakeClient(raws, delays)
            seq = sequential_reference(c1, b"video", concepts, 100, 100)
            c2 = FakeClient(raws, delays)
            par = track_video(c2, b"video", concepts, 100, 100)
            self.assertEqual(par, seq)
            self.assertEqual([t["concept"] for t in par], concepts)
            # delete called once each
            self.assertEqual(len(c1.deleted), 1)
            self.assertEqual(len(c2.deleted), 1)

    def test_failure_in_one_concept_raises_and_still_deletes_once(self):
        concepts = ["a", "b", "c"]
        raws = {"a": "raw_a", "c": "raw_c"}

        class FailingClient:
            def __init__(self):
                self.deleted = []
            def upload_video(self, v):
                return "file-test123"
            def delete_file(self, fid):
                self.deleted.append(fid)
            def segment_video_file(self, fid, concept):
                time.sleep(0.01)
                if concept == "b":
                    raise RuntimeError("boom")
                return raws.get(concept, "raw")

        with patch("app.pipeline.sam_track.parse_sam_video", side_effect=_fake_parse), \
             patch("app.pipeline.sam_track.to_dedup_tracks", side_effect=_fake_convert), \
             patch("app.pipeline.sam_track.deduplicate_tracks", side_effect=_fake_dedup):
            client = FailingClient()
            with self.assertRaises(RuntimeError):
                track_video(client, b"vid", concepts, 100, 100)
            self.assertEqual(client.deleted.count("file-test123"), 1)
            self.assertEqual(len(client.deleted), 1)

    def test_delete_after_workers_finished(self):
        # slow worker should still be waited for even when another fails fast
        concepts = ["slow", "fast_fail"]

        class Client:
            def __init__(self):
                self.deleted = []
                self.delete_time = None
                self.start = time.time()
            def upload_video(self, v):
                return "file-test123"
            def delete_file(self, fid):
                self.delete_time = time.time()
                self.deleted.append(fid)
            def segment_video_file(self, fid, concept):
                if concept == "slow":
                    time.sleep(0.06)
                    return "raw_slow"
                else:
                    time.sleep(0.005)
                    raise RuntimeError("fast boom")

        with patch("app.pipeline.sam_track.parse_sam_video", side_effect=_fake_parse), \
             patch("app.pipeline.sam_track.to_dedup_tracks", side_effect=_fake_convert), \
             patch("app.pipeline.sam_track.deduplicate_tracks", side_effect=_fake_dedup):
            client = Client()
            start = time.time()
            with self.assertRaises(RuntimeError):
                track_video(client, b"vid", concepts, 100, 100)
            elapsed = time.time() - start
            # should have waited for slow worker (~0.06) before delete
            self.assertGreaterEqual(elapsed, 0.05)
            self.assertEqual(len(client.deleted), 1)
            # delete happened after slow worker finished
            self.assertGreaterEqual(client.delete_time - start, 0.05)

    def test_invalid_confidence_raises_TrackingError(self):
        concepts = ["a"]

        class Client:
            def __init__(self):
                self.deleted = []
            def upload_video(self, v):
                return "file-test123"
            def delete_file(self, fid):
                self.deleted.append(fid)
            def segment_video_file(self, fid, concept):
                return "raw_a"

        def fake_parse_inf(raw, concept):
            return [{"track_id": "1", "concept": concept, "frames": [{"frame": 0, "score": float("inf"), "box": [0,0,10,10], "mask_rle": "rle", "mask_offset": [0,0]}]}]

        def fake_parse_bool(raw, concept):
            return [{"track_id": "1", "concept": concept, "frames": [{"frame": 0, "score": True, "box": [0,0,10,10], "mask_rle": "rle", "mask_offset": [0,0]}]}]

        def fake_parse_oob(raw, concept):
            return [{"track_id": "1", "concept": concept, "frames": [{"frame": 0, "score": 1.5, "box": [0,0,10,10], "mask_rle": "rle", "mask_offset": [0,0]}]}]

        for bad_parse in [fake_parse_inf, fake_parse_bool, fake_parse_oob]:
            with self.subTest(bad_parse=bad_parse.__name__):
                with patch("app.pipeline.sam_track.parse_sam_video", side_effect=bad_parse), \
                     patch("app.pipeline.sam_track.to_dedup_tracks", side_effect=_fake_convert), \
                     patch("app.pipeline.sam_track.deduplicate_tracks", side_effect=_fake_dedup):
                    client = Client()
                    with self.assertRaises(TrackingError):
                        track_video(client, b"vid", concepts, 100, 100)
                    self.assertEqual(len(client.deleted), 1)

    def test_worker_count_clamp(self):
        cases = [
            (None, 4),
            ("0", 1),
            ("1", 1),
            ("5", 5),
            ("10", 10),
            ("15", 10),
            ("-3", 1),
            ("abc", 4),
            ("", 4),
            (" 7 ", 7),
        ]

        class DummyClient:
            def upload_video(self, v):
                return "file-123"
            def delete_file(self, fid):
                pass
            def segment_video_file(self, fid, concept):
                return "raw"

        for env_val, expected in cases:
            with self.subTest(env_val=env_val, expected=expected):
                captured = {}
                orig = concurrent.futures.ThreadPoolExecutor

                def fake_executor(*args, **kwargs):
                    cw = kwargs.get("max_workers")
                    if cw is None and args:
                        cw = args[0]
                    captured["max_workers"] = cw
                    return orig(*args, **kwargs)

                with patch("concurrent.futures.ThreadPoolExecutor", side_effect=fake_executor):
                    # also patch the reference in sam_track module if it imported directly
                    with patch("app.pipeline.sam_track.concurrent.futures.ThreadPoolExecutor", side_effect=fake_executor):
                        with patch("app.pipeline.sam_track.parse_sam_video", side_effect=_fake_parse), \
                             patch("app.pipeline.sam_track.to_dedup_tracks", side_effect=_fake_convert), \
                             patch("app.pipeline.sam_track.deduplicate_tracks", side_effect=_fake_dedup):
                            # manage env
                            orig_env = os.environ.get("TRACK_WORKERS")
                            try:
                                if env_val is None:
                                    os.environ.pop("TRACK_WORKERS", None)
                                else:
                                    os.environ["TRACK_WORKERS"] = env_val
                                track_video(DummyClient(), b"vid", ["a"], 100, 100)
                            finally:
                                if orig_env is None:
                                    os.environ.pop("TRACK_WORKERS", None)
                                else:
                                    os.environ["TRACK_WORKERS"] = orig_env
                            self.assertEqual(captured.get("max_workers"), expected)

    def test_retry_transient_errors(self):
        concepts = ["a"]

        class RetryClient:
            def __init__(self):
                self.calls = 0
                self.deleted = []
            def upload_video(self, v):
                return "file-123"
            def delete_file(self, fid):
                self.deleted.append(fid)
            def segment_video_file(self, fid, concept):
                self.calls += 1
                if self.calls <= 2:
                    raise urllib.error.HTTPError("url", 429, "Too Many Requests", hdrs=None, fp=None)
                return "raw_a"

        with patch("app.pipeline.sam_track.parse_sam_video", side_effect=_fake_parse), \
             patch("app.pipeline.sam_track.to_dedup_tracks", side_effect=_fake_convert), \
             patch("app.pipeline.sam_track.deduplicate_tracks", side_effect=_fake_dedup), \
             patch("app.pipeline.sam_track.time.sleep") as mock_sleep, \
             patch("app.pipeline.sam_track.random.uniform", return_value=0.005):
            client = RetryClient()
            result = track_video(client, b"vid", concepts, 100, 100)
            self.assertEqual(client.calls, 3)
            self.assertEqual(mock_sleep.call_count, 2)
            first = mock_sleep.call_args_list[0][0][0]
            second = mock_sleep.call_args_list[1][0][0]
            self.assertGreater(second, first)
            self.assertEqual(len(client.deleted), 1)

    def test_retry_exhausted_raises(self):
        class FailClient:
            def upload_video(self, v):
                return "file-123"
            def delete_file(self, fid):
                pass
            def segment_video_file(self, fid, concept):
                raise urllib.error.HTTPError("url", 500, "Server Error", hdrs=None, fp=None)

        with patch("app.pipeline.sam_track.parse_sam_video", side_effect=_fake_parse), \
             patch("app.pipeline.sam_track.to_dedup_tracks", side_effect=_fake_convert), \
             patch("app.pipeline.sam_track.deduplicate_tracks", side_effect=_fake_dedup), \
             patch("app.pipeline.sam_track.time.sleep"), \
             patch("app.pipeline.sam_track.random.uniform", return_value=0.0):
            with self.assertRaises(urllib.error.HTTPError):
                track_video(FailClient(), b"vid", ["a"], 100, 100)

    def test_non_transient_not_retried(self):
        class Client:
            def __init__(self):
                self.calls = 0
            def upload_video(self, v):
                return "file-123"
            def delete_file(self, fid):
                pass
            def segment_video_file(self, fid, concept):
                self.calls += 1
                raise ValueError("bad request")

        with patch("app.pipeline.sam_track.parse_sam_video", side_effect=_fake_parse), \
             patch("app.pipeline.sam_track.to_dedup_tracks", side_effect=_fake_convert), \
             patch("app.pipeline.sam_track.deduplicate_tracks", side_effect=_fake_dedup), \
             patch("app.pipeline.sam_track.time.sleep") as mock_sleep:
            c = Client()
            with self.assertRaises(ValueError):
                track_video(c, b"vid", ["a"], 100, 100)
            self.assertEqual(c.calls, 1)
            mock_sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
