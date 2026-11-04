import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.pipeline.evidence import Evidence
from app.results import ResultsError, ResultsStore


def image(ext, array):
    ok, encoded = cv2.imencode(ext, array)
    assert ok
    return encoded.tobytes()


def evidence():
    frame = np.full((24, 24, 3), 50, np.uint8)
    mask = np.zeros((24, 24), np.uint8)
    mask[2:4, 3:5] = 255
    return Evidence('object/unsafe', 2, 0.2, image('.jpg', frame), image('.jpg', frame[0:8, 0:8]), image('.png', mask), (0, 0, 8, 8))


class ResultsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = ResultsStore(Path(self.tmp.name))
        self.scan = '71c13fee-6fd8-4702-b9c0-edc83bed9cf9'

    def test_atomic_persistence_and_load(self):
        doc = self.store.save(self.scan, [{'evidence': evidence(), 'risk': 'review', 'rule_id': 'R1'}], privacy_verified=True)
        self.assertEqual(self.store.load(self.scan), doc)
        self.assertEqual(doc['containers'][0]['track_id'], 'object/unsafe')
        for name in ('frame.jpg', 'crop.jpg', 'mask.png'):
            self.assertTrue((Path(self.tmp.name) / self.scan / '000000' / name).is_file())
        self.assertFalse((Path(self.tmp.name) / 'object').exists())
        with self.assertRaises(ResultsError):
            self.store.save(self.scan, [], privacy_verified=True)

    def test_checks_persist_and_old_results_load(self):
        answer = {'label': 'Muse Spark Open', 'holds_water': 'yes', 'covered': 'no', 'water_visible': 'unsure', 'orientation': 'upright', 'exposed_to_rain': None}
        checks = {'agree': False, 'a': answer, 'b': {**answer, 'label': 'Muse Spark 1.3'}}
        doc = self.store.save(self.scan, [{'evidence': evidence(), 'risk': 'review', 'rule_id': 'R1', 'checks': checks}], privacy_verified=True)
        self.assertEqual(self.store.load(self.scan)['containers'][0]['checks'], checks)
        self.assertEqual(doc['containers'][0]['checks'], checks)

    def test_rejects_malformed_checks(self):
        answer = {'label': 'Model', 'holds_water': 'yes', 'covered': 'no', 'water_visible': 'unsure', 'orientation': 'upright', 'exposed_to_rain': None}
        for checks in ({'agree': 1, 'a': answer, 'b': answer}, {'agree': True, 'a': {**answer, 'covered': 'maybe'}, 'b': answer}, {'agree': True, 'a': {**answer, 'raw': 'secret'}, 'b': answer}):
            with self.subTest(checks=checks), self.assertRaises(ResultsError):
                self.store.save(self.scan, [{'evidence': evidence(), 'risk': 'review', 'rule_id': 'R1', 'checks': checks}], privacy_verified=True)

    def test_requires_privacy_and_valid_scan_id(self):
        for scan, verified in ((self.scan, False), ('../outside', True)):
            with self.assertRaises(ResultsError):
                self.store.save(scan, [], privacy_verified=verified)
        self.assertFalse((Path(self.tmp.name) / self.scan).exists())

    def test_invalid_evidence_does_not_publish(self):
        e = evidence()
        invalid = Evidence(e.track_id, e.frame_index, e.timestamp_seconds, e.frame_jpeg, e.crop_jpeg, b'bad', e.crop_box)
        with self.assertRaises(ResultsError):
            self.store.save(self.scan, [{'evidence': invalid, 'risk': 'high', 'rule_id': 'R3'}], privacy_verified=True)
        self.assertFalse((Path(self.tmp.name) / self.scan).exists())
        self.assertEqual(list(Path(self.tmp.name).glob('.result-*')), [])

    def test_work_order_persists_bilingual_text_and_validates_link(self):
        from datetime import date

        from app.pipeline.orders import create_work_order
        e = evidence()
        order = create_work_order(e.track_id, 'high', due_date=date(2026, 11, 5), assignee='A')
        doc = self.store.save(self.scan, [{'evidence': e, 'risk': 'high', 'rule_id': 'R3', 'work_order': order}], privacy_verified=True)
        saved = doc['containers'][0]['work_order']
        self.assertEqual(saved['container_ids'], [e.track_id])
        self.assertEqual(saved['due_date'], '2026-11-05')
        self.assertEqual(saved['action_bn'], order.action_bn)
        self.assertEqual(self.store.load(self.scan), doc)

    def test_mismatched_order_is_not_published(self):
        from app.pipeline.orders import create_work_order
        order = create_work_order('wrong', 'low')
        with self.assertRaises(ResultsError):
            self.store.save(self.scan, [{'evidence': evidence(), 'risk': 'high', 'rule_id': 'R3', 'work_order': order}], privacy_verified=True)
        self.assertFalse((Path(self.tmp.name) / self.scan).exists())

    def test_annotated_video_and_timeline_saved(self):
        video_path = Path(self.tmp.name) / 'annotated.mp4'
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (24, 24))
        self.assertTrue(writer.isOpened())
        for _ in range(3):
            writer.write(np.full((24, 24, 3), 60, np.uint8))
        writer.release()
        timeline = [{'track_id': 'object/unsafe', 'risk': 'high', 'frame_index': 2, 'timestamp_seconds': 0.2}]
        doc = self.store.save(self.scan, [{'evidence': evidence(), 'risk': 'high', 'rule_id': 'R3'}], privacy_verified=True, annotated_video=video_path.read_bytes(), timeline=timeline)
        self.assertEqual(doc['timeline'], timeline)
        self.assertEqual(doc['annotated_video'], 'annotated.mp4')
        self.assertTrue((Path(self.tmp.name) / self.scan / 'annotated.mp4').is_file())

    def test_invalid_video_or_timeline_fails_without_publishing(self):
        for video, timeline in [(b'not video', []), (None, [{'track_id': 'unknown'}])]:
            with self.subTest(video=video), self.assertRaises(ResultsError):
                self.store.save(self.scan, [], privacy_verified=True, annotated_video=video, timeline=timeline)
            self.assertFalse((Path(self.tmp.name) / self.scan).exists())

    def test_comparison_features_derived_from_evidence_and_video(self):
        video_path = Path(self.tmp.name) / 'private-annotated.mp4'
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (24, 24))
        for _ in range(3):
            writer.write(np.full((24, 24, 3), 60, np.uint8))
        writer.release()
        timeline = [{'track_id': 'object/unsafe', 'risk': 'high', 'frame_index': 2, 'timestamp_seconds': .2}]
        doc = self.store.save(self.scan, [{'evidence': evidence(), 'risk': 'high', 'rule_id': 'R3', 'concept': 'bucket'}], privacy_verified=True, annotated_video=video_path.read_bytes(), timeline=timeline)
        row = doc['containers'][0]
        self.assertEqual(row['concept'], 'bucket')
        self.assertAlmostEqual(row['position'], 2/3)
        self.assertEqual(len(row['histogram']), 64)
        self.assertAlmostEqual(sum(row['histogram']), 1)
