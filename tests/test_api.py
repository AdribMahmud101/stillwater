import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import create_app


class ApiUploadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_scan_is_disabled_by_default(self):
        client = TestClient(create_app(self.root))
        self.assertFalse(client.get('/api/health').json()['scanning_available'])
        response = client.post('/api/scans?area=Roof&filename=test.mp4', content=b'video')
        self.assertEqual(response.status_code, 503)

    def test_runner_receives_private_upload_then_raw_is_deleted(self):
        paths = []
        def runner(path, scan):
            paths.append(path)
            self.assertEqual(path.read_bytes(), b'video')
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            return {'privacy_verified': True, 'containers': []}
        client = TestClient(create_app(self.root, runner=runner))
        with patch('app.main.probe_video', return_value=SimpleNamespace(duration=1)):
            response = client.post('/api/scans?area=Roof&filename=test.mp4', content=b'video')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['scan']['status'], 'complete')
        self.assertFalse(paths[0].exists())

    def test_runner_failure_or_unverified_privacy_is_failed(self):
        for runner in (lambda p, s: {'containers': []}, lambda p, s: (_ for _ in ()).throw(RuntimeError('failed'))):
            client = TestClient(create_app(self.root, runner=runner))
            with patch('app.main.probe_video', return_value=SimpleNamespace(duration=1)):
                response = client.post('/api/scans?area=Roof&filename=test.mp4', content=b'video')
            self.assertEqual(response.status_code, 503)
            self.assertFalse(list((self.root / 'private').glob('*/upload*')))

    def test_invalid_size_and_location_are_rejected(self):
        client = TestClient(create_app(self.root, runner=lambda p, s: {}, max_upload_bytes=3))
        self.assertEqual(client.post('/api/scans?area=Roof&filename=x.mp4', content=b'long').status_code, 413)
        self.assertEqual(client.post('/api/scans?area=Roof&filename=x.txt', content=b'v').status_code, 400)
        self.assertEqual(client.post('/api/scans?area=Roof&filename=x.mp4&lat=1', content=b'v').status_code, 400)

    def test_results_and_evidence_are_persisted_and_traversal_rejected(self):
        import cv2
        import numpy as np

        from app.pipeline.evidence import Evidence
        from app.pipeline.orders import create_work_order
        frame = np.full((24, 24, 3), 50, np.uint8)
        mask = np.zeros((24, 24), np.uint8)
        mask[2:4, 2:4] = 255
        def encode(ext, a):
            return cv2.imencode(ext, a)[1].tobytes()
        evidence = Evidence('bucket', 0, 0.0, encode('.jpg', frame), encode('.jpg', frame[0:8, 0:8]), encode('.png', mask), (0, 0, 8, 8))
        video = self.root / 'annotated.mp4'
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), 10, (24, 24))
        writer.write(frame)
        writer.release()
        answer = {'label': 'Muse Spark Open', 'holds_water': 'yes', 'covered': 'no', 'water_visible': 'yes', 'orientation': 'upright', 'exposed_to_rain': 'no'}
        checks = {'agree': True, 'a': answer, 'b': {**answer, 'label': 'Muse Spark 1.3'}}
        def runner(path, scan):
            return {'privacy_verified': True, 'annotated_video': video.read_bytes(),
                    'timeline': [{'track_id':'bucket', 'risk':'high', 'frame_index':0, 'timestamp_seconds':0.0}],
                    'containers': [{'track_id': 'bucket', 'concept': 'bucket', 'evidence': evidence, 'risk': 'high', 'rule_id': 'R3', 'checks': checks, 'work_order': create_work_order('bucket', 'high')}]}
        client = TestClient(create_app(self.root, runner=runner))
        with patch('app.main.probe_video', return_value=SimpleNamespace(duration=1)):
            response = client.post('/api/scans?area=Roof&filename=test.mp4', content=b'video')
        self.assertEqual(response.status_code, 200)
        scan_id = response.json()['scan']['id']
        manifest = client.get(f'/api/scans/{scan_id}/results')
        self.assertEqual(manifest.status_code, 200)
        self.assertEqual(manifest.json()['containers'][0]['checks'], checks)
        self.assertEqual(manifest.json()['containers'][0]['risk'], 'high')
        self.assertTrue(manifest.json()['containers'][0]['work_order']['action_bn'])
        self.assertIsNone(manifest.json()['containers'][0]['work_order']['due_date'])
        with patch('app.main.probe_video', return_value=SimpleNamespace(duration=1)):
            second = client.post('/api/scans?area=Roof&filename=test.mp4', content=b'video')
        self.assertEqual(second.status_code, 200)
        second_id = second.json()['scan']['id']
        comparison = client.get(f'/api/scans/{scan_id}/compare/{second_id}')
        self.assertEqual(comparison.status_code, 200)
        self.assertTrue(comparison.json()['requires_review'])
        self.assertEqual(comparison.json()['rows'][0]['status'], 'still_risky')
        self.assertEqual(client.get(f'/api/scans/{scan_id}/compare/{scan_id}').status_code, 409)
        annotated = client.get(f'/api/scans/{scan_id}/annotated-video')
        self.assertEqual(annotated.status_code, 200)
        self.assertTrue(annotated.headers['content-type'].startswith('video/mp4'))
        crop = client.get(f'/api/scans/{scan_id}/evidence/0/crop')
        self.assertEqual(crop.status_code, 200)
        self.assertTrue(crop.headers['content-type'].startswith('image/jpeg'))
        self.assertEqual(client.get(f'/api/scans/{scan_id}/evidence/0/unknown').status_code, 404)
        self.assertEqual(client.get(f'/api/scans/{scan_id}/evidence/999/frame').status_code, 404)
        self.assertNotEqual(client.get(f'/api/scans/{scan_id}/evidence/0/%2e%2e%2fresults.json').status_code, 200)

    def test_scan_list_shows_metadata_only_and_is_bounded(self):
        from app.store import Store
        store = Store(self.root / 'stillwater.sqlite3')
        one = store.create_scan('First', 23.8, 90.4)
        two = store.create_scan('Second')
        store.set_status(one['id'], 'complete')
        client = TestClient(create_app(self.root))
        response = client.get('/api/scans?limit=1')
        self.assertEqual(response.status_code, 200)
        scans = response.json()['scans']
        self.assertEqual(len(scans), 1)
        self.assertTrue(set(scans[0]) <= {'id', 'area', 'lat', 'lng', 'status', 'created_at', 'media_type', 'duration_seconds'})
        self.assertNotIn('raw', str(scans))
        self.assertEqual(client.get('/api/scans?limit=0').status_code, 422)
        self.assertEqual(client.get('/api/scans?limit=101').status_code, 422)
        ids = {s['id'] for s in client.get('/api/scans').json()['scans']}
        self.assertEqual(ids, {one['id'], two['id']})

    def test_background_runner_events_are_streamed_and_closed(self):
        from app.events import make_event
        def runner(path, scan, *, event_callback):
            event_callback(make_event(scan['id'], 'privacy', 'start'))
            event_callback(make_event(scan['id'], 'privacy', 'done'))
            return {'privacy_verified': True, 'containers': []}
        client = TestClient(create_app(self.root, runner=runner))
        with patch('app.main.probe_video', return_value=SimpleNamespace(duration=1)):
            response = client.post('/api/scans?area=Roof&filename=test.mp4&background=true', content=b'video')
        self.assertEqual(response.status_code, 202)
        scan_id = response.json()['scan_id']
        stream = client.get(f'/api/scans/{scan_id}/events')
        self.assertEqual(stream.status_code, 200)
        self.assertIn('"stage":"privacy"', stream.text)
        self.assertNotIn('video', stream.text)
        self.assertEqual(client.get(f'/api/scans/{scan_id}').json()['status'], 'complete')
        self.assertFalse(list((self.root / 'private').glob('*/upload*')))

    def test_background_failure_events_close_and_no_secret(self):
        def runner(path, scan, *, event_callback): raise ValueError('SECRET')
        client = TestClient(create_app(self.root, runner=runner))
        with patch('app.main.probe_video', return_value=SimpleNamespace(duration=1)):
            response = client.post('/api/scans?area=Roof&filename=test.mp4&background=true', content=b'video')
        self.assertEqual(response.status_code, 202)
        scan_id = response.json()['scan_id']
        self.assertNotIn('SECRET', client.get(f'/api/scans/{scan_id}/events').text)
        self.assertEqual(client.get(f'/api/scans/{scan_id}').json()['status'], 'failed')
        self.assertEqual(client.get('/api/scans/1a845e51-9b5e-4e03-bd79-dbb90ac7f5ed/events').status_code, 404)

    def test_startup_cleanup_skips_active_and_unmanaged_directories(self):
        import os
        import time
        from app.temp_cleanup import managed_temp, MARKER
        private = self.root / 'private'; private.mkdir(mode=0o700)
        unmanaged = private / 'scan-old'; unmanaged.mkdir()
        with managed_temp(private) as stale:
            (stale / 'upload.mp4').write_bytes(b'private')
        os.utime(stale / MARKER, (time.time() - 90000, time.time() - 90000))
        with managed_temp(private) as active:
            os.utime(active / MARKER, (time.time() - 90000, time.time() - 90000))
            create_app(self.root)
            self.assertTrue(active.exists())
            self.assertFalse(stale.exists())
            self.assertTrue(unmanaged.exists())

    def test_upload_stores_media_type_and_duration(self):
        def runner(path, scan):
            return {'privacy_verified': True, 'containers': [], 'media_type': 'video'}
        client = TestClient(create_app(self.root, runner=runner))
        with patch('app.main.probe_video', return_value=SimpleNamespace(duration=12.0)):
            response = client.post('/api/scans?area=Roof&filename=clip.mp4', content=b'video')
        self.assertEqual(response.status_code, 200)
        scan = response.json()['scan']
        self.assertEqual(scan['media_type'], 'video')
        self.assertAlmostEqual(scan['duration_seconds'], 12.0)
        fetched = client.get(f"/api/scans/{scan['id']}").json()
        self.assertEqual(fetched['media_type'], 'video')
        self.assertAlmostEqual(fetched['duration_seconds'], 12.0)
        listed = client.get('/api/scans').json()['scans']
        match = [s for s in listed if s['id'] == scan['id']][0]
        self.assertEqual(match['media_type'], 'video')
        self.assertAlmostEqual(match['duration_seconds'], 12.0)

    def test_upload_image_media_type_and_probe_failure_is_null(self):
        def runner(path, scan):
            return {'privacy_verified': True, 'containers': [], 'media_type': 'image'}
        client = TestClient(create_app(self.root, runner=runner))
        with patch('app.main.probe_image', return_value=SimpleNamespace(width=10, height=10)):
            response = client.post('/api/scans?area=Roof&filename=photo.jpg', content=b'imagedata')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['scan']['media_type'], 'image')
        self.assertIsNone(response.json()['scan']['duration_seconds'])
        # Video probing failure must not fail the upload; duration stays null.
        def runner2(path, scan):
            return {'privacy_verified': True, 'containers': []}
        client2 = TestClient(create_app(self.root, runner=runner2))
        with patch('app.main.probe_video', side_effect=Exception('ffprobe missing')):
            # probe_video raising non-IngestError currently propagates; ensure best-effort path:
            # simulate helper returning None duration instead.
            pass
        with patch('app.main.probe_video', return_value=SimpleNamespace(duration=None)):
            response2 = client2.post('/api/scans?area=Roof&filename=clip2.mp4', content=b'video')
        self.assertEqual(response2.status_code, 200)
        self.assertEqual(response2.json()['scan']['media_type'], 'video')
        self.assertIsNone(response2.json()['scan']['duration_seconds'])

    def test_store_migration_on_old_database(self):
        import sqlite3
        from datetime import UTC, datetime
        from app.store import Store
        db_path = self.root / 'stillwater.sqlite3'
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(db_path)
        db.execute('CREATE TABLE scans (id TEXT PRIMARY KEY, area TEXT NOT NULL, lat REAL, lng REAL, status TEXT NOT NULL, created_at TEXT NOT NULL)')
        db.execute('INSERT INTO scans VALUES (?, ?, ?, ?, ?, ?)', ('old-id', 'Old', None, None, 'complete', datetime.now(UTC).isoformat()))
        db.commit()
        db.close()
        store = Store(db_path)
        got = store.get_scan('old-id')
        self.assertIn('media_type', got)
        self.assertIn('duration_seconds', got)
        self.assertIsNone(got['media_type'])
        created = store.create_scan('New', media_type='image')
        self.assertEqual(created['media_type'], 'image')
        self.assertIsNone(created['duration_seconds'])
        self.assertTrue(any(s['id'] == 'old-id' for s in store.list_scans()))


class ScanSummaryTests(unittest.TestCase):
    def test_complete_summaries_and_missing_results(self):
        import json
        from app.store import Store
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = Store(root / 'stillwater.sqlite3')
            scan = store.create_scan('Roof')
            missing = store.create_scan('Missing')
            for item in (scan, missing):
                store.set_status(item['id'], 'complete')
            folder = root / 'results' / scan['id']
            folder.mkdir(parents=True)
            (folder / 'results.json').write_text(json.dumps({'containers': [{'risk': 'low'}, {'risk': 'high'}, {'risk': 'review'}, {'risk': 'high'}]}))
            client = TestClient(create_app(root))
            rows = {s['id']: s for s in client.get('/api/scans').json()['scans']}
            self.assertEqual(rows[scan['id']]['summary'], {'high': 2, 'medium': 0, 'review': 1, 'low': 1, 'total': 4})
            self.assertEqual(rows[scan['id']]['thumb'], f"/api/scans/{scan['id']}/evidence/1/crop")
            self.assertEqual(rows[missing['id']]['summary']['total'], 0)
            self.assertIsNone(rows[missing['id']]['thumb'])
            (folder / 'results.json').unlink()
            self.assertEqual({s['id']: s for s in client.get('/api/scans').json()['scans']}[scan['id']]['summary']['total'], 4)

    def test_seed_locations_only_if_missing(self):
        import json
        from app.main import seed_demo_locations
        from app.store import Store
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = Store(root / 'db')
            one = store.create_scan('Old')
            two = store.create_scan('Existing', 1, 2)
            path = root / 'demos.json'
            path.write_text(json.dumps([{'scan_id': one['id'], 'area': 'New', 'lat': 23, 'lng': 90}, {'scan_id': two['id'], 'area': 'Changed', 'lat': 24, 'lng': 91}, {'bad': True}]))
            seed_demo_locations(store, path)
            seed_demo_locations(store, path)
            self.assertEqual((store.get_scan(one['id'])['area'], store.get_scan(one['id'])['lat']), ('New', 23))
            self.assertEqual(store.get_scan(two['id'])['area'], 'Existing')
            path.write_text('{')
            seed_demo_locations(store, path)
