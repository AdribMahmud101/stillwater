import asyncio
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import modal_app


class ModalAppTests(unittest.TestCase):
    def test_config_constants(self):
        self.assertEqual(modal_app.MAX_CONTAINERS, 1)
        self.assertEqual(modal_app.DATA_DIR, Path('/data'))
        self.assertEqual(modal_app.SCALEDOWN_WINDOW, 600)
        self.assertEqual(modal_app.TIMEOUT, 3600)
        self.assertEqual(modal_app.MAX_CONCURRENT_INPUTS, 100)

    def test_modal_app_and_resources_names(self):
        app_name = modal_app.app.name if hasattr(modal_app.app, "name") else modal_app.app._name
        self.assertEqual(app_name, "stillwater-web")
        self.assertEqual(modal_app.volume.name, "stillwater-web-data")
        self.assertEqual(modal_app.secret.name, "stillwater-secrets")

    def test_docstring_accurate(self):
        doc = modal_app.__doc__ or ""
        self.assertNotIn("Scanning is intentionally disabled", doc)
        self.assertIn("raw", doc.lower())
        self.assertIn("detector", doc.lower())
        self.assertIn("SCAN_MAX_USD", doc)
        self.assertIn("sqlite", doc.lower())

    def test_wrap_runner_with_volume_commit(self):
        self.assertIsNone(modal_app.wrap_runner_with_volume_commit(None))

        mock_volume = MagicMock()
        mock_runner = MagicMock(return_value={"status": "ok"})
        wrapped = modal_app.wrap_runner_with_volume_commit(mock_runner, volume_target=mock_volume)

        result = wrapped("test_source.mp4", {"id": "scan-1"})
        self.assertEqual(result, {"status": "ok"})
        mock_runner.assert_called_once_with("test_source.mp4", {"id": "scan-1"})
        mock_volume.commit.assert_called_once()

    def test_wrap_runner_commit_exception_handled(self):
        mock_volume = MagicMock()
        mock_volume.commit.side_effect = RuntimeError("commit failed")
        mock_runner = MagicMock(return_value={"status": "ok"})
        wrapped = modal_app.wrap_runner_with_volume_commit(mock_runner, volume_target=mock_volume)

        result = wrapped("test_source.mp4", {"id": "scan-1"})
        self.assertEqual(result, {"status": "ok"})

    def test_build_web_app_reloads_volume_and_wires_runner(self):
        calls = []
        mock_volume = MagicMock()
        mock_runner = MagicMock()

        with (
            patch.object(modal_app, 'volume', mock_volume),
            patch('app.pipeline.providers.make_runner_from_env', return_value=mock_runner),
        ):
            app = modal_app.build_web_app(
                factory=lambda *args, **kwargs: calls.append((args, kwargs)) or 'dummy_app'
            )

        self.assertEqual(app, 'dummy_app')
        mock_volume.reload.assert_called_once()
        self.assertEqual(calls[0][0], (Path('/data'),))
        passed_runner = calls[0][1]['runner']
        self.assertIsNotNone(passed_runner)
        passed_runner("test.mp4", {"id": "scan-2"})
        mock_runner.assert_called_once_with("test.mp4", {"id": "scan-2"})
        mock_volume.commit.assert_called_once()

    def test_build_web_app_middleware_commit_on_scan(self):
        mock_volume = MagicMock()
        fake_app = MagicMock()
        registered_middleware = []

        def fake_middleware(middleware_type):
            def decorator(fn):
                registered_middleware.append((middleware_type, fn))
                return fn
            return decorator

        fake_app.middleware = fake_middleware

        with patch.object(modal_app, 'volume', mock_volume):
            modal_app.build_web_app(factory=lambda *args, **kwargs: fake_app)

            self.assertTrue(len(registered_middleware) > 0)
            mw_type, mw_fn = registered_middleware[0]
            self.assertEqual(mw_type, 'http')

            req = MagicMock()
            req.method = "POST"
            req.url.path = "/api/scans"

            async def call_next(r):
                resp = MagicMock()
                resp.status_code = 200
                return resp

            asyncio.run(mw_fn(req, call_next))
            mock_volume.commit.assert_called_once()

    def test_web_function_modal_spec(self):
        web_fn = modal_app.web
        orig = next(v for k, v in web_fn.__dict__.items() if k.startswith('_sync_original'))
        info = orig._function_info

        self.assertEqual(info.timeout, 3600)
        self.assertIsNotNone(info.concurrency_info)
        self.assertEqual(info.concurrency_info.max_inputs, 100)
        self.assertIn('/data', info.volumes)
        self.assertEqual(info.volumes['/data'].name, 'stillwater-web-data')
        self.assertIsNotNone(info.web_info)
        self.assertTrue(info.web_info.unauthenticated)
