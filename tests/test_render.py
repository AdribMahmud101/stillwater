import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.pipeline.render import RenderError, render_video


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'private.mp4'
        out = cv2.VideoWriter(str(self.source), cv2.VideoWriter_fourcc(*'mp4v'), 10, (160, 120))
        self.assertTrue(out.isOpened())
        for _ in range(3):
            out.write(np.full((120, 160, 3), 80, np.uint8))
        out.release()

    def test_overlays_only_tracked_frames(self):
        mask = {(x, y) for x in range(60, 90) for y in range(50, 80)}
        records = [{'track_id': 'a', 'concept': 'bucket', 'risk': 'high', 'frames': [{'frame': 1, 'mask': mask}]}]
        result = render_video(self.source, self.root / 'annotated.mp4', records)
        self.assertEqual(result.frame_count, 3)
        probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=codec_name,pix_fmt', '-of', 'json', str(result.path)], capture_output=True, text=True, check=True)
        stream = json.loads(probe.stdout)['streams'][0]
        self.assertEqual(stream['codec_name'], 'h264')
        self.assertEqual(stream['pix_fmt'], 'yuv420p')
        self.assertEqual(result.timeline[0]['frame_index'], 1)
        cap = cv2.VideoCapture(str(result.path))
        images = [cap.read()[1] for _ in range(3)]
        cap.release()
        self.assertTrue(all(x is not None for x in images))
        self.assertGreater(float(images[1][55:75, 65:85, 2].mean()), float(images[0][55:75, 65:85, 2].mean()) + 20)
        self.assertLess(abs(float(images[2][55:75, 65:85, 2].mean()) - float(images[0][55:75, 65:85, 2].mean())), 5)

    def test_invalid_or_missing_frames_never_publish(self):
        for frame, mask in [(1, set()), (99, {(1, 1)}), (1, {(-1, 0)})]:
            with self.subTest(frame=frame, mask=mask), self.assertRaises(RenderError):
                render_video(self.source, self.root / 'output.mp4', [{'track_id': 'a', 'concept': 'bucket', 'risk': 'high', 'frames': [{'frame': frame, 'mask': mask}]}])
            self.assertFalse((self.root / 'output.mp4').exists())
            self.assertFalse(list(self.root.glob('.render-*')))
