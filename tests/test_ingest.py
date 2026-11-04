import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from app.pipeline.ingest import IngestError, ingest_video, probe_video


@unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg required')
class IngestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def make_video(self, name='input.mp4', size='160x240', duration=2):
        path = self.root / name
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', f'color=blue:size={size}:rate=12', '-t', str(duration), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(path)], check=True)
        return path

    def test_normalizes_portrait_and_trims(self):
        source = self.make_video()
        result = ingest_video(source, self.root / 'normalized.mp4', max_seconds=1, max_side=120, fps=10)
        meta = probe_video(result.path)
        self.assertLessEqual(max(meta.width, meta.height), 120)
        self.assertLessEqual(meta.duration, 1.1)
        self.assertTrue(result.trimmed)
        self.assertTrue(result.warnings)
        self.assertEqual(meta.codec, 'h264')

    def test_rejects_corrupt_empty_and_wrong_extension(self):
        for name, content in [('empty.mp4', b''), ('bad.mov', b'not video'), ('bad.txt', b'anything')]:
            path = self.root / name
            path.write_bytes(content)
            with self.subTest(name=name), self.assertRaises(IngestError):
                ingest_video(path, self.root / 'out.mp4')

    def test_rejects_audio_only_and_oversize(self):
        audio = self.root / 'audio.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'sine=frequency=1000', '-t', '1', '-c:a', 'aac', str(audio)], check=True)
        with self.assertRaises(IngestError):
            probe_video(audio)
        video = self.make_video()
        with self.assertRaises(IngestError):
            ingest_video(video, self.root / 'out.mp4', max_bytes=5)

    def test_output_does_not_replace_input(self):
        source = self.make_video()
        with self.assertRaises(IngestError):
            ingest_video(source, source)
        self.assertTrue(source.exists())
