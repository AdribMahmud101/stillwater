import unittest
from pathlib import Path

from app.pipeline.sam_parse import SAMParseError, parse_sam_video


class SAMParseTest(unittest.TestCase):
    def test_captured_mask_converts_to_track(self):
        # Meta's published parser conformance fixture, replayed as video frame 0.
        text = Path('tests/fixtures/sam_capture.txt').read_text()
        tracks = parse_sam_video(text, 'bucket')
        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracks[0]['track_id'], '0')
        frame = tracks[0]['frames'][0]
        self.assertEqual(frame['frame_index'], 0)
        self.assertEqual(frame['box'], [513, 371, 524, 382])
        self.assertEqual(frame['mask_rle']['size'], [11, 11])
        self.assertEqual(frame['mask_offset'], [513, 371])

    def test_documented_box_confidence_is_preserved(self):
        text = Path('tests/fixtures/sam_capture.txt').read_text()
        text = text.replace(';w=1440;h=900|>', ';w=1440;h=900;c=0.87|>', 1)
        self.assertEqual(parse_sam_video(text, 'bucket')[0]['frames'][0]['score'], 0.87)

    def test_malformed_output_is_rejected(self):
        with self.assertRaises(SAMParseError):
            parse_sam_video('<0f>bad', 'bucket')

class RecordedVideoTests(unittest.TestCase):
    def test_real_synthetic_video_response_has_ten_frames(self):
        from pathlib import Path
        raw = (Path(__file__).parent / 'fixtures' / 'sam_synthetic_video.txt').read_text()
        tracks = parse_sam_video(raw, 'red square')
        self.assertEqual(len(tracks), 1)
        self.assertEqual(len(tracks[0]['frames']), 10)
