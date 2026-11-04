import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.pipeline.privacy import PrivacyError
from app.pipeline.sam_privacy import _sam_privacy_unverified, sam_privacy


class Client:
    def __init__(self): self.calls = []
    def upload_video(self, data): self.calls.append('upload'); return 'file-1'
    def segment_video_file_with_usage(self, file_id, prompt): self.calls.append(('segment', prompt)); return {'text': 'completed', 'video_frames_processed': 3}
    def delete_file(self, file_id): self.calls.append('delete')


class SamPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / 'raw.mp4'; self.source.write_bytes(b'video')
        self.destination = self.source.parent / 'private.mp4'; self.client = Client()

    @patch('app.pipeline.sam_privacy._video_dimensions', return_value=(4, 3, 3))
    @patch('app.pipeline.sam_privacy.to_dedup_tracks')
    @patch('app.pipeline.sam_privacy.parse_sam_video', return_value=[{}])
    @patch('app.pipeline.sam_privacy.blur_video')
    def test_masks_union_and_missing_frames(self, blur, parse, decode, dims):
        decode.return_value = [{'frames': [{'frame': 1, 'mask': {(1, 2)}, 'score': None}]},
                               {'frames': [{'frame': 1, 'mask': {(2, 2)}, 'score': None}]}]
        _sam_privacy_unverified(self.source, self.destination, self.client)
        self.assertEqual(self.client.calls, ['upload', ('segment', 'person'), 'delete'])
        masks = list(blur.call_args.args[2]); self.assertEqual(len(masks), 3)
        self.assertEqual(int(masks[1].sum()), 2); self.assertEqual(int(masks[0].sum()), 0)

    @patch('app.pipeline.sam_privacy._video_dimensions', return_value=(4, 3, 3))
    @patch('app.pipeline.sam_privacy.parse_sam_video', side_effect=ValueError('bad'))
    @patch('app.pipeline.sam_privacy.blur_video')
    def test_parse_failure_deletes_remote_and_never_blurs(self, blur, parse, dims):
        with self.assertRaises(ValueError): _sam_privacy_unverified(self.source, self.destination, self.client)
        self.assertEqual(self.client.calls[-1], 'delete'); blur.assert_not_called()

    @patch('app.pipeline.sam_privacy._video_dimensions', return_value=(4, 3, 3))
    @patch('app.pipeline.sam_privacy.to_dedup_tracks', return_value=[{'frames': [{'frame': 3, 'mask': set()}]}])
    @patch('app.pipeline.sam_privacy.parse_sam_video', return_value=[])
    @patch('app.pipeline.sam_privacy.blur_video')
    def test_out_of_range_fails_closed(self, blur, parse, decode, dims):
        with self.assertRaises(ValueError): _sam_privacy_unverified(self.source, self.destination, self.client)
        blur.assert_not_called(); self.assertEqual(self.client.calls[-1], 'delete')

    def test_public_adapter_fails_closed_without_coverage_evidence(self):
        with self.assertRaises(PrivacyError):
            sam_privacy(self.source, self.destination, self.client)
        self.assertEqual(self.client.calls, [])
        self.assertFalse(self.destination.exists())
