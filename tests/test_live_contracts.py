"""Regression checks for sanitized, real provider responses on synthetic inputs."""
import json
import re
import unittest
from pathlib import Path

from app.pipeline.sam_parse import parse_sam_video
from app.pipeline.scout import validate_scout
from app.pipeline.witness import _validate

FIXTURES = Path(__file__).parent / 'fixtures'


class LiveContractTests(unittest.TestCase):
    def test_baseten_witness_fixture_matches_contract(self):
        response = json.loads((FIXTURES / 'baseten_synthetic_witness.json').read_text())
        choice = response['choices'][0]
        self.assertEqual(choice['finish_reason'], 'stop')
        attrs = _validate(choice['message']['content'])
        self.assertIn(attrs['holds_water'], ('yes', 'no', 'unsure'))

    def test_baseten_scout_fixture_matches_contract(self):
        fixture = json.loads((FIXTURES / 'baseten_synthetic_scout.json').read_text())
        content = fixture['attempts'][-1]['choices'][0]['message']['content']
        parsed = json.loads(content)
        self.assertEqual(parsed, fixture['validated'])
        self.assertIn(validate_scout(parsed)['scene']['setting'], ('rooftop','lane','yard','construction','indoor','other'))

    def test_real_synthetic_sam_video_has_actual_confidences(self):
        raw = (FIXTURES / 'sam_synthetic_video_confidence.txt').read_text()
        tracks = parse_sam_video(raw, 'person')
        frames = [frame for track in tracks for frame in track['frames']]
        self.assertTrue(frames)
        self.assertTrue(all(isinstance(frame['score'], (int, float)) and 0 <= frame['score'] <= 1 for frame in frames))

    def test_fixtures_do_not_include_common_token_formats(self):
        token = re.compile(r'\b(?:hf_[A-Za-z0-9]{20,}|LLM_[A-Za-z0-9_-]{20,}|b10_[A-Za-z0-9.]{20,})')
        for path in FIXTURES.iterdir():
            if path.is_file() and path.suffix in ('.json', '.txt'):
                self.assertIsNone(token.search(path.read_text()), f'Potential secret in {path.name}')
