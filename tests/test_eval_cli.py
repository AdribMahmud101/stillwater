import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class EvaluationCliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.script = Path(__file__).resolve().parents[1] / 'scripts' / 'eval.py'
        self.truth = self.root / 'truth.csv'
        self.truth.write_text('object_id,type,risk_truth\n1,bucket,high\n2,tire,low\n', encoding='utf-8')
        self.pred = self.root / 'matched.json'
        self.pred.write_text(json.dumps([{'prediction_id':'a','type':'bucket','truth_id':'1','risk':'medium'}]), encoding='utf-8')

    def run_script(self, *args):
        return subprocess.run([sys.executable, str(self.script), *map(str, args)], capture_output=True, text=True, check=False)

    def test_reads_manual_matches_and_writes_report(self):
        out = self.root / 'report.json'
        result = self.run_script('--clip', 'roof', self.truth, self.pred, '--output', out)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(out.read_text())
        self.assertEqual(report['worst_class'], 'tire')
        self.assertEqual(report['risk_confusion']['matrix']['high']['medium'], 1)
        self.assertEqual(report['risk_confusion']['unmatched_truth'], 1)

    def test_missing_truth_match_and_invalid_input_fail_without_report(self):
        self.pred.write_text(json.dumps([{'prediction_id':'a','type':'bucket'}]))
        out = self.root / 'report.json'
        result = self.run_script('--clip', 'roof', self.truth, self.pred, '--output', out)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(out.exists())

    def test_no_inputs_does_not_create_report(self):
        self.assertNotEqual(self.run_script().returncode, 0)
