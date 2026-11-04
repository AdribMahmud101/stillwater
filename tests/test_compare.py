import unittest

from app.compare import ComparisonError, compare_verified_scans


def scan(scan_id, track_id, risk='high', position=.5, label='bucket'):
    return {'scan_id': scan_id, 'privacy_verified': True, 'containers': [{
        'track_id': track_id, 'risk': risk, 'concept': label,
        'frame': '000000/frame.jpg', 'crop': '000000/crop.jpg', 'mask': '000000/mask.png',
        'position': position, 'histogram': [1.0, 0.0],
    }]}


class CompareTests(unittest.TestCase):
    def test_matched_and_unseen_are_explicit(self):
        before, after = scan('before', 'a'), scan('after', 'b', 'low')
        result = compare_verified_scans(before, after, evidence_exists=lambda scan_id, path: True)
        self.assertEqual(result['rows'][0]['status'], 'fixed')
        self.assertTrue(result['requires_review'])
        after['containers'][0]['concept'] = 'tire'
        result = compare_verified_scans(before, after, evidence_exists=lambda scan_id, path: True)
        self.assertEqual([row['status'] for row in result['rows']], ['not_seen', 'new'])
        self.assertEqual(result['counts']['fixed'], 0)

    def test_missing_features_evidence_or_privacy_fails_closed(self):
        before, after = scan('before', 'a'), scan('after', 'b')
        before['containers'][0].pop('histogram')
        with self.assertRaises(ComparisonError):
            compare_verified_scans(before, after, evidence_exists=lambda *args: True)
        before = scan('before', 'a')
        with self.assertRaises(ComparisonError):
            compare_verified_scans(before, after, evidence_exists=lambda *args: False)
        before['privacy_verified'] = False
        with self.assertRaises(ComparisonError):
            compare_verified_scans(before, after, evidence_exists=lambda *args: True)

    def test_rejects_duplicate_and_untrusted_evidence_paths(self):
        before, after = scan('before', 'a'), scan('after', 'b')
        before['containers'][0]['crop'] = '../../raw.mp4'
        with self.assertRaises(ComparisonError):
            compare_verified_scans(before, after, evidence_exists=lambda *args: True)
        before = scan('before', 'a')
        before['containers'].append(before['containers'][0].copy())
        with self.assertRaises(ComparisonError):
            compare_verified_scans(before, after, evidence_exists=lambda *args: True)
