import unittest

from app.pipeline.rescan import compare_scans


def obj(id, concept='bucket', risk='high', position=.5, histogram=(1., 0.), aliases=()):
    return {'id': id, 'concept': concept, 'aliases': list(aliases), 'risk': risk,
            'position': position, 'histogram': histogram}


class RescanTests(unittest.TestCase):
    def test_matched_low_fixed_and_unmatched_never_fixed(self):
        original = [obj('a'), obj('b', 'tire', position=.9)]
        after = [obj('c', risk='low')]
        rows = compare_scans(original, after)
        self.assertEqual([(r['original_id'], r['status']) for r in rows], [('a', 'fixed'), ('b', 'not_seen')])

    def test_still_risky_review_and_new(self):
        orig = [obj('a'), obj('b', 'tire', position=.1)]
        after = [obj('c', risk='medium'), obj('d', 'tire', risk='review', position=.1), obj('e', 'pot')]
        rows = compare_scans(orig, after)
        self.assertEqual([r['status'] for r in rows], ['still_risky', 'needs_review', 'new'])

    def test_ambiguous_pairs_do_not_claim_fixed(self):
        rows = compare_scans([obj('a')], [obj('b', risk='low'), obj('c', risk='low')])
        self.assertEqual(rows[0]['status'], 'not_seen')
        self.assertEqual([r['status'] for r in rows[1:]], ['new', 'new'])

    def test_aliases_match_and_missing_evidence_does_not_match(self):
        rows = compare_scans([obj('a', aliases=('container',))], [obj('b', 'container', risk='low')])
        self.assertEqual(rows[0]['status'], 'fixed')
        missing = obj('a'); missing['histogram'] = None
        rows = compare_scans([missing], [obj('b', risk='low')])
        self.assertEqual(rows[0]['status'], 'not_seen')

    def test_nearby_positions_but_different_appearance_rejected(self):
        rows = compare_scans([obj('a')], [obj('b', risk='low', histogram=(0., 1.))])
        self.assertEqual(rows[0]['status'], 'not_seen')
