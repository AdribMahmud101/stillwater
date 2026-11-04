"""Conservative, one-to-one visual candidate matching across two scans.

Input positions are normalized along the same walk path. Histograms are
nonnegative, equally sized color-bin counts. This is a heuristic: matches and
thresholds require evaluation on labelled before/after footage.
"""
import math


def _labels(item):
    return {str(s).strip().lower() for s in [item.get('concept', ''), *item.get('aliases', [])] if s}


def _score(a, b):
    if not _labels(a) & _labels(b):
        return None
    p, q = a.get('position'), b.get('position')
    h, k = a.get('histogram'), b.get('histogram')
    if (isinstance(p, bool) or isinstance(q, bool) or not isinstance(p, (int, float))
            or not isinstance(q, (int, float)) or not math.isfinite(p) or not math.isfinite(q)
            or not (0 <= p <= 1 and 0 <= q <= 1)
            or not isinstance(h, (list, tuple)) or not isinstance(k, (list, tuple))
            or not h or len(h) != len(k)):
        return None
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in (*h, *k)):
        return None
    sh, sk = sum(h), sum(k)
    if sh <= 0 or sk <= 0:
        return None
    appearance = sum(min(x / sh, y / sk) for x, y in zip(h, k))
    return .5 * (1 - abs(p - q)) + .5 * appearance


def compare_scans(original, current, match_min=.75, ambiguity_margin=.05):
    """Return original status rows followed by new current objects.

    Candidates must be each other's unique best match by `ambiguity_margin`.
    Without adequate evidence or a clear match, original objects are not seen;
    unmatched current objects are new. Needs-review risk stays needs review.
    """
    if not 0 <= match_min <= 1 or not 0 <= ambiguity_margin <= 1:
        raise ValueError('invalid matching threshold')
    if len({x['id'] for x in original}) != len(original) or len({x['id'] for x in current}) != len(current):
        raise ValueError('duplicate object id in scan')
    scores = {}
    for i, a in enumerate(original):
        for j, b in enumerate(current):
            score = _score(a, b)
            if score is not None and score >= match_min:
                scores[i, j] = score
    def unique_best(candidates):
        values = sorted(candidates, key=lambda x: (-x[1], x[0]))
        if not values or (len(values) > 1 and values[0][1] - values[1][1] <= ambiguity_margin):
            return None
        return values[0][0]
    matches = {}
    for i in range(len(original)):
        j = unique_best([(j, score) for (ii, j), score in scores.items() if ii == i])
        if j is not None and unique_best([(ii, score) for (ii, jj), score in scores.items() if jj == j]) == i:
            matches[i] = j
    rows = []
    for i, a in enumerate(original):
        j = matches.get(i)
        if j is None:
            rows.append({'original_id': a['id'], 'current_id': None, 'status': 'not_seen', 'match_score': None})
        else:
            b = current[j]
            risk = b.get('risk')
            status = 'fixed' if risk == 'low' else 'still_risky' if risk in ('medium', 'high') else 'needs_review'
            rows.append({'original_id': a['id'], 'current_id': b['id'], 'status': status, 'match_score': scores[i, j]})
    for j, b in enumerate(current):
        if j not in matches.values():
            rows.append({'original_id': None, 'current_id': b['id'], 'status': 'new', 'match_score': None})
    return rows
