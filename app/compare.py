"""Validate persisted evidence and compare two inspection snapshots.

This matching heuristic has not been evaluated on labeled paired footage.
All statuses, including 'fixed', are estimates and require on-site review.
"""
import math

from app.pipeline.rescan import compare_scans


class ComparisonError(ValueError):
    pass


def _prepare(manifest, evidence_exists):
    if not isinstance(manifest, dict) or manifest.get('privacy_verified') is not True:
        raise ComparisonError('Scan privacy has not been verified.')
    scan_id, containers = manifest.get('scan_id'), manifest.get('containers')
    if not isinstance(scan_id, str) or not scan_id or not isinstance(containers, list):
        raise ComparisonError('Invalid scan metadata.')
    result, ids, histogram_length = [], set(), None
    for index, item in enumerate(containers):
        if not isinstance(item, dict):
            raise ComparisonError('Invalid container.')
        ident, concept, risk = item.get('track_id'), item.get('concept'), item.get('risk')
        position, histogram = item.get('position'), item.get('histogram')
        if not isinstance(ident, str) or not ident or ident in ids or not isinstance(concept, str) or not concept.strip() or risk not in {'high', 'medium', 'low', 'review'}:
            raise ComparisonError('Container identity, concept, or risk is invalid.')
        ids.add(ident)
        if isinstance(position, bool) or not isinstance(position, (int, float)) or not math.isfinite(position) or not 0 <= position <= 1:
            raise ComparisonError('Comparison position is unavailable or invalid.')
        if not isinstance(histogram, (list, tuple)) or not histogram or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in histogram) or sum(histogram) <= 0:
            raise ComparisonError('Comparison histogram is unavailable or invalid.')
        if histogram_length is None:
            histogram_length = len(histogram)
        if len(histogram) != histogram_length:
            raise ComparisonError('Comparison histograms do not match.')
        for key, name in (('frame', 'frame.jpg'), ('crop', 'crop.jpg'), ('mask', 'mask.png')):
            expected = f'{index:06d}/{name}'
            if item.get(key) != expected:
                raise ComparisonError('Evidence path is invalid.')
            try:
                exists = evidence_exists(scan_id, expected)
            except Exception as exc:
                raise ComparisonError('Evidence is unavailable.') from exc
            if exists is not True:
                raise ComparisonError('Evidence is unavailable.')
        result.append({'id': ident, 'concept': concept, 'risk': risk, 'position': position, 'histogram': list(histogram)})
    return scan_id, result, histogram_length


def compare_verified_scans(original, current, *, evidence_exists):
    """Compare validated, evidence-backed snapshots with cautious statuses.

    Caller must load completed scans from trusted storage and supply a function
    checking private evidence existence; this function cannot prove privacy or
    identity from metadata. Features must come from real privacy-filtered crops.
    """
    if not callable(evidence_exists):
        raise ComparisonError('Evidence verification is required.')
    old_id, old, old_bins = _prepare(original, evidence_exists)
    new_id, new, new_bins = _prepare(current, evidence_exists)
    if old_id == new_id:
        raise ComparisonError('Compare two different scans.')
    if old_bins is not None and new_bins is not None and old_bins != new_bins:
        raise ComparisonError('Comparison histograms do not match.')
    rows = compare_scans(old, new)
    statuses = ('fixed', 'still_risky', 'needs_review', 'not_seen', 'new')
    return {'original_scan_id': old_id, 'current_scan_id': new_id, 'rows': rows,
            'counts': {status: sum(row['status'] == status for row in rows) for status in statuses},
            'requires_review': True,
            'notice': 'Estimated matches require on-site review. Not seen does not mean fixed.'}
