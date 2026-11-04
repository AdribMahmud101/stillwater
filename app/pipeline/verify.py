"""Parse and merge independent witness descriptions conservatively."""
import json
from collections.abc import Mapping

TRI_FIELDS = ('holds_water', 'covered', 'water_visible', 'exposed_to_rain')
RISK_FIELDS = (*TRI_FIELDS, 'orientation')
FIELDS = frozenset((*RISK_FIELDS, 'object_label', 'evidence'))
TRI = frozenset(('yes', 'no', 'unsure'))
ORIENTATIONS = frozenset(('upright', 'inverted', 'on_side', 'unsure'))


def _unknown():
    return {**{field: 'unsure' for field in RISK_FIELDS}, 'object_label': '', 'evidence': ''}


def parse_witness(raw):
    """Return (attributes, valid). Invalid whole response becomes all unsure.

    Calling code owns the one retry specified for malformed model output.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return _unknown(), False
    if not isinstance(raw, Mapping) or set(raw) != FIELDS:
        return _unknown(), False
    if any(not isinstance(raw[k], str) or raw[k] not in TRI for k in TRI_FIELDS):
        return _unknown(), False
    if not isinstance(raw['orientation'], str) or raw['orientation'] not in ORIENTATIONS:
        return _unknown(), False
    label, evidence = raw['object_label'], raw['evidence']
    if not isinstance(label, str) or len(label.split()) > 3 or not isinstance(evidence, str) or len(evidence) > 100:
        return _unknown(), False
    return dict(raw), True


def merge_witnesses(first, second):
    """Return (merged_attributes, agreement) with no implied model independence."""
    a, a_valid = parse_witness(first)
    b, b_valid = parse_witness(second)
    agree = a_valid and b_valid and all(a[k] == b[k] and a[k] != 'unsure' for k in ('holds_water', 'covered'))
    result = {k: a[k] if a[k] == b[k] else 'unsure' for k in RISK_FIELDS}
    for field in ('object_label', 'evidence'):
        result[field] = a[field] if a[field] == b[field] else ''
    return result, bool(agree)
