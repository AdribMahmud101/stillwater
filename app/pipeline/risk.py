"""Auditable, conservative container and water-source risk rules.

Input is merged witness attributes, a detector confidence and agreement flag.
Returns (risk, rule_id); models never directly assign risk.

Water sources (puddle, standing water, drain, gutter, ditch, pond, etc.)
use the same deterministic attributes: a source with visible standing water
(water_visible yes, holds_water yes, not covered) is HIGH; an uncertain or
contradictory source is REVIEW, never low by default.
"""
import math
import re
from collections.abc import Mapping

TRI = frozenset(('yes', 'no', 'unsure'))
ORIENTATIONS = frozenset(('upright', 'inverted', 'on_side', 'unsure'))
RISK_FIELDS = frozenset(('holds_water', 'covered', 'water_visible', 'orientation', 'exposed_to_rain'))
OPTIONAL_FIELDS = frozenset(('object_label', 'evidence'))
# Stable rule ids validated by results layer: R<number> with optional letter suffix.
RULE_RE = re.compile(r'^R\d+[a-z]?$')


def assign_risk(attrs: Mapping, det_score: float, witness_agreement: bool, det_review: float = 0.5) -> tuple[str, str]:
    """Return lowercase risk and the first matching rule; malformed data fails closed."""
    if (isinstance(det_score, bool) or not isinstance(det_score, (float, int))
            or not 0 <= det_score <= 1 or not math.isfinite(det_score)
            or isinstance(det_review, bool) or not isinstance(det_review, (float, int))
            or not 0 <= det_review <= 1 or not math.isfinite(det_review)
            or det_score < det_review):
        return 'review', 'R0'
    if witness_agreement is not True:
        return 'review', 'R1'
    if not isinstance(attrs, Mapping) or not RISK_FIELDS <= attrs.keys() or attrs.keys() - RISK_FIELDS - OPTIONAL_FIELDS:
        return 'review', 'R7'
    if any(not isinstance(attrs[name], str) or attrs[name] not in TRI
           for name in RISK_FIELDS - {'orientation'}):
        return 'review', 'R7'
    if not isinstance(attrs['orientation'], str) or attrs['orientation'] not in ORIENTATIONS:
        return 'review', 'R7'
    holds, covered, water = attrs['holds_water'], attrs['covered'], attrs['water_visible']
    orientation, rain = attrs['orientation'], attrs['exposed_to_rain']
    if water == 'yes' and (holds == 'no' or orientation == 'inverted'):
        return 'review', 'R1b'
    if holds == 'no' or orientation == 'inverted':
        return 'low', 'R2'
    if holds == 'yes' and covered == 'no' and water == 'yes':
        return 'high', 'R3'
    if holds == 'yes' and covered == 'no' and rain == 'yes':
        return 'high', 'R4'
    if holds == 'yes' and covered == 'no':
        return 'medium', 'R5'
    if holds == 'yes' and covered == 'yes':
        return 'low', 'R6'
    return 'review', 'R7'
