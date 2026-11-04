"""Combine independent descriptions with real evidence and deterministic rules."""
import concurrent.futures
from collections.abc import Callable
from dataclasses import dataclass

from app.pipeline.evidence import extract_evidence
from app.pipeline.orders import create_work_order
from app.pipeline.risk import assign_risk
from app.pipeline.verify import merge_witnesses, parse_witness


class DecisionError(RuntimeError):
    pass


@dataclass(frozen=True)
class Witness:
    provider: str
    model: str
    describe: Callable[[bytes, str], dict]
    label: str | None = None


def _valid_pair(a, b):
    return a is not None and b is not None


def _soft_decision(a, b, merged, score, fallback, det_review=0.5):
    """Rules R8/R9: soft agreement without strict witness agreement.

    R8: one definite witness plus one 'unsure' (never a contradiction) can
    raise 'medium'. R9: if one witness reports holds_water=yes AND
    water_visible=yes AND covered=no and the other witness does not
    contradict (no 'no' on holds_water/water_visible, no yes-vs-no on
    covered), the result is HIGH -- one clear view of water proves water.
    A direct contradiction (yes vs no on holds_water, water_visible or
    covered), a weak detection, an inverted orientation or malformed witness
    data stays 'review'. Strict agreement is still required for 'low'.
    The result is deterministic and auditable; models only describe.
    """
    try:
        if not isinstance(score, (int, float)) or isinstance(score, bool) or score < det_review:
            return fallback
        for k in ('holds_water', 'covered', 'water_visible'):
            if a[k] != b[k] and 'unsure' not in (a[k], b[k]):
                return fallback  # contradiction (yes vs no)
        if a.get('orientation') == 'inverted' or b.get('orientation') == 'inverted':
            return fallback
        # R9 soft-high: one witness sees standing water in an open container.
        for w in (a, b):
            if w.get('holds_water') == 'yes' and w.get('water_visible') == 'yes' and w.get('covered') == 'no':
                return 'high', 'R9'
        soft = {k: (a[k] if a[k] != 'unsure' else b[k]) for k in ('holds_water', 'covered')}
        if soft['holds_water'] == 'yes' and soft['covered'] == 'no':
            return 'medium', 'R8'
    except (KeyError, TypeError):
        pass
    return fallback


def _aggregate_field_yes_wins(values):
    """Any 'yes' wins (one clear view proves presence); else 'no' if any
    conclusive 'no' remains; else 'unsure'. Never fabricates: unsure stays
    unsure when no frame is conclusive."""
    vals = [v for v in values if v in ('yes', 'no', 'unsure')]
    if not vals:
        return 'unsure'
    if 'yes' in vals:
        return 'yes'
    if 'no' in vals:
        # All non-unsure votes say no (there is no yes by now).
        return 'no'
    return 'unsure'


def _aggregate_orientation(values):
    counts: dict[str, int] = {}
    for v in values:
        if v in ('upright', 'inverted', 'on_side'):
            counts[v] = counts.get(v, 0) + 1
    if not counts:
        return 'unsure'
    best_n = max(counts.values())
    winners = [k for k, n in counts.items() if n == best_n]
    if len(winners) == 1:
        return winners[0]
    return 'unsure'


def aggregate_witness_frames(frame_attrs: list[dict]) -> tuple[dict, int | None]:
    """Aggregate one witness's per-frame answers deterministically.

    - holds_water / water_visible: any 'yes' wins (occlusion rule).
    - covered / exposed_to_rain: 'yes' if any frame says yes, else 'no'
      only if all non-unsure votes say no, else 'unsure'.
    - orientation: majority of non-unsure, ties -> unsure.
    - object_label/evidence: from the first supporting 'yes' frame if any,
      else from the best (first) frame.
    Returns (aggregated_attrs, supporting_crop_index|None) where the index
    is the position in frame_attrs (0 == best crop) that first reported
    holds_water=yes or water_visible=yes.
    """
    if not frame_attrs:
        raise ValueError('no frames')
    support = None
    for i, fa in enumerate(frame_attrs):
        try:
            if fa.get('holds_water') == 'yes' or fa.get('water_visible') == 'yes':
                support = i
                break
        except Exception:
            continue
    holds = _aggregate_field_yes_wins([f.get('holds_water') for f in frame_attrs])
    water = _aggregate_field_yes_wins([f.get('water_visible') for f in frame_attrs])
    covered = _aggregate_field_yes_wins([f.get('covered') for f in frame_attrs])
    rain = _aggregate_field_yes_wins([f.get('exposed_to_rain') for f in frame_attrs])
    orient = _aggregate_orientation([f.get('orientation') for f in frame_attrs])
    src = frame_attrs[support] if support is not None else frame_attrs[0]
    try:
        label = src.get('object_label', '')
        ev = src.get('evidence', '')
    except Exception:
        label, ev = '', ''
    if not isinstance(label, str):
        label = ''
    if not isinstance(ev, str):
        ev = ''
    agg = {'holds_water': holds, 'covered': covered, 'water_visible': water,
           'exposed_to_rain': rain, 'orientation': orient,
           'object_label': label, 'evidence': ev}
    return agg, support


def _describe_crops(witness: Witness, crops: list[bytes]) -> list[dict]:
    """Describe up to 3 crops in parallel; order preserved (0 == best)."""
    crops = list(crops)[:3]
    if len(crops) <= 1:
        return [witness.describe(crops[0], 'image/jpeg')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(crops)) as ex:
        futs = [ex.submit(witness.describe, c, 'image/jpeg') for c in crops]
        return [f.result() for f in futs]


def decide_track(private_video, track: dict, first: Witness, second: Witness, *, extractor=extract_evidence) -> dict:
    """Decide one track. The input video must already be privacy processed.

    Each witness describes up to 3 crops (best + alt_crops) in parallel;
    per-witness frames are aggregated deterministically (any-'yes'-wins for
    water). Extra witness calls go through the witness's metered client, so
    SCAN_MAX_USD admission still applies.

    Identity values describe configuration, not proof the endpoint serves that
    model; deployment must independently verify provider and model identity.
    """
    identities = (first.provider, first.model, second.provider, second.model)
    if any(not isinstance(x, str) or not x.strip() for x in identities):
        raise DecisionError('witness provider and model identities are required')
    if first.provider.strip().casefold() == second.provider.strip().casefold() or first.model.strip().casefold() == second.model.strip().casefold():
        raise DecisionError('witnesses must use distinct providers and models')
    evidence = extractor(private_video, track)
    try:
        alts = tuple(getattr(evidence, 'alt_crops', ()) or ())
    except Exception:
        alts = ()
    crops = [evidence.crop_jpeg, *[c for c in alts if isinstance(c, (bytes, bytearray)) and c]]
    crops = [bytes(c) for c in crops][:3]
    if not crops:
        raise DecisionError('evidence has no crops')
    raw_a_list = _describe_crops(first, crops)
    raw_b_list = _describe_crops(second, crops)
    parsed_a = [parse_witness(r)[0] for r in raw_a_list]
    parsed_b = [parse_witness(r)[0] for r in raw_b_list]
    attrs_a, support_a = aggregate_witness_frames(parsed_a)
    attrs_b, support_b = aggregate_witness_frames(parsed_b)
    merged, agreement = merge_witnesses(attrs_a, attrs_b)
    score = max(frame['score'] for frame in track['frames'])
    risk, rule = assign_risk(merged, score, agreement)
    if not agreement:
        risk, rule = _soft_decision(attrs_a, attrs_b, merged, score, (risk, rule))
    order = create_work_order(track['track_id'], risk, object_label=merged['object_label'])
    fields = ('holds_water', 'covered', 'water_visible', 'orientation', 'exposed_to_rain')
    checks = {'agree': agreement,
              'a': {'label': getattr(first, 'label', None) or first.model, **{k: attrs_a.get(k) for k in fields}},
              'b': {'label': getattr(second, 'label', None) or second.model, **{k: attrs_b.get(k) for k in fields}}}
    return {'track_id': track['track_id'], 'evidence': evidence, 'checks': checks,
            'witness_a': attrs_a, 'witness_b': attrs_b,
            'witness_a_frames': parsed_a, 'witness_b_frames': parsed_b,
            'witness_a_support': support_a, 'witness_b_support': support_b,
            'witness_a_identity': {'provider': first.provider, 'model': first.model},
            'witness_b_identity': {'provider': second.provider, 'model': second.model},
            'witness_agreement': agreement, 'attributes': merged,
            'risk': risk, 'rule_id': rule, 'work_order': order}
