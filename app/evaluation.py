"""Evaluate one manually matched clip, without generating or assuming labels.

Each prediction is matched by a human to a truth object or None. A correct
class and object match counts once per truth object; duplicates and class
mismatches remain false positives. Inputs are not modified.
"""
from collections import Counter


class EvaluationError(ValueError):
    pass


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise EvaluationError(f'{name} must be a nonempty string')
    return value.strip()


def evaluate_clip(truth: list[dict], predictions: list[dict]) -> dict:
    """Return per-class precision/recall and counting metrics for one clip.

    Truth rows require object_id and type. Prediction rows require prediction_id,
    type, and truth_id (None for false positive). Class names are case-folded.
    Undefined denominators are reported as None; worst class excludes classes
    with no truth objects and breaks tied recall alphabetically.
    """
    if not isinstance(truth, list) or not isinstance(predictions, list):
        raise EvaluationError('truth and predictions must be lists')
    truth_by_id, pred_ids = {}, set()
    for row in truth:
        if not isinstance(row, dict):
            raise EvaluationError('invalid truth row')
        ident = _text(row.get('object_id'), 'object_id')
        label = _text(row.get('type'), 'type').casefold()
        if ident in truth_by_id:
            raise EvaluationError('duplicate truth ID')
        truth_by_id[ident] = label
    counts, matched, references = Counter(), set(), Counter()
    for row in predictions:
        if not isinstance(row, dict) or 'truth_id' not in row:
            raise EvaluationError('invalid prediction row')
        ident = _text(row.get('prediction_id'), 'prediction_id')
        label = _text(row.get('type'), 'type').casefold()
        if ident in pred_ids:
            raise EvaluationError('duplicate prediction ID')
        pred_ids.add(ident)
        counts[label] += 1
        target = row['truth_id']
        if target is not None:
            target = _text(target, 'truth_id')
            if target not in truth_by_id:
                raise EvaluationError('prediction refers to unknown truth ID')
            references[target] += 1
            if truth_by_id[target] == label:
                matched.add(target)
    true_counts = Counter(truth_by_id.values())
    hit_counts = Counter(truth_by_id[ident] for ident in matched)
    classes = {}
    for label in sorted(counts.keys() | true_counts.keys()):
        predicted, expected, hits = counts[label], true_counts[label], hit_counts[label]
        classes[label] = {'true_positive': hits, 'predicted': predicted, 'truth': expected,
                          'precision': hits / predicted if predicted else None,
                          'recall': hits / expected if expected else None}
    candidates = [label for label, metric in classes.items() if metric['recall'] is not None]
    worst = min(candidates, key=lambda label: (classes[label]['recall'], label)) if candidates else None
    return {'classes': classes, 'worst_class': worst,
            'absolute_count_error': abs(len(predictions) - len(truth)),
            'duplicate_rate': sum(n > 1 for n in references.values()) / len(truth) if truth else None,
            'truth_count': len(truth), 'predicted_count': len(predictions)}


_RISKS = ('high', 'medium', 'low', 'review')


def _risk(value):
    if value is None or value == '':
        return None
    if not isinstance(value, str):
        raise EvaluationError('risk must be a string')
    value = value.strip().casefold().replace('_', ' ')
    if value == 'needs review':
        value = 'review'
    if value not in _RISKS:
        raise EvaluationError('invalid risk level')
    return value


def risk_confusion(truth: list[dict], predictions: list[dict]) -> dict:
    """Build a 4x4 risk matrix from manual matches, reporting exclusions.

    Multiple predictions matched to one truth are reduced to one: prefer a
    matching class, then choose the lexicographically smallest prediction ID.
    All other duplicates are counted separately, not placed in the matrix.
    Unmatched objects and pairs missing risk labels are separately reported.
    """
    evaluate_clip(truth, predictions)  # Reuse strict ID and match validation.
    truths = {_text(row['object_id'], 'object_id'): row for row in truth}
    groups = {}
    unmatched_predictions = 0
    for row in predictions:
        _risk(row.get('risk'))
        if row['truth_id'] is None:
            unmatched_predictions += 1
        else:
            groups.setdefault(_text(row['truth_id'], 'truth_id'), []).append(row)
    for row in truth:
        _risk(row.get('risk_truth'))
    matrix = {actual: {pred: 0 for pred in _RISKS} for actual in _RISKS}
    matched_pairs = missing = duplicates = 0
    for target, group in groups.items():
        truth_row = truths[target]
        truth_type = _text(truth_row['type'], 'type').casefold()
        selected = min(group, key=lambda row: (_text(row['type'], 'type').casefold() != truth_type, _text(row['prediction_id'], 'prediction_id')))
        duplicates += len(group) - 1
        actual, predicted = _risk(truth_row.get('risk_truth')), _risk(selected.get('risk'))
        if actual is None or predicted is None:
            missing += 1
        else:
            matrix[actual][predicted] += 1
            matched_pairs += 1
    return {'matrix': matrix, 'matched_pairs': matched_pairs, 'missing_risk_pairs': missing,
            'unmatched_truth': len(truth) - len(groups), 'unmatched_predictions': unmatched_predictions,
            'duplicate_predictions': duplicates}


def evaluate_dataset(clips: list[dict]) -> dict:
    """Aggregate disjoint, manually labeled clips; IDs are local to each clip."""
    if not isinstance(clips, list):
        raise EvaluationError('clips must be a list')
    seen, class_totals, clip_metrics = set(), {}, []
    confusion = {'matrix': {a: {p: 0 for p in _RISKS} for a in _RISKS}, 'matched_pairs': 0,
                 'missing_risk_pairs': 0, 'unmatched_truth': 0, 'unmatched_predictions': 0,
                 'duplicate_predictions': 0}
    duplicate_numerator = total_truth = 0
    for clip in clips:
        if not isinstance(clip, dict):
            raise EvaluationError('invalid clip')
        clip_id = _text(clip.get('clip_id'), 'clip_id')
        if clip_id in seen:
            raise EvaluationError('duplicate clip ID')
        seen.add(clip_id)
        result = evaluate_clip(clip.get('truth'), clip.get('predictions'))
        matrix = risk_confusion(clip['truth'], clip['predictions'])
        clip_metrics.append({'clip_id': clip_id, **result})
        total_truth += result['truth_count']
        if result['duplicate_rate'] is not None:
            duplicate_numerator += round(result['duplicate_rate'] * result['truth_count'])
        for name, value in result['classes'].items():
            entry = class_totals.setdefault(name, {'true_positive': 0, 'predicted': 0, 'truth': 0})
            for field in entry:
                entry[field] += value[field]
        for actual in _RISKS:
            for predicted in _RISKS:
                confusion['matrix'][actual][predicted] += matrix['matrix'][actual][predicted]
        for key in ('matched_pairs', 'missing_risk_pairs', 'unmatched_truth', 'unmatched_predictions', 'duplicate_predictions'):
            confusion[key] += matrix[key]
    classes = {}
    for name, totals in sorted(class_totals.items()):
        hits, predicted, expected = totals['true_positive'], totals['predicted'], totals['truth']
        classes[name] = {**totals, 'precision': hits / predicted if predicted else None,
                         'recall': hits / expected if expected else None}
    candidates = [name for name, value in classes.items() if value['recall'] is not None]
    worst = min(candidates, key=lambda name: (classes[name]['recall'], name)) if candidates else None
    return {'clip_count': len(clips), 'clips': clip_metrics, 'classes': classes, 'worst_class': worst,
            'mean_absolute_count_error': sum(c['absolute_count_error'] for c in clip_metrics) / len(clips) if clips else None,
            'duplicate_rate': duplicate_numerator / total_truth if total_truth else None,
            'risk_confusion': confusion}
