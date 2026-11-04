"""Strict allowlisted trace events; raw prompts and provider messages are excluded."""
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from uuid import UUID

STAGES = frozenset(('ingest', 'privacy', 'scout', 'track', 'verify', 'decide', 'act', 'rescan', 'render'))
KINDS = frozenset(('start', 'progress', 'result', 'error', 'done', 'tool_call'))
MODELS = frozenset(('sam-3.1', 'muse-spark-1.3', 'meta-models/muse-spark-open-1026', 'muse-glimmer'))
PROVIDERS = frozenset(('meta', 'baseten', 'modal-gpu', 'local'))


class EventError(ValueError):
    pass


def make_event(scan_id, stage, kind, **fields):
    """Construct a fact-only JSON-safe event, rejecting unknown and raw fields.

    Timestamp is event creation time. Numeric fields are omitted unless supplied;
    callers must pass measured values and ensure model/provider labels are true.
    """
    try:
        if not isinstance(scan_id, str) or str(UUID(scan_id)) != scan_id:
            raise ValueError
    except (ValueError, AttributeError):
        raise EventError('invalid scan identifier') from None
    if not isinstance(stage, str) or not isinstance(kind, str) or stage not in STAGES or kind not in KINDS:
        raise EventError('invalid event category')
    if fields.keys() - {'model', 'provider', 'latency_ms', 'tokens_in', 'tokens_out', 'gpu_seconds', 'cost_usd', 'done', 'total', 'unit', 'fps'}:
        raise EventError('unsupported event field')
    if 'model' in fields and (not isinstance(fields['model'], str) or fields['model'] not in MODELS):
        raise EventError('unsupported model')
    if 'provider' in fields and (not isinstance(fields['provider'], str) or fields['provider'] not in PROVIDERS):
        raise EventError('unsupported provider')
    for key in ('latency_ms', 'tokens_in', 'tokens_out'):
        if key in fields and (type(fields[key]) is not int or fields[key] < 0):
            raise EventError('invalid integer measurement')
    for key, minimum in (('done', 0), ('total', 1)):
        if key in fields and (type(fields[key]) is not int or fields[key] < minimum):
            raise EventError('invalid progress count')
    if 'done' in fields and 'total' in fields and fields['done'] > fields['total']:
        raise EventError('progress exceeds total')
    if 'unit' in fields and (not isinstance(fields['unit'], str) or fields['unit'] not in ('frames', 'concepts', 'containers')):
        raise EventError('invalid progress unit')
    if 'fps' in fields:
        value = fields['fps']
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < float('inf'):
            raise EventError('invalid frame rate')
    if 'gpu_seconds' in fields:
        value = fields['gpu_seconds']
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < float('inf'):
            raise EventError('invalid GPU measurement')
    if 'cost_usd' in fields:
        value = fields['cost_usd']
        if not isinstance(value, (str, Decimal)):
            raise EventError('cost must be a decimal string')
        try:
            cost = Decimal(value)
            if not cost.is_finite() or cost < 0:
                raise ValueError
        except (InvalidOperation, ValueError):
            raise EventError('invalid cost') from None
        fields['cost_usd'] = str(cost)
    return {'scan_id': scan_id, 'stage': stage, 'kind': kind,
            'created_at': datetime.now(timezone.utc).isoformat(), **fields}
