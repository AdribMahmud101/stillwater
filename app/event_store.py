"""Bounded in-process store for allowlisted events and SSE consumption.

Not durable or shared across processes. Call close and remove when appropriate;
callers own access control. Never pass raw provider output to this store.
"""
from collections import deque
from threading import Condition
import json
from app.events import EventError, make_event


class EventStoreError(ValueError):
    pass


class EventStore:
    def __init__(self, max_events=256, max_scans=128):
        if type(max_events) is not int or type(max_scans) is not int or min(max_events, max_scans) < 1:
            raise ValueError('store bounds must be positive integers')
        self.max_events, self.max_scans = max_events, max_scans
        self._scans = {}
        self._condition = Condition()

    def _get(self, scan_id):
        try:
            make_event(scan_id, 'ingest', 'start')
        except EventError as exc:
            raise EventStoreError('invalid scan') from exc
        if scan_id not in self._scans:
            if len(self._scans) >= self.max_scans:
                raise EventStoreError('scan limit reached')
            self._scans[scan_id] = {'events': deque(maxlen=self.max_events), 'seq': 0, 'closed': False}
        return self._scans[scan_id]

    def register(self, scan_id):
        with self._condition:
            self._get(scan_id)

    def has_scan(self, scan_id):
        with self._condition:
            return scan_id in self._scans

    def publish(self, event):
        if not isinstance(event, dict):
            raise EventStoreError('invalid event')
        try:
            allowed = {'scan_id', 'stage', 'kind', 'created_at', 'model', 'provider', 'latency_ms', 'tokens_in', 'tokens_out', 'gpu_seconds', 'cost_usd', 'done', 'total', 'unit', 'fps'}
            if event.keys() - allowed:
                raise ValueError('unsupported fields')
            fields = {k: v for k, v in event.items() if k not in {'scan_id', 'stage', 'kind', 'created_at'}}
            safe = make_event(event['scan_id'], event['stage'], event['kind'], **fields)
        except (KeyError, ValueError, TypeError) as exc:
            raise EventStoreError('invalid event') from exc
        with self._condition:
            state = self._get(safe['scan_id'])
            if state['closed']:
                raise EventStoreError('scan is closed')
            state['seq'] += 1
            state['events'].append((state['seq'], safe))
            self._condition.notify_all()

    def close(self, scan_id):
        with self._condition:
            self._get(scan_id)['closed'] = True
            self._condition.notify_all()

    def remove(self, scan_id):
        with self._condition:
            state = self._scans.get(scan_id)
            if state is None or not state['closed']:
                raise EventStoreError('only closed scans can be removed')
            del self._scans[scan_id]
            self._condition.notify_all()

    def stream(self, scan_id, *, after=0, heartbeat_seconds=15):
        if type(after) is not int or after < 0 or not isinstance(heartbeat_seconds, (int, float)) or not 0 < heartbeat_seconds <= 60:
            raise EventStoreError('invalid stream cursor or heartbeat')
        cursor = after
        with self._condition:
            self._get(scan_id)
        while True:
            with self._condition:
                state = self._scans.get(scan_id)
                if state is None:
                    return
                if state['events'] and cursor < state['events'][0][0] - 1:
                    cursor = state['events'][0][0] - 1
                    chunk = 'event: gap\ndata: {"reason":"history_expired"}\n\n'
                else:
                    item = next(((seq, event) for seq, event in state['events'] if seq > cursor), None)
                    if item:
                        cursor, event = item
                        chunk = f'id: {cursor}\nevent: stage\ndata: {json.dumps(event, separators=(",", ":"))}\n\n'
                    elif state['closed']:
                        return
                    else:
                        self._condition.wait(timeout=heartbeat_seconds)
                        chunk = ': heartbeat\n\n'
            yield chunk
