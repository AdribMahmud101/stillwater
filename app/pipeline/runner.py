"""Composition of scan stages; intentionally not wired to public endpoints.

Caller supplies private upload and enforces budget and access controls. Returned
video and evidence are privacy-processed but detector recall is not guaranteed.
"""
import shutil
import tempfile
import uuid
from pathlib import Path

from app.events import make_event
from app.pipeline.budget_runner import BudgetAccountingError
from app.pipeline.decide import decide_track
from app.pipeline.evidence import extract_evidence
from app.pipeline.ingest import ingest_video, probe_video
from app.pipeline.render import render_video
from app.pipeline.sam_privacy import sam_privacy
from app.pipeline.sam_track import track_video
from app.pipeline.scout import scout_concepts


class RunnerError(RuntimeError):
    def __init__(self, message, reservation_id=None):
        super().__init__(message)
        self.reservation_id = reservation_id


def run_scan(raw_video, sam_client, witness_a, witness_b, *, scout=None,
             ingest=ingest_video, privacy=sam_privacy, track=track_video,
             probe=probe_video, extractor=extract_evidence, renderer=render_video, temp_root=None,
             scan_id=None, event_callback=None):
    """Execute a private scan using injected providers; source remains caller-owned.

    `scout`, if supplied, receives the private video path and returns scout JSON.
    Invalid scout output falls back to the base vocabulary. Exceptions fail the
    scan. No unblurred file is returned, and all local temporary files are removed.
    """
    stage = 'input'
    stage_started = False
    if event_callback is not None:
        if not callable(event_callback):
            raise ValueError('event_callback must be callable')
        make_event(scan_id, 'ingest', 'start')  # Validate before work starts.

    def emit(kind):
        if event_callback is not None:
            event_callback(make_event(scan_id, stage, kind))

    try:
        source = Path(raw_video)
        if not source.is_file() or source.suffix.lower() not in ('.mp4', '.mov', '.webm'):
            raise ValueError('invalid input')
        with tempfile.TemporaryDirectory(prefix='stillwater-', dir=temp_root) as tmp:
            root = Path(tmp)
            raw = root / ('source' + source.suffix.lower())
            shutil.copyfile(source, raw)
            stage = 'ingest'
            stage_started = True
            emit('start')
            normalized = root / 'normalized.mp4'
            ingest(raw, normalized)
            emit('done')
            stage_started = False
            stage = 'privacy'
            stage_started = True
            emit('start')
            private = root / 'private.mp4'
            privacy(normalized, private, sam_client)
            if not private.is_file() or not private.stat().st_size or private.samefile(normalized) or private.samefile(raw):
                raise ValueError('privacy output is missing or invalid')
            raw.unlink()
            normalized.unlink()
            emit('done')
            stage_started = False
            stage = 'scout'
            stage_started = True
            emit('start')
            concepts = scout_concepts(scout(private) if scout is not None else None)
            emit('done')
            stage_started = False
            stage = 'track'
            stage_started = True
            emit('start')
            info = probe(private)
            tracks = track(sam_client, private.read_bytes(), concepts, info.width, info.height)
            emit('done')
            stage_started = False
            stage = 'verify'
            stage_started = True
            emit('start')
            records = [decide_track(private, item, witness_a, witness_b, extractor=extractor) for item in tracks]
            emit('done')
            stage_started = False
            stage = 'render'
            stage_started = True
            emit('start')
            if len({item['track_id'] for item in tracks}) != len(tracks):
                raise ValueError('duplicate track identifiers')
            render_tracks = []
            for item, record in zip(tracks, records, strict=True):
                if item['track_id'] != record['track_id']:
                    raise ValueError('track and decision identifiers differ')
                record['concept'] = item['concept']
                render_tracks.append({**item, 'risk': record['risk']})
            annotated = root / 'annotated.mp4'
            rendered = renderer(private, annotated, render_tracks)
            if Path(rendered.path) != annotated or not annotated.is_file() or not annotated.stat().st_size or annotated.is_symlink() or annotated.samefile(private) or not isinstance(rendered.timeline, list):
                raise ValueError('rendered output is missing or invalid')
            emit('done')
            stage_started = False
            return {'private_video': private.read_bytes(), 'annotated_video': annotated.read_bytes(),
                    'timeline': rendered.timeline, 'containers': records,
                    'work_orders': [item['work_order'] for item in records], 'concepts': concepts}
    except Exception as exc:  # noqa: BLE001
        reservation_id = None
        if isinstance(exc, BudgetAccountingError):
            try:
                candidate = exc.reservation_id
                if isinstance(candidate, str) and str(uuid.UUID(candidate)) == candidate:
                    reservation_id = candidate
            except (ValueError, TypeError, AttributeError):
                pass
        if stage_started:
            try:
                emit('error')
            except Exception:  # noqa: BLE001
                pass
        raise RunnerError(f'{stage} stage failed', reservation_id=reservation_id) from None
