"""Pipeline ordering contract for injected stages, not a production runner.

The privacy stage must return an existing distinct video with every frame
properly obscured. This module cannot verify mask coverage, erase temporary raw
media, ensure provider safety, or validate arbitrary stage implementations.
"""
from pathlib import Path


class PipelineError(RuntimeError):
    pass


def run_pipeline(raw_video, stages):
    """Ingest, produce a privacy-filtered file, then scout, track, verify and act.

    Stage object implements ingest(path), privacy(normalized), scout(private),
    track(private, concepts), verify(private, tracks), act(records). Exceptions
    propagate as stage-labelled PipelineError without including sensitive data.
    """
    stage = 'ingest'
    try:
        raw = Path(raw_video)
        normalized = Path(stages.ingest(raw))
        stage = 'privacy'
        private = Path(stages.privacy(normalized))
        if (not private.is_file() or private.stat().st_size == 0
                or private.samefile(raw) or private.samefile(normalized)):
            raise ValueError('privacy stage did not create distinct media')
        stage = 'scout'
        concepts = stages.scout(private)
        stage = 'track'
        tracks = stages.track(private, concepts)
        stage = 'verify'
        records = stages.verify(private, tracks)
        stage = 'act'
        orders = stages.act(records)
        return {'private_video': private, 'containers': records, 'work_orders': orders}
    except Exception as exc:
        raise PipelineError(f'{stage} stage failed') from exc
