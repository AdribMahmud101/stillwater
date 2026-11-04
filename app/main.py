"""StillWater HTTP entry point; uploads fail closed without a verified runner."""
import json
import logging
import math
import os
import shutil
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.compare import ComparisonError, compare_verified_scans
from app.event_store import EventStore, EventStoreError
from app.hardening import setup_hardening
from app.pipeline.ingest import IngestError, probe_video, probe_image
from app.results import ResultsError, ResultsStore
from app.store import Store
from app.temp_cleanup import managed_temp, cleanup_abandoned, TempCleanupError

VIDEO_SUFFIXES = {'.mp4', '.mov', '.webm'}
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp'}
IMAGE_MAX_BYTES = 15 * 1024 * 1024


def seed_demo_locations(store: Store, path: Path):
    try:
        entries = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(entries, list):
            return
        for entry in entries:
            try:
                store.set_location(entry['scan_id'], entry['area'], entry['lat'], entry['lng'])
            except (KeyError, TypeError, ValueError, sqlite3.Error, OverflowError):
                continue
    except (OSError, ValueError, UnicodeError):
        pass


def create_app(data_dir: Path | None = None, *, runner=None, max_upload_bytes: int = 200 * 1024 * 1024) -> FastAPI:
    if max_upload_bytes < 1:
        raise ValueError('max_upload_bytes must be positive')
    root = Path(data_dir or os.environ.get('DATA_DIR', './data'))
    store = Store(root / 'stillwater.sqlite3')
    store.fail_interrupted()
    seed_demo_locations(store, Path(__file__).parent / "web" / "samples" / "demos.json")
    results = ResultsStore(root / 'results')
    private_root = root / 'private'
    private_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        cleanup_abandoned(private_root, older_than_seconds=86400)
    except TempCleanupError:
        # Refuse to sweep an insecure or unowned directory.
        pass
    event_store = EventStore()
    app = FastAPI(title='StillWater')
    setup_hardening(app)
    samples_dir = Path(__file__).parent / 'web' / 'samples'
    try:
        if samples_dir.is_dir():
            app.mount('/samples', StaticFiles(directory=str(samples_dir), check_dir=False), name='samples')
    except Exception:
        pass

    @app.get('/')
    def index():
        return FileResponse(Path(__file__).parent / 'web' / 'index.html', headers={'Cache-Control': 'no-cache'})

    @app.get('/api/health')
    def health():
        return {'status': 'ok', 'scanning_available': runner is not None}

    summary_cache = {}

    @app.get('/api/scans')
    def list_scans(limit: int = Query(default=50, ge=1, le=100)):
        scans = store.list_scans(limit)
        for scan in scans:
            if scan['status'] != 'complete':
                continue
            ident = scan['id']
            if ident not in summary_cache:
                counts = dict.fromkeys(('high', 'medium', 'review', 'low', 'total'), 0)
                thumb = None
                try:
                    containers = results.load(ident)['containers']
                    if not isinstance(containers, list):
                        raise ValueError('Invalid containers')
                    best = 4
                    for index, container in enumerate(containers):
                        risk = container.get('risk')
                        if risk in ('high', 'medium', 'review', 'low'):
                            counts[risk] += 1
                            counts['total'] += 1
                            rank = ('high', 'medium', 'review', 'low').index(risk)
                            if rank < best:
                                best = rank
                                thumb = f'/api/scans/{ident}/evidence/{index}/crop'
                except (ResultsError, KeyError, TypeError, AttributeError, ValueError):
                    counts = dict.fromkeys(('high', 'medium', 'review', 'low', 'total'), 0)
                    thumb = None
                summary_cache[ident] = (counts, thumb)
            scan['summary'], scan['thumb'] = summary_cache[ident]
        return {'scans': scans}

    @app.get('/api/scans/{scan_id}')
    def scan(scan_id: str):
        result = store.get_scan(scan_id)
        if result is None:
            raise HTTPException(status_code=404, detail='Scan not found')
        return result

    def get_manifest(scan_id: str):
        record = store.get_scan(scan_id)
        if record is None or record['status'] != 'complete':
            raise HTTPException(status_code=404, detail='Scan results not found')
        try:
            return results.load(scan_id)
        except ResultsError:
            raise HTTPException(status_code=404, detail='Scan results not found')

    @app.get('/api/scans/{scan_id}/results')
    def scan_results(scan_id: str):
        return get_manifest(scan_id)

    @app.get('/api/scans/{scan_id}/compare/{current_scan_id}')
    def compare(scan_id: str, current_scan_id: str):
        original, current = get_manifest(scan_id), get_manifest(current_scan_id)
        def evidence_exists(identity, relative):
            scan_root = results.root / identity
            item = scan_root / relative
            if scan_root.is_symlink() or item.parent.is_symlink() or item.is_symlink():
                return False
            return item.is_file() and item.resolve().is_relative_to(scan_root.resolve())
        try:
            return compare_verified_scans(original, current, evidence_exists=evidence_exists)
        except ComparisonError:
            raise HTTPException(status_code=409, detail='Scan comparison needs complete evidence and matching features.')

    def _annotated_response(scan_id: str):
        manifest = get_manifest(scan_id)
        media_type = manifest.get('media_type', 'video')
        name = manifest.get('annotated_video')
        if media_type == 'image':
            if name not in ('annotated.jpg', 'annotated.jpeg', 'annotated.mp4'):
                raise HTTPException(status_code=404, detail='Annotated image not found')
            # Prefer annotated.jpg; fall back to stored name.
            for cand in (name, 'annotated.jpg', 'annotated.jpeg'):
                if isinstance(cand, str):
                    p = results.root / scan_id / Path(cand).name
                    if p.is_file():
                        return FileResponse(str(p), media_type='image/jpeg', headers={'Cache-Control': 'private, no-store', 'X-Content-Type-Options': 'nosniff'})
            raise HTTPException(status_code=404, detail='Annotated image not found')
        if name != 'annotated.mp4':
            raise HTTPException(status_code=404, detail='Annotated video not found')
        path = results.root / scan_id / 'annotated.mp4'
        if not path.is_file():
            raise HTTPException(status_code=404, detail='Annotated video not found')
        return FileResponse(path, media_type='video/mp4', headers={'Cache-Control': 'private, no-store', 'X-Content-Type-Options': 'nosniff'})

    @app.get('/api/scans/{scan_id}/annotated-video')
    def annotated_video(scan_id: str):
        return _annotated_response(scan_id)

    @app.get('/api/scans/{scan_id}/annotated')
    def annotated(scan_id: str):
        return _annotated_response(scan_id)

    @app.get('/api/scans/{scan_id}/evidence/{index}/{kind}')
    def scan_evidence(scan_id: str, index: int, kind: str):
        manifest = get_manifest(scan_id)
        names = {'frame': 'frame.jpg', 'crop': 'crop.jpg', 'mask': 'mask.png'}
        if kind not in names or not 0 <= index < len(manifest['containers']):
            raise HTTPException(status_code=404, detail='Evidence not found')
        path = results.root / scan_id / f'{index:06d}' / names[kind]
        if not path.is_file():
            raise HTTPException(status_code=404, detail='Evidence not found')
        return FileResponse(path, media_type='image/png' if kind == 'mask' else 'image/jpeg', headers={'Cache-Control': 'private, no-store', 'X-Content-Type-Options': 'nosniff'})

    @app.get('/api/scans/{scan_id}/events')
    def scan_events(scan_id: str, after: int = Query(default=0, ge=0)):
        if store.get_scan(scan_id) is None or not event_store.has_scan(scan_id):
            raise HTTPException(status_code=404, detail='Scan events not found')
        return StreamingResponse(event_store.stream(scan_id, after=after), media_type='text/event-stream',
                                 headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})

    def process_scan(source, record, *, background=False):
        scan_id = record['id']
        try:
            if background:
                result = runner(source, record, event_callback=event_store.publish)
            else:
                result = runner(source, record)
            if not isinstance(result, dict) or result.get('privacy_verified') is not True or not isinstance(result.get('containers'), list):
                raise ValueError('Runner did not verify privacy or return results.')
            records = []
            for item in result['containers']:
                if not isinstance(item, dict):
                    raise TypeError('Invalid container result.')
                evidence = item['evidence']
                if 'track_id' in item and item['track_id'] != evidence.track_id:
                    raise ValueError('Container and evidence do not match.')
                projected = {'evidence': evidence, 'risk': item['risk'], 'rule_id': item['rule_id']}
                concept = item.get('concept') if isinstance(item.get('concept'), str) and item.get('concept').strip() else None
                if concept is None and isinstance(evidence.track_id, str) and ':' in evidence.track_id:
                    concept = evidence.track_id.rsplit(':', 1)[0].strip() or None  # track ids are '<concept>:<n>'
                if concept:
                    projected['concept'] = concept
                if 'checks' in item:
                    projected['checks'] = item['checks']
                if 'work_order' in item:
                    projected['work_order'] = item['work_order']
                records.append(projected)
            media_type = result.get('media_type', 'video')
            if media_type not in ('image', 'video'):
                media_type = 'video'
            try:
                runner_duration = result.get('duration_seconds', result.get('duration'))
                if runner_duration is not None:
                    store.set_media(scan_id, media_type, float(runner_duration))
                else:
                    store.set_media(scan_id, media_type, None)
                    # Preserve any duration probed at upload time when runner omits it.
                    if store.get_scan(scan_id).get('media_type') is None:
                        store.set_media(scan_id, media_type, None)
            except Exception:
                try:
                    store.set_media(scan_id, media_type, None)
                except Exception:
                    pass
            manifest = results.save(scan_id, records, privacy_verified=True, annotated_video=result.get('annotated_video'), timeline=result.get('timeline'), cost_usd=result.get('cost_usd'), timings=result.get('timings'), media_type=media_type)
            store.set_status(scan_id, 'complete')
            return manifest
        except Exception as exc:
            logging.getLogger('stillwater').error('scan %s failed: %s: %s', scan_id, type(exc).__name__, str(exc)[:200])
            store.set_status(scan_id, 'failed')
            raise
        finally:
            event_store.close(scan_id)
            if background:
                shutil.rmtree(source.parent, ignore_errors=True)

    def background_scan(source, record, manager):
        try:
            process_scan(source, record, background=True)
        except Exception:
            pass
        finally:
            manager.__exit__(None, None, None)

    @contextmanager
    def upload_temp(root):
        with managed_temp(root) as directory:
            try:
                yield directory
            finally:
                shutil.rmtree(directory, ignore_errors=True)

    @app.post('/api/scans')
    async def upload(request: Request, background_tasks: BackgroundTasks, area: str, filename: str, lat: float | None = None, lng: float | None = None, background: bool = False):
        if runner is None:
            raise HTTPException(status_code=503, detail='Scan processing is not available yet.')
        suffix = Path(filename).suffix.lower()
        is_image = suffix in IMAGE_SUFFIXES
        is_video = suffix in VIDEO_SUFFIXES
        if not is_image and not is_video:
            raise HTTPException(status_code=400, detail='Use an MP4, MOV, WebM video or JPG, JPEG, PNG, WebP image.')
        upload_media_type = 'image' if is_image else 'video'
        limit = IMAGE_MAX_BYTES if is_image else max_upload_bytes
        limit_msg = 'Image exceeds the upload limit.' if is_image else 'Video exceeds the upload limit.'
        if not area.strip() or len(area) > 200 or (lat is None) != (lng is None) or (lat is not None and (not math.isfinite(lat) or not math.isfinite(lng) or not -90 <= lat <= 90 or not -180 <= lng <= 180)):
            raise HTTPException(status_code=400, detail='Enter a valid area and coordinates.')
        content_length = request.headers.get('content-length')
        if content_length:
            try:
                length = int(content_length)
            except ValueError:
                raise HTTPException(status_code=400, detail='Invalid content length.')
            if length < 0 or length > limit:
                raise HTTPException(status_code=413, detail=limit_msg)
        private = root / 'private'
        private.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(private, 0o700)
        with upload_temp(private) as temp:
            source = Path(temp) / ('upload' + suffix)
            fd = os.open(source, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            size = 0
            with os.fdopen(fd, 'wb') as output:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(status_code=413, detail=limit_msg)
                    output.write(chunk)
            if not size:
                raise HTTPException(status_code=400, detail='Video is empty.' if is_video else 'Image is empty.')
            probed_duration = None
            try:
                if is_image:
                    info = await run_in_threadpool(probe_image, source)
                    _ = info
                else:
                    info = await run_in_threadpool(probe_video, source)
                    if info is not None and info.duration is not None and info.duration > 60:
                        raise HTTPException(status_code=400, detail='Video must be at most 60 seconds.')
                    # Best effort: record probed duration, never fail the upload for it.
                    try:
                        if info is not None and info.duration is not None:
                            d = float(info.duration)
                            if math.isfinite(d) and d >= 0:
                                probed_duration = d
                    except Exception:
                        probed_duration = None
            except IngestError:
                raise HTTPException(status_code=400, detail='Image could not be read.' if is_image else 'Video could not be read.')
            record = store.create_scan(area, lat, lng, media_type=upload_media_type, duration_seconds=probed_duration)
            scan_id = record['id']
            try:
                event_store.register(scan_id)
            except EventStoreError:
                store.set_status(scan_id, 'failed')
                raise HTTPException(status_code=503, detail='Scan event capacity is full.')
            store.set_status(scan_id, 'running')
            if background:
                manager = managed_temp(private)
                background_dir = None
                try:
                    background_dir = manager.__enter__()
                    background_source = background_dir / ('upload' + suffix)
                    fd = os.open(background_source, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                    with source.open('rb') as src, os.fdopen(fd, 'wb') as dst:
                        shutil.copyfileobj(src, dst)
                    background_tasks.add_task(background_scan, background_source, record, manager)
                except Exception:
                    if background_dir is not None:
                        shutil.rmtree(background_dir, ignore_errors=True)
                        manager.__exit__(None, None, None)
                    store.set_status(scan_id, 'failed')
                    event_store.close(scan_id)
                    raise HTTPException(status_code=503, detail='Scan could not be started safely.')
                return JSONResponse({'scan_id': scan_id}, status_code=202)
            try:
                manifest = await run_in_threadpool(process_scan, source, record)
            except Exception:
                raise HTTPException(status_code=503, detail='Scan could not be completed safely.')
            return {'scan': store.get_scan(scan_id), 'result': manifest}

    return app


def _default_app():
    import os
    from app.pipeline.providers import make_runner_from_env
    return create_app(runner=make_runner_from_env(os.environ))


app = _default_app()
