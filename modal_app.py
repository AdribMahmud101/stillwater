"""Modal CPU web configuration for StillWater.

Privacy disclosure: Decoded raw video frames (JPEG) are transmitted to the
hosted Meta SAM detector endpoint for person segmentation before any client-side
blurring occurs. Only sanitized evidence (crops, binary masks, anonymized video)
is persisted in scan results.

Cost discipline: Live scans enforce strict cost caps via in-memory ScanMeter
admissions up to SCAN_MAX_USD per scan and DAILY_BUDGET_USD across the ledger,
preventing unexpected spend overruns.

Persistence & Concurrency: SQLite state is constrained to a single writer
(max_containers=1) on persistent Modal Volume storage ('stillwater-web-data'
mounted at /data). The volume is reloaded on container startup (surviving
scale-to-zero) and committed after each completed scan. The public ASGI endpoint
runs with TRUST_PROXY=1 and container concurrency (@modal.concurrent) to keep
SSE event streams and polling responsive throughout active scans.
"""
import asyncio
import os
from pathlib import Path

import modal

DATA_DIR = Path('/data')
MAX_CONTAINERS = 1
SCALEDOWN_WINDOW = 600
TIMEOUT = 3600
MAX_CONCURRENT_INPUTS = 100
ROOT = Path(__file__).resolve().parent

app = modal.App('stillwater-web')
volume = modal.Volume.from_name('stillwater-web-data', create_if_missing=True)
secret = modal.Secret.from_name('stillwater-secrets')
secrets = [secret]

image = (
    modal.Image.debian_slim(python_version='3.12')
    .apt_install('ffmpeg', 'libgl1', 'libglib2.0-0')
    .pip_install_from_pyproject(str(ROOT / 'pyproject.toml'))
    .add_local_dir(str(ROOT / 'app'), '/root/app')
)


def wrap_runner_with_volume_commit(runner, volume_target=volume):
    """Wrap runner to commit the volume after each completed scan."""
    if runner is None:
        return None

    def committing_runner(source, record, **kwargs):
        res = runner(source, record, **kwargs)
        try:
            volume_target.commit()
        except Exception:  # noqa: BLE001, S110
            pass
        return res

    return committing_runner


_COMMIT_STARTED = []


def _start_periodic_commit(interval=15):
    """Background scans finish after the HTTP response, so persist the volume periodically."""
    if _COMMIT_STARTED:
        return
    import threading
    import time

    def loop():
        while True:
            time.sleep(interval)
            try:
                volume.commit()
            except Exception:  # noqa: BLE001, S110
                pass

    threading.Thread(target=loop, daemon=True, name='volume-commit').start()
    _COMMIT_STARTED.append(True)


def build_web_app(factory=None, runner=None):
    """Build the public FastAPI web application with runner and persistence hooks."""
    os.environ.setdefault('DATA_DIR', str(DATA_DIR))
    os.environ.setdefault('TRUST_PROXY', '1')

    try:
        volume.reload()
    except Exception:  # noqa: BLE001, S110
        pass

    if runner is None:
        from app.pipeline.providers import make_runner_from_env

        runner = make_runner_from_env(os.environ)

    runner = wrap_runner_with_volume_commit(runner, volume)

    if factory is None:
        from app.main import create_app

        factory = create_app

    try:
        web_app = factory(DATA_DIR, runner=runner)
    except TypeError:
        web_app = factory(DATA_DIR)

    _start_periodic_commit()

    if hasattr(web_app, 'middleware'):
        @web_app.middleware('http')
        async def commit_on_scan(request, call_next):
            response = await call_next(request)
            if request.method == 'POST' and request.url.path == '/api/scans':
                try:
                    aio_commit = getattr(volume.commit, 'aio', None)
                    if callable(aio_commit):
                        coro = aio_commit()
                        if asyncio.iscoroutine(coro):
                            await coro
                        else:
                            volume.commit()
                    else:
                        volume.commit()
                except Exception:  # noqa: BLE001, S110
                    pass
            return response

    return web_app


@app.function(
    image=image,
    volumes={DATA_DIR: volume},
    secrets=secrets,
    cpu=4.0,
    memory=4096,
    max_containers=MAX_CONTAINERS,
    scaledown_window=SCALEDOWN_WINDOW,
    timeout=TIMEOUT,
    env={'DATA_DIR': str(DATA_DIR), 'TRUST_PROXY': '1'},
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_INPUTS)
@modal.asgi_app(requires_proxy_auth=False)
def web():
    try:
        volume.reload()
    except Exception:  # noqa: BLE001, S110
        pass
    return build_web_app()
