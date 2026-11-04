# StillWater — Deployment & Production Guide

This guide describes how to deploy the StillWater containerized web application to **Fly.io** or **Modal** with automatic **scale-to-zero**.

StillWater couples a FastAPI backend with video ingestion, keyframe processing, SAM 3.1 video tracking, and dual-witness verification models. Running with scale-to-zero ensures compute costs remain at $0.00 during idle periods while keeping inspection state preserved in a persistent storage volume.

---

## 1. Secrets & Environment Variables

StillWater relies on several runtime configuration variables. In production, configure sensitive keys and runtime limits in your platform's secret manager rather than baking them into container images or committing them to Git.

> [!CAUTION]
> **Credential Security:**
> Never commit real API keys to version control, Docker images, or unencrypted configuration files. Always supply secrets at runtime via platform secret managers (`fly secrets` or `modal secret`).

| Variable | Type | Default | Required for Live Scanning | Description |
|---|:---:|:---:|:---:|---|
| `MODEL_API_KEY` | String (Secret) | `""` | Yes | API authentication token for the hosted Meta Model API (`sam-3.1` spatio-temporal segmentation). |
| `BASETEN_API_KEY` | String (Secret) | `""` | Yes | API key for Baseten (`meta-models/muse-spark-open-1026` scene scouting and witness verification). |
| `SCANNING_ENABLED` | String (`0` or `1`) | `0` | Yes | Master toggle for live video analysis. Defaults to `0` (fails closed, returning HTTP 503 on video upload while allowing cached/demo inspection). Set to `1` when verified. |
| `DAILY_BUDGET_USD` | Decimal String | `5.00` | Recommended | Daily spend ceiling in USD. The internal accounting ledger verifies available budget before initiating billable model invocations. |
| `RATE_LIMIT_SCANS_PER_HOUR` | Integer String | `20` | Recommended | Maximum number of video scans accepted per client IP per hour. Mitigates denial-of-wallet and bot flooding attacks. |
| `TRUST_PROXY` | String (`0` or `1`) | `1` | Recommended | Set to `1` (or `true`) when running behind a reverse proxy or load balancer (Fly.io edge proxy or Modal ingress) so client IP resolution correctly inspects `X-Forwarded-For`. |
| `DATA_DIR` | Filesystem Path | `/data` | Optional | Path to the persistent volume mount storing SQLite records, scan evidence, and cache artifacts. |

---

## 2. Persistent Volume Architecture (`DATA_DIR`)

StillWater persists scan records, container crops, binary masks, and annotated videos to the filesystem directory pointed to by `DATA_DIR` (defaults to `/data` inside the container).

```
/data/
├── stillwater.sqlite3       # SQLite database: scan index, status, geo-coordinates, timestamps
├── results/                 # Permanent evidence artifacts
│   └── <scan_id>/
│       ├── manifest.json    # Verified scan manifest & container catalog
│       ├── annotated.mp4    # Bounding-box and tracking overlay video
│       └── <track_index>/   # Frame crops and binary mask PNGs
└── private/                 # Staging area for raw video uploads (mode 0700)
    └── <upload_uuid>/       # Ephemeral input footage; swept immediately post-analysis
```

### SQLite Concurrency & Scale-to-Zero Constraints

- **Single-Writer Constraint:** SQLite operates in-process with filesystem-level locking. It cannot safely coordinate concurrent writes across multiple independent container instances sharing a standard networked block storage volume.
- **Scale-to-Zero Sizing:** To avoid database corruption and lock contention, configure your deployment to run **at most one container at a time** (`max_containers = 1` on Modal; single-machine with `min_machines_running = 0` on Fly.io).
- **Scale-to-Zero Safety:** When idle traffic triggers a scale-down event, the running process closes all SQLite database handles before the machine halts. On the next incoming HTTP request, the platform starts a fresh machine, re-attaches the persistent volume, and opens SQLite cleanly.

---

## 3. Deploying to Fly.io

[Fly.io](https://fly.io) runs Docker containers as lightweight virtual machines (Fly Machines) that can automatically stop when idle and wake up on incoming HTTP requests.

### Prerequisites

1. Install the `flyctl` CLI:
   ```sh
   curl -L https://fly.io/install.sh | sh
   ```
2. Authenticate with your Fly account:
   ```sh
   fly auth login
   ```

### Step 1: Create Persistent Volume

Create a 10 GB persistent volume named `stillwater_data` in your target region (e.g., `iad` or `ord`):

```sh
fly volumes create stillwater_data --region iad --size 10
```

### Step 2: Configure `fly.toml`

Create or update `fly.toml` in the project root:

```toml
app = "stillwater-web"
primary_region = "iad"

[build]
  dockerfile = "Dockerfile"

[mounts]
  source = "stillwater_data"
  destination = "/data"
  initial_size = "10gb"

[http_service]
  internal_port = 8000
  force_https = true
  auto_stop_machines = "stop"
  auto_start_machines = true
  min_machines_running = 0

  [http_service.concurrency]
    type = "requests"
    soft_limit = 20
    hard_limit = 30

[[http_service.checks]]
  grace_period = "10s"
  interval = "30s"
  method = "GET"
  timeout = "5s"
  path = "/api/health"

[[vm]]
  cpu_kind = "shared"
  cpus = 1
  memory_mb = 2048
```

> [!NOTE]
> **Scale-to-Zero Configuration:**
> Setting `auto_stop_machines = "stop"`, `auto_start_machines = true`, and `min_machines_running = 0` configures Fly Proxy to halt the machine when traffic subsides and awaken it within milliseconds upon the next request.

### Step 3: Set Secrets

Inject environment variables securely using `fly secrets set` (replace sample values with your actual keys):

```sh
fly secrets set \
  MODEL_API_KEY="your-hosted-sam3-model-api-key" \
  BASETEN_API_KEY="your-baseten-api-key" \
  SCANNING_ENABLED="1" \
  DAILY_BUDGET_USD="5.00" \
  RATE_LIMIT_SCANS_PER_HOUR="20" \
  TRUST_PROXY="1"
```

### Step 4: Deploy Container

Deploy the application using the local `Dockerfile`:

```sh
fly deploy
```

### Step 5: Verify Deployment and Scale-to-Zero

1. Check machine status:
   ```sh
   fly status
   ```
2. Monitor live application logs:
   ```sh
   fly logs
   ```
3. Verify the health check:
   ```sh
   curl https://stillwater-web.fly.dev/api/health
   # Expected output: {"status":"ok","scanning_available":true}
   ```
4. Allow the machine to idle for ~5–10 minutes. Check `fly status` to confirm the machine transitions to the `stopped` state. Sending another HTTP request should restart the machine automatically.

---

## 4. Deploying to Modal

[Modal](https://modal.com) provides serverless container execution with integrated GPU/CPU functions, fast cold starts, and native scale-to-zero.

### Prerequisites

1. Install Modal and authenticate:
   ```sh
   uv run modal setup
   ```

### Step 1: Create Persistent Volume

Create a named Modal Volume to store SQLite data and evidence:

```sh
modal volume create stillwater-web-data
```

### Step 2: Create Secrets

Create a Modal secret group containing your production credentials:

```sh
modal secret create stillwater-secrets \
  MODEL_API_KEY="your-hosted-sam3-model-api-key" \
  BASETEN_API_KEY="your-baseten-api-key" \
  SCANNING_ENABLED="1" \
  DAILY_BUDGET_USD="5.00" \
  RATE_LIMIT_SCANS_PER_HOUR="20" \
  TRUST_PROXY="1"
```

### Step 3: Wire Application in `modal_app.py`

Modal deploys ASGI applications via `modal_app.py`. Ensure your `modal_app.py` attaches the volume, secrets, and scale-to-zero parameters:

```python
from pathlib import Path
import modal

DATA_DIR = Path("/data")
ROOT = Path(__file__).resolve().parent

app = modal.App("stillwater-web")
volume = modal.Volume.from_name("stillwater-web-data", create_if_missing=True)
secrets = [modal.Secret.from_name("stillwater-secrets")]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install_from_pyproject(str(ROOT / "pyproject.toml"))
    .add_local_dir(str(ROOT / "app"), "/root/app")
)

def build_web_app():
    from app.main import create_app
    from app.pipeline.providers import make_runner_from_env
    import os
    runner = make_runner_from_env(os.environ)
    return create_app(data_dir=DATA_DIR, runner=runner)

@app.function(
    image=image,
    volumes={DATA_DIR: volume},
    secrets=secrets,
    max_containers=1,         # Required: single SQLite database writer
    scaledown_window=60,      # Scale down to 0 after 60s of inactivity
    timeout=300,
)
@modal.asgi_app()
def web():
    return build_web_app()
```

### Step 4: Deploy to Modal

Deploy the web application to Modal's cloud:

```sh
modal deploy modal_app.py
```

The CLI will display the public URL (e.g., `https://<workspace>--stillwater-web-web.modal.run`).

---

## 5. Privacy & Cost Caveats

### Privacy Caveats

1. **Bystanders & Residential Property:**
   - Phone footage recorded outdoors often captures faces, bystanders, vehicles, or private domestic premises.
   - StillWater isolates raw incoming video in a restricted directory (`/data/private/` with POSIX permission mode `0700`) during analysis.
   - The raw uploaded video file is deleted immediately after ingest and frame sampling. Only segmented container crops and binary masks are retained in `/data/results/`.
2. **Access Control on Public Instances:**
   - The base StillWater web interface does not include built-in multi-tenant user authentication. Anyone with the URL can inspect existing scans if directory listings are exposed.
   - For sensitive deployments, gate access behind an authenticating proxy (e.g., Cloudflare Access, Fly.io proxy authentication, or Modal proxy auth).
3. **Regulatory Compliance:**
   - Ensure video recording and automated processing comply with local laws governing surveillance, video recording in public spaces, and data retention.

### Cost Caveats

1. **Multi-Model Inference Metering:**
   - Every live scan calls hosted model endpoints: Meta Model API (`sam-3.1`) for spatio-temporal segmentation and Baseten (`muse-spark-open-1026`) for scene scouting and witness verification.
   - High traffic volumes or unconstrained video uploads can quickly deplete API balances.
2. **Enforced Daily Budget (`DAILY_BUDGET_USD`):**
   - The application maintains an in-memory budget ledger. If daily usage exceeds `DAILY_BUDGET_USD`, incoming scans fail closed with HTTP 503 before making downstream API calls.
3. **Rate Limiting (`RATE_LIMIT_SCANS_PER_HOUR`):**
   - Prevents automated bots or abusive clients from exhausting your daily budget in a few minutes.
4. **Cold-Start Latency:**
   - Scale-to-zero saves 100% of compute costs when inactive, but the initial HTTP request after an idle period incurs a cold-start spin-up latency (typically 5 to 15 seconds) while the container boots and loads OpenCV/Python modules.
5. **Video Duration Limits:**
   - Ingest strictly enforces a maximum video duration of 60 seconds and maximum file size of 200 MB to prevent long-running inference jobs from monopolizing resources.

---

## 6. Pre-Launch Checklist

Complete this checklist prior to flipping `SCANNING_ENABLED="1"` on a production instance:

- [ ] **Container Image Verified:** Dockerfile builds cleanly with non-root user `appuser` (UID 1000) and no dev dependencies.
- [ ] **Persistent Volume Mounted:** Persistent volume created and verified mounted at `/data`.
- [ ] **Volume Permissions Checked:** Non-root user `appuser` has read/write permissions to `/data`.
- [ ] **Single Writer Enforced:** Deployment constrained to at most 1 active container (`max_containers=1` or single Fly machine) to preserve SQLite integrity.
- [ ] **Secrets Configured:** `MODEL_API_KEY`, `BASETEN_API_KEY`, `SCANNING_ENABLED`, `DAILY_BUDGET_USD`, `RATE_LIMIT_SCANS_PER_HOUR`, and `TRUST_PROXY` are set in the platform secret store.
- [ ] **Zero Secrets Committed:** Confirmed `.env`, `api credentials.md`, and local caches are excluded by `.dockerignore` and git history.
- [ ] **Fail-Closed Default Verified:** `/api/health` returns `{"status":"ok","scanning_available":false}` when `SCANNING_ENABLED="0"`.
- [ ] **Scale-to-Zero Verified:** Machine shuts down cleanly when idle and resumes automatically on an incoming HTTP request.
- [ ] **Budget Guard Tested:** Confirmed that scans are rejected with an explicit error once `DAILY_BUDGET_USD` is reached.
- [ ] **Rate Limiting Tested:** Confirmed client uploads exceeding `RATE_LIMIT_SCANS_PER_HOUR` are throttled.
- [ ] **Privacy Sweep Tested:** Verified that temporary video files in `/data/private/` are unlinked immediately after scan completion.
- [ ] **Health Endpoint Operational:** Cloud platform HTTP health check configured against `/api/health` and returning HTTP 200.
