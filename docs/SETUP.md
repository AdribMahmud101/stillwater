# StillWater — Setup & Local Development Guide

**Status:** Verified Development Prototype (Updated 2026-11-03)  
**Scope:** Environmental container inspection prototype built on Meta's open-weight models.

> [!NOTE]
> **Operational Status (2026-11-03):**
> - The full pipeline runs end to end on real phone video footage and single photos through the local web UI: ingest, per-frame person-privacy blur (every frame sent to hosted Meta SAM 3.1 image segmentation; unblurred frames are sent externally; person recall is not guaranteed and not evaluated), scene scouting (Muse Spark Open on Baseten), SAM 3.1 tracking per concept, two independent witnesses (Muse Spark Open on Baseten and Muse Spark 1.3 on Meta API), deterministic risk rules ([`app/pipeline/risk.py`](file:///home/adrib/meta-developers-ai/app/pipeline/risk.py) R0–R7 + rule R8 in [`app/pipeline/decide.py`](file:///home/adrib/meta-developers-ai/app/pipeline/decide.py)), annotated output, and bilingual (Bangla/English) work orders.
> - All 245 unit and integration tests run offline against local mocks and synthetic fixtures; **no external API keys or GPU resources are required to run the test suite**.
> - Live scanning requires enabling `SCANNING_ENABLED=1` along with provider API credentials. Scans are guarded by `SCAN_MAX_USD` (default $1.00), `DAILY_BUDGET_USD` (default $5.00), a single concurrent scan lock, rate limits, security headers, and fail-closed privacy.
> - **Honesty Disclosures:** Evaluation against human ground-truth labels is NOT done (no accuracy numbers). Person-privacy recall on real video is unverified and not guaranteed. Standing-water risk rules do not extend beyond the generic 'puddle' concept. Public deployment and custom domain are not done (Oracle VM had no capacity). Demo video and final hackathon submission are pending.

---

## 1. System Prerequisites

Before setting up StillWater, ensure the following system tools are installed on your workstation:

1. **Python 3.12+**
   Verify installation:
   ```sh
   python3 --version
   ```

2. **`uv` (Astral Python Package Manager)**
   If not already installed, install `uv` via the official standalone installer:
   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Or via package managers:
   - macOS: `brew install uv`
   - Linux / Windows: see [astral.sh/uv](https://astral.sh/uv)

3. **FFmpeg & `ffprobe`**
   Required for video metadata probing, keyframe extraction, and mask overlay rendering:
   - **Ubuntu/Debian:**
     ```sh
     sudo apt-get update && sudo apt-get install -y ffmpeg
     ```
   - **macOS (Homebrew):**
     ```sh
     brew install ffmpeg
     ```
   - **Verification:**
     ```sh
     ffmpeg -version && ffprobe -version
     ```

---

## 2. Project Installation

Clone the repository and initialize the Python virtual environment and dependencies using `uv`:

```sh
# Clone repository
git clone https://github.com/[ORG_OR_USER]/stillwater.git
cd stillwater

# Install dependencies and sync virtual environment
uv sync
```

This creates a local `.venv/` containing all required packages (`fastapi`, `uvicorn`, `opencv-python-headless`, `modal`, `pytest`, `hypothesis`, `ruff`, etc.).

---

## 3. Environment Configuration

Copy the template `.env.example` to create your local `.env` file:

```sh
cp .env.example .env
```

### Environment Variables Reference

| Variable | Required for Tests? | Required for Live Runner? | Default | Description |
|---|:---:|:---:|:---:|---|
| `SCANNING_ENABLED` | No | Yes | `0` | Set to `1` to enable live scanning in the application; when `0`, uploads fail closed with HTTP 503. |
| `MODEL_API_KEY` | No (mocked) | Yes | None | API key for the hosted Meta Model API. Used for hosted Meta SAM 3.1 segmentation and Meta Muse Spark 1.3 (default Witness B). |
| `BASETEN_API_KEY` | No (mocked) | Yes | None | API key for Baseten. Used for Muse Spark Open scene scouting and Witness A verification. |
| `WITNESS_B` | No | Optional | `meta` | Selects provider for Witness B: `meta` (Muse Spark 1.3 on Meta API) or `glimmer` (Muse Glimmer 30B on Modal). Default is `meta`. |
| `SCAN_MAX_USD` | No | Recommended | `1.00` | Per-scan cost cap in USD. Scans whose estimated reservation exceeds this fail closed before dispatching requests. |
| `DAILY_BUDGET_USD` | No | Recommended | `5.00` | Maximum daily spending limit in USD. Enforced across all calls by the internal budget ledger. |
| `HF_TOKEN` | No (mocked) | Optional | None | Hugging Face User Access Token (used to authenticate access to official Meta checkpoints, e.g., `facebook/sam3.1`). |
| `DATA_DIR` | No | Optional | `./data` | Local path for SQLite database (`stillwater.sqlite3`), scan evidence, and cache storage. |
| `MODAL_TOKEN_ID` | No | Optional | None | Modal Token ID, required only if deploying or running Glimmer on Modal. |
| `MODAL_TOKEN_SECRET` | No | Optional | None | Modal Token Secret, required only if deploying or running Glimmer on Modal. |
| `SAM_PER_FRAME` | No | Optional | `0.001` | Configured per-frame pricing in USD for hosted Meta SAM 3.1 segmentation. |
| `META_SPARK_MODEL` | No | Optional | `muse-spark-1.3` | Model identifier for Meta Muse Spark witness. |
| `GLIMMER_MODEL` | No | Optional | `muse-glimmer` | Model identifier for Modal Glimmer witness. |

> [!NOTE]
> **Cost Accounting Distinction:**
> Recorded scan costs (~0.16 USD for a 7 s video scan, ~0.01–0.03 USD for photo scans) use list-price assumptions for chat tokens (placeholders) and configured per-frame SAM prices ($0.001/frame). They are measured token and frame counts multiplied by configured prices in the internal ledger, **not provider billing records**.

> [!CAUTION]
> **Credential Security:**
> Never commit `.env` or paste actual API secrets into version control or documentation. The `.gitignore` file excludes `.env` by default.

---

## 4. Running the Test Suite

StillWater includes a comprehensive unit test suite covering video and photo ingest, per-frame privacy masking, deterministic risk rules (R0–R8), witness reconciliation, deduplication, rescan delta matching, and cost accounting.

Run all tests via Python's `unittest` runner:

```sh
uv run python -m unittest discover -s tests -v
```

Alternatively, run with `pytest`:

```sh
uv run pytest
```

*(All 245 automated tests pass completely offline using synthetic fixtures and mocks without incurring API costs or making network calls.)*

To check formatting and linting:

```sh
uv run ruff check .
```

---

## 5. Running the Local Application

Start the FastAPI application with Uvicorn:

```sh
uv run uvicorn app.main:app --reload
```

The application will bind to `http://127.0.0.1:8000`.

### Verifying Application Status

1. **Web Dashboard:** Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser to view the StillWater user interface.
2. **API Health Check:**
   ```sh
   curl http://127.0.0.1:8000/api/health
   ```
   **Expected Response (default development mode):**
   ```json
   {"status":"ok","scanning_available":false}
   ```
   `scanning_available` is `false` when `SCANNING_ENABLED=0` (or required keys are unset) to prevent unbudgeted API consumption.

---

## 6. Architecture & Providers

- **Hosted Meta SAM 3.1:** Primary segmentation and tracking provider accessed via Meta Model API. Performs per-frame person-privacy blurring and per-concept object tracking across frames.
  - *Privacy Note:* Every decoded frame is sent externally to hosted Meta SAM 3.1 for masking; unblurred frames are sent externally; person recall is not guaranteed and not evaluated. Fails closed on error.
- **Baseten (Muse Spark Open):** Acts as scene scout and independent Witness A.
- **Meta API (Muse Spark 1.3):** Serves as default live independent Witness B.
- **Modal (Muse Glimmer 30B):** Deployed on Modal with an L4 GPU and scale-to-zero configuration. Verified to return valid JSON attributes, but cold-start latency exceeded 10 minutes (>600 s). It is therefore NOT used as the default live second witness; set `WITNESS_B=glimmer` to switch to it.
- **Self-Hosted SAM on Modal:** Defined as a deferred fallback; not currently active in production failover.
- **Deterministic Decision Logic:** Merges witness outputs; risk is assigned exclusively by rules R0–R7 ([`app/pipeline/risk.py`](file:///home/adrib/meta-developers-ai/app/pipeline/risk.py)) and rule R8 ([`app/pipeline/decide.py`](file:///home/adrib/meta-developers-ai/app/pipeline/decide.py)). Rule R8 maps one definite and one unsure witness to `medium` risk. Contradictions, weak detections, and malformed witness data route to `review`.
