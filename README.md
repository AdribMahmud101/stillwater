# StillWater

**Find the still water before the mosquitoes do.**

StillWater turns a phone video or a single photo into a mosquito-breeding inspection: it finds water-holding containers and standing water, decides how risky each one is with fixed, auditable rules, and writes bilingual (Bangla / English) cleanup work orders with photo evidence.

Built for the Meta Global AI Developer Hackathon (Nov 2–4, 2026).

- **Live demo:** https://stillwater.onefusionlab.systems (six one-click demo scans, or upload your own photo or video)
- **Demo video:** _add link_

## How it works

| # | Stage | What happens | Runs on |
|---|---|---|---|
| 1 | Ingest | Video or photo is validated and normalized (ffmpeg, ~10 fps, max 1280 px). | App server (Modal, CPU) |
| 2 | Privacy | Every frame is sent to a hosted person detector; detected people are blurred before anything is saved or shown. | Meta SAM 3.1 (hosted) |
| 3 | Scout | Proposes which kinds of containers to look for (tyres, buckets, pots, drums, puddles, drains). It names things; it does not segment. | Muse Spark Open (via Baseten) |
| 4 | Track | Segments and tracks each proposed concept across frames so every container is counted once. A 480p / 5 fps proxy is used for speed; masks are scaled back and held between proxy frames. | Meta SAM 3.1 (hosted) |
| 5 | Verify | Two independent witnesses describe each candidate (holds water? covered? water visible? exposed to rain?). | Muse Spark Open (Baseten) and Muse Spark 1.3 (Meta API) |
| 6 | Decide | Deterministic rules assign risk. **Models describe; code decides.** | Plain Python (`app/pipeline/risk.py`, `decide.py`) |
| 7 | Orders | Bilingual cleanup work orders with evidence crops, plus an annotated video. | App server |

Disagreement or "unsure" between the two witnesses never produces a confident answer: it is routed to **needs a check** for a person to look at.

### Decision rules

| Rule | Risk | Meaning |
|---|---|---|
| R0 | review | Detector not confident this is a container |
| R1 / R1b | review | The two checks disagree or are unsure / water seen but container reported as not holding water |
| R2 | low | Both checks agree it looks empty or upside down |
| R3 | high | Both agree an uncovered container holds visible water |
| R4 | high | Both agree an uncovered container holds water and is exposed to rain |
| R5 | medium | Both agree an uncovered container can hold water, none visible yet |
| R6 | low | Both agree the container is covered |
| R7 | review | Incomplete answers |
| R8 | medium | One check says an uncovered container can hold water, the other is unsure |
| R9 | high | One check clearly saw standing water in an uncovered container, the other did not disagree |

### Other features

- Live progress for every scan (stage, frames or containers done, time estimate) over Server-Sent Events.
- A "Two checks" panel on each container showing both models' answers side by side.
- Rescan comparison: Fixed / Still risky / New / Not seen (never treats "not seen" as fixed).
- Scan map, printable bilingual work orders, light and dark themes.
- Cost guards: per-scan cap, daily cap, request rate limits, fail-closed privacy, security headers (CSP, frame and content-type protection).

## Honest limits

- **No ground-truth evaluation.** The numbers in the app's "How it works" page come from our own visual check of 10 team-recorded clips, not human-labelled data. They are not an accuracy benchmark. Known failure: on one empty-roof clip, red-stained pots were wrongly flagged high.
- **Person privacy recall is not guaranteed.** Unblurred frames are sent to the hosted detector, and recall was not measured. Always check before sharing results.
- **Scan time is minutes for video.** A 16 s rooftop clip took about 5 minutes (hosted segmentation latency varies); a photo takes about 20 s.
- **Costs are estimates** computed from measured token and frame counts times list prices: roughly $0.36 for a 16 s video and $0.01–0.06 for a photo, not provider billing records.
- Results are an AI estimate. A person should confirm on site.

## Run locally

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/), FFmpeg (`ffmpeg`, `ffprobe`).

```sh
uv sync
cp .env.example .env     # add your own API keys (see below)
uv run python -m unittest discover -s tests   # offline tests, no network
uv run uvicorn app.main:app --reload          # http://127.0.0.1:8000
```

Configuration (environment variables, see `.env.example`):

| Variable | Purpose |
|---|---|
| `MODEL_API_KEY` | Meta Model API key (SAM 3.1, Muse Spark 1.3) |
| `BASETEN_API_KEY` | Baseten key (Muse Spark Open) |
| `DAILY_BUDGET_USD`, `SCAN_MAX_USD` | Daily and per-scan spending caps |
| `WITNESS_B` | `meta` (default, Muse Spark 1.3) or `glimmer` (Muse Glimmer 30B on Modal) |
| `TRACK_PROXY`, `TRACK_PROXY_FPS`, `TRACK_PROXY_SHORT_SIDE` | Tracking speed-up proxy settings |
| `PRIVACY_WORKERS`, `TRACK_WORKERS`, `VERIFY_WORKERS` | Parallelism per stage |
| `MAX_CONCURRENT_SCANS`, `RATE_LIMIT_SCANS_PER_HOUR`, `RATE_LIMIT_API_PER_MIN` | Abuse limits (`<= 0` disables a limit) |

Never commit `.env`.

## Deploy

The app runs as a public ASGI app on [Modal](https://modal.com) (`modal_app.py`, secrets in a Modal secret named `stillwater-secrets`, data on a Modal volume). An optional GPU service for Muse Glimmer 30B is in `gpu_services.py`. A small Cloudflare Worker (`deploy/cloudflare/`) proxies a custom domain to the Modal app.

```sh
modal deploy modal_app.py
```

## Repository layout

```
app/            FastAPI app, pipeline stages, clients, web UI (app/web/index.html)
app/pipeline/   ingest, privacy, scout, track, verify, decide, orders, render
app/clients/    Meta, Baseten and Glimmer clients, cost ledger, cache
app/web/        single-file UI and demo scan data (samples/)
tests/          offline unit tests (fakes only)
scripts/        evaluation and spike helpers
deploy/         Cloudflare Worker proxy
modal_app.py    Modal deployment
gpu_services.py optional Glimmer GPU service
```

## Models used

Meta SAM 3.1 (hosted segmentation), Muse Spark Open (open weights, via Baseten), Muse Spark 1.3 (Meta Model API), and Muse Glimmer 30B (optional second witness on a Modal GPU).
