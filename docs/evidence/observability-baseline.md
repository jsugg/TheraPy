# O0.4 telemetry-off baseline and current-log leak audit

**Status:** leak audit complete; measurement sections filled from
`scripts/observability/baseline.py` runs (raw JSON under `.local/obs-baseline/`,
gitignored — this report quotes only content-free distributions).

Reproduce:

```bash
# live dogfood instance (read-only + research round-trip workloads)
.venv/bin/python scripts/observability/baseline.py --label off

# disposable test-mode instance for LLM/scheduler workloads
docker compose run --rm -d --name therapy-baseline \
  -e THERAPY_TEST_MODE=1 -e THERAPY_DATA_DIR=/tmp/baseline-data \
  -p 8001:8000 therapy
.venv/bin/python scripts/observability/baseline.py --label off-testmode \
  --server http://localhost:8001
docker stop therapy-baseline

# scripted voice TTFA (in-container; do not run while the owner is testing)
docker compose exec therapy uv run --no-dev python \
  scripts/verify_voice_text_loop.py
```

## 1. Current-log leak audit (what O1 must remove from the broad plane)

Statically audited every `logger.*` / `print()` call site in `src/therapy`
on branch `feature/phase5-observability` (see git history for the exact
tree). Findings, each mapped to its O1 fix:

| # | Site | Leak | Class | O1 disposition |
|---|------|------|-------|----------------|
| 1 | `integrations/pipecat/pipeline.py:576` | input-audio RMS logged at INFO per utterance | signal detail on hot path | demote to explicit debug-only, per-utterance, short retention (plan O3.2 VAD bullet) |
| 2 | `integrations/pipecat/pipeline.py:596` | free-form `TTFA:` line | free-form log used as metric | replace with owned TTFA metric + fixed event |
| 3 | `integrations/pipecat/pipeline.py:743` | raw `resume_session_id!r` at WARN | product session ID in broad log | fixed event `voice.resume_unknown` with no ID |
| 4 | `integrations/pipecat/pipeline.py:752` | `session_id` + turn count at INFO | product session ID | fixed lifecycle event, IDs restricted only |
| 5 | `integrations/pipecat/pipeline.py:908` | `reply_language: {code!r}` raw unsupported value | unbounded external value | bounded `unsupported` enum value + counter (plan O1.5 names this site) |
| 6 | `integrations/pipecat/pipeline.py:1052` | `Session {id} closed (...)` | product session ID | fixed event with counts only |
| 7 | `integrations/pipecat/pipeline.py:181-205,1081` | `session_id` in artifact-failure WARN/ERROR | product session ID | fixed event `finalizer.artifact_failed{artifact,failure_type}` |
| 8 | `dialogue/outreach.py:627` | `job=%s` raw job ID at WARN | product job ID | drop ID; keep bounded channel + failure_type (plan O1.5 names this site) |
| 9 | `dialogue/outreach.py:811` | `logger.exception(...)` full traceback + message on scheduler tick | exception payload may embed endpoint/provider detail | classified event + filtered owned-code stack only |
| 10 | `dialogue/policy.py:98` | `Invalid crisis contact configuration: %s` with `exc` | exception payload may embed config content | fixed category event, payload dropped |
| 11 | `memory/__main__.py:128` | `error: {exc}` on CLI stderr | exception payload | fixed error category; human CLI stderr may keep filtered diagnostics (plan O3.3) |
| 12 | Uvicorn access log (default on) | concrete request paths incl. query strings | URL/query in broad stdout | disabled in O1.1; broad FastAPI spans replace it in O2 |
| 13 | Docker `json-file` default log driver | unbounded broad log retention | retention | `local` driver + `max-size`/`max-file` in O1.5 |

Clean findings worth preserving: `runtime.py` error paths already log only
`failure_type=<ExceptionClassName>` (no messages) — this is the pattern the
O1 structured events keep. `runtime.py:75` disables Pipecat's Loguru output;
`policy.py` truncation guards avoid content echo. No provider header, SDP,
or credential logging was found in owned code. Pipecat's own content-bearing
Loguru output is disabled at `_load_pipecat()`; the O0 snapshot
(`tests/fixtures/observability/pipecat/snapshot-1.5.0.json`) records which
span attributes would carry content if tracing were enabled unrouted.

## 2. Workload measurements (telemetry off)

Runs of 2026-07-16 (raw: `.local/obs-baseline/baseline-off-1784204349.json`,
`baseline-off-testmode-1784204458.json`). Host: Intel Mac, Docker Desktop,
live dogfood container `Up 18h`. Distributions are seconds unless noted.

### 2.1 Live instance (read + research round-trip)

| Workload | n | p50 | p95 | max |
|---|---:|---:|---:|---:|
| `GET /health` | 50 | 0.0014 | 0.0019 | 0.021 |
| `GET /api/sessions` | 20 | 0.0265 | 0.0547 | 0.0547 |
| `GET /api/graph` | 20 | 0.0382 | 0.0482 | 0.0482 |
| `GET /api/research` | 20 | 0.0028 | 0.0040 | 0.0040 |
| research ingest (2 txt fixtures) | 2 | 2.23 | 4.38 | 4.38 |
| research query | 20 | 0.0106 | 0.0162 | 0.0162 |
| research delete | 2 | 0.0157 | 0.0164 | 0.0164 |
| data export (108.6 MB) | 5 | 1.36 | 7.78 | 7.78 |

### 2.2 Disposable test-mode instance (agent turn + scheduler)

Launched with `THERAPY_TEST_MODE=1` and a throwaway data dir on port 8001.
The acceptance agent-turn route is deterministic by design (no live LLM), so
these are pipeline-orchestration costs, not provider latency:

| Workload | n | p50 | p95 | failures |
|---|---:|---:|---:|---:|
| `POST /api/testing/agent/turn` | 5 | 0.052 | 0.085 | 0 |
| `POST /api/testing/proactivity/run` (greeting) | 3 | 0.046 | 0.051 | 0 |

### 2.3 Scripted voice (verify_voice_text_loop.py, in-container)

Full run **PASS — all scenarios green** (es/en/pt, code-switch keep/flip,
typed, barge-in, pinned/unpinned). Client-side TTFA per spoken turn
(includes 0.7 s VAD close; provider = host Ollama `smollm3:3b-q4_k_m`):

| Scenario | client TTFA (s) |
|---|---:|
| es (cold: first utterance, model loads) | 48.83 |
| en | 15.28 |
| pt | 11.39 |
| anchor-es | 27.97 |
| code-switch-keep | 9.96 |
| code-switch-flip | 12.28 |
| pinned-pt | 15.55 |
| unpinned-auto | 7.96 |

Warm-turn envelope ≈ **8-16 s**; cold-start ≈ **49 s**. Server-side TTFA
lines (`pipeline.py:596`) did NOT appear on stdout during the run — the root
log level hides owned INFO logs while uvicorn's access log prints; recorded
as leak-audit context and fixed by the O1 logging bootstrap.

### 2.4 Container resources and storage

| Measure | idle (60 s) | under load |
|---|---:|---:|
| CPU % (p50 / p95) | 0.44 / 16.73 | 92.95 / 143.9 |
| RSS (p50) | 4.281 GB | 4.342 GB (max 4.76 GB; 5 GB compose cap) |

Storage at run time: data dir **334,248,194 B**; container stdout log since
start **397,948 B / 5,406 lines** (Docker default `json-file`, unbounded —
O1.5 replaces it with the `local` driver + caps).

## 2.5 O2 telemetry off/on comparison (2026-07-16)

Matched disposable instances (identical image/volumes, fresh data dirs;
`baseline-offd-*` vs `baseline-on-*` under `.local/obs-baseline/`); the on
instance exported to the compose collector -> LGTM and Phoenix.

| Workload (p50 s) | off | on |
|---|---:|---:|
| /health | 0.0013 | 0.0012 |
| /api/sessions | 0.0029 | 0.0036 |
| /api/graph | 0.0089 | 0.0084 |
| research query | 0.0041 | 0.0042 |
| agent turn (deterministic) | 0.0556 | 0.0553 |
| scheduler run | 0.0275 | 0.0382 |
| research ingest (n=2, warm-up variance) | 0.029 | 0.085 |

Idle CPU mean: off 1.40% vs on 0.24% (both noise-level); under-load CPU
mean off 7.0% vs on 4.0% — **steady-state overhead well under the 2%
budget**. Idle RSS: off 55.7 MB vs on 74.1 MB (+18 MB, the OTel SDK).
The ingest p50 delta (2 samples, embedding warm-up) is flagged for O4
re-measurement.

Scripted voice off/on: both runs **PASS all scenarios**. Warm steady-state
turns show no meaningful TTFA regression (off 15.55/7.96/27.97 s vs on
13.04/9.98/15.79 s for pinned/unpinned/anchor); the on run's first turns
were slower (66.6 s cold es) because that instance was freshly started
while the off reference ran on a long-warm container — an instance-warmth
confound, re-measured on one instance across the flag flip in O4 dogfood.

Live routing verification (same session): broad Prometheus series carry
route templates only (`/api/sessions/{session_id}`); zero canary hits in
metric names/labels and Tempo; Phoenix holds the restricted attempt,
ACK-recorded against journaled IDs.

## 3. Remaining measurements (named deferrals)

- **Data restore duration:** destructive against the live owner volume;
  measured in the isolated disposable environment during O3 alert drills.
- **Event-loop lag / TTFA stage decomposition:** requires O1/O2 owned
  instrumentation; the O2 off/on gate reuses `baseline.py` so both sides of
  the comparison share identical workloads.
- **Per-stage TTFA (vad/stt/context/llm/tts):** same as above; only whole-turn
  client TTFA and server-log TTFA are available pre-instrumentation.
