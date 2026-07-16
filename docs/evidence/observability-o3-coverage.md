# O3 instrumentation coverage ledger

Per plan §7, every O3 surface is **implemented**, **covered** (by an earlier
phase's artifact), or **deferred** with a named trigger. Date: 2026-07-16.

## O3.1 server routes

| Surface | Disposition |
|---|---|
| lifespan start/ready/degraded/stop + bounded drains | **Implemented** — `app.ready/stopping/stopped` events, bounded scheduler/gateway/journal/OTel shutdown |
| `/health` liveness only, excluded from logs/traces | **Implemented** — FastAPI exclusion list; uvicorn access log off |
| `GET /ready` readiness model | **Implemented** — bounded DB/data-dir/capture/component checks, enums only |
| offer validation/resolution/negotiate stages | **Partially implemented** — negotiate outcome + connection metrics at the gateway boundary; separate per-stage child spans deferred: *trigger = first offer-latency investigation that broad route spans cannot answer* |
| disconnect/ICE/resumable finite outcomes | **Covered** — route spans (templates only) + gateway events; no IDs/credentials in any broad output (canary-verified) |
| session/graph/insight/research/proactivity reads | **Covered** — broad HTTP route spans carry template/method/status/duration; no filters/SQL/content (denylist + collector scrub) |
| turn_audio range/full + bytes | **Deferred** — *trigger: first audio-serving incident or VPS move*; route span covers duration/status today |
| session deletion staged children | **Partially implemented** — audit event + live-session guard exist; distinct DB/audio-stage spans deferred to the same trigger as restore staging |
| destructive/research/data mutation audits | **Implemented** — content-free `owner.audit` events on delete_session/node/edge, remove_boundary, research ingest/correct/reindex/delete, export/restore/delete |
| testing routes: capture required, test-mode dimension | **Implemented** — acceptance routes reject disabled capture; `THERAPY_ENVIRONMENT=test` resource label; startup-alert outside test deployments deferred: *trigger = first non-test deployment of a test-mode image* |
| crisis/static: config validity only, no crisis inference | **Covered** — existing `crisis_config_invalid` fixed event; static excluded from traces; no crisis-activity signal exists anywhere (§9 no-alert list) |

## O3.2 voice/speech/context/providers/finalizers

| Surface | Disposition |
|---|---|
| `_load_pipecat` vendor disable/redaction | **Covered** — preserved (O1.4 regression tests) |
| gateway negotiate outcomes success/conflict/invalid/unavailable | **Implemented** — finite outcome metric + one boundary event, no SDP/peer IDs |
| `_replace_pipeline`/`_stop_pipeline`/disconnect/close lifecycle | **Partially implemented** — existing fixed `failure_type` events (O1.5); per-transition counters deferred: *trigger = first preemption/drain incident* |
| live claim/owns/release | **Implemented** — transitions counter (claim/release/preemption/mismatch) + active-connections gauge |
| app-message accepted/rejected, reply_language | **Implemented** — `voice.reply_language_unsupported` bounded event; typed-turn transcripts already reach restricted capture only when they become LLM input (O1.4) |
| ReplyLanguage/Modality/TurnRelay transitions | **Deferred** — *trigger = first language/modality regression the restricted transcripts cannot explain* |
| prompt/crisis/continuity/rehydration exact restricted | **Covered** — the full rendered system prompt + context enter the canonical record pre-dispatch (O1.4 emitted evidence) |
| VAD/STT signals (RTF, load, redecode, no-speech) | **Partially implemented** — Pipecat MetricsFrame adapter feeds stage/TTFB instruments; named redecode/empty counters deferred: *trigger = STT-quality investigation or Whisper model change* (fixtures + JiWER harness ready) |
| persistence of turns/audio/inbox | **Implemented** — storage_operation instrumentation on add_turn/session lifecycle |
| ContextAssembler stages | **Deferred** — *trigger = first context-latency regression*; exact assembled context is already restricted evidence |
| LLM operations exact + twin | **Covered** — O1.3/O1.4 capture + broad twin metrics |
| Kokoro/TTFA/barge-in | **Covered** — MetricsFrame adapter (TTFA/TTFB/usage/processing) + owned `turn.ttfa` event |
| presence/data-channel/send | **Deferred** — *trigger = first data-channel delivery incident* |
| finalize_session linked root + artifact outcomes | **Implemented** — linked trace root, per-artifact `finalizer.artifact_failed`, terminal `finalizer.failed`, `voice.session_closed` |
| outbound destination wrappers + no-cloud-audio test | **Partially implemented** — per-destination httpx instrumentation helper exists (never global); the architecture test that raw audio bytes cannot reach cloud adapters is deferred: *trigger = any change to provider adapters or audio handling* |

## O3.3 persistence/graph/insight/research/proactivity/sovereignty

| Surface | Disposition |
|---|---|
| MemoryStore db.operation spans/metrics | **Implemented** — create/reopen/end/delete/add_turn/title/recap wrapped (count/duration/busy; no SQL/paths) |
| schema.migrate/backup criticality | **Covered** — journal migration tested; product-schema migration events deferred: *trigger = next product schema change* |
| UserModel suppression/purge/mutation metrics | **Deferred** — *trigger = first graph-integrity investigation*; graph lifecycle tables remain the product audit (plan) |
| distillation attempts/validation/idempotency | **Covered** — exact prompts/completions restricted (DISTILL/JUDGE operations); broad run metrics deferred with the same trigger |
| InsightService state metrics | **Deferred** — *trigger = first insight-queue contention*; never engagement signals (§9) |
| periodic storage inspection | **Partially implemented** — journal WAL/checkpoint/integrity gauges scheduled; data-dir byte gauges deferred: *trigger = first disk-pressure alert* |
| ResearchKB ingest/reindex/query staged | **Implemented (operation level)** — storage_operation wraps ingest_bytes/reindex/query; per-stage (extract/OCR/embed) spans deferred: *trigger = first ingest-latency investigation* |
| OCR/FastEmbed instruments | **Deferred** — same trigger; fixtures + report contracts exist |
| proactivity scheduler/job metrics | **Implemented** — last-tick gauge, non-empty batch roots, delivery-failure events |
| WebPush lifecycle | **Covered** — delivery failure events without endpoint/keys/payload (O1.5) |
| DataSovereignty staged roots/rollback audits | **Partially implemented** — route-level audits + CLI fixed error categories; per-stage spans and rollback-precedence events deferred: *trigger = restore-drill in the disposable environment (O3 gate item)* |

## O3.4 evaluation — implemented via the delegated harness

Speech WER/CER (reviewed-only), deterministic behavioral checks with
human-review flags, Phoenix dataset push, owner TTS listening protocol,
on-demand profiling policy: `scripts/observability/evaluate_*.py`,
`phoenix_datasets.py`, `docs/evidence/observability-evaluation.md`.

## O3.5 TURN/supervisors/host — implemented (delegated, validated)

Structured watchdog/hostwatch JSON with plain fallback, atomic owner-only
state, none|macos notifier, machine-readable verification records, coturn
`--prometheus` internal 9641 (flag verified), no username labels.

## O3 gate notes

- Fault injection: journal crash/busy/disk-full/torn/kill, export outage,
  capture-loss, provider error paths — test suites; finalizer/scheduler
  failure paths covered by unit tests; OCR/embedding fault injection
  deferred with their instrumentation trigger.
- Every LLM boundary is classified (manifest test) and replayable
  (`reconstruct()` + Phoenix export).
- Dashboards/alerts run in the disposable LGTM profile (O2 live check).
- Relay: coturn metrics + `verify_relay_connectivity.py --relay-only`
  machine record distinguish real allocation from a port check.
