# Two-plane observability implementation — final handoff

**Branch:** `feature/phase5-observability` (from `main` @ `e5c8f4e`)  
**Binding spec:** `.local/obs-needs-impl-plan.md` (O0-O4)  
**Date:** 2026-07-16  
**Independent audits acted on:** phases 0-1
(`.local/obs-needs-impl-plan-phases-00-01-audit.md`, all Critical/High
findings remediated) and phase 2
(`.local/obs-needs-impl-plan-phase-02-audit.md`, all three Critical leaks
closed and verified live; High findings fixed or dispositioned below).

## What shipped, by phase

- **O0** — architecture/dependency guards; 54-route policy manifest;
  LLM-boundary manifest (7 boundaries, AST-verified); pinned Pipecat 1.5.0
  static + EMITTED telemetry snapshots; deterministic golden interaction
  fixtures (15 cases, 3 providers) with content/forbidden canaries;
  speech/research/behavior fixture corpora; canary scanner; telemetry-off
  baseline + current-log leak audit (13 findings, all removed);
  three-candidate backend spike with an executable gate.
- **O1** — owner-only SQLite interaction journal (WAL/FULL, monotonic
  idempotent lifecycle, single-transaction group commit, v2 schema with the
  complete canonical envelope, exact `reconstruct()`, whole-row checksums,
  migration path, retention with ACK exception, bounded shutdown); capture
  service with runtime-fail-open-with-visible-gap / evaluation-fail-closed
  policy; non-realtime capture through `summarizer.complete()` (six callers,
  explicit operations, anthropic request-id/retries via raw response);
  realtime Pipecat frame-boundary capture (documented seam); Phoenix export
  worker (idempotent ACK, backoff, journal-only rollback); fixed-schema
  broad JSON logging with root-handler third-party policy; leak fixes.
- **O2** — pinned OTel (SDK 1.43, instrumentation 0.64b0, stable HTTP
  semconv → route templates); default-deny plane router with strict broad
  span envelopes (attributes scrubbed, events dropped, status stripped);
  53-instrument frozen metric manifest with bounded-attribute enforcement;
  MetricsFrame adapter; per-destination httpx instrumentation; collector
  (contrib 0.156.0) with isolated broad/restricted pipelines + file_storage
  WAL + attribute denylist; grafana/otel-lgtm 0.29.0 profile; six generated
  dashboards; six provisioned alert rules; three runbooks; off/on
  performance gate (steady-state CPU well under 2%, +18 MB RSS, no warm
  TTFA regression).
- **O3** — `/ready` readiness model; destructive-route audits; negotiate
  outcome set; live-ownership metrics; storage-operation instrumentation
  (memory/research); scheduler heartbeat; linked finalizer/batch roots;
  isolated restore drill (measured); evaluation harness (WER/CER
  reviewed-only, deterministic behavior checks with human-review flags,
  Phoenix datasets); structured supervisors + machine-readable verification
  records + coturn metrics; per-surface coverage ledger
  (`observability-o3-coverage.md`) with named triggers for every deferral.
- **O4** — strict client telemetry endpoint (schema v1, finite enums,
  bounded numbers, 16 KiB/20-event caps, same-origin, process-wide bucket,
  no IP retention) with fuzz/property tests; PWA/WebRTC/SW instrumentation
  (bounded 50-event queue, 24 h expiry, stripped getStats sampling, SW
  lifecycle/push diagnostics); dogfood rollout ENABLED on the live
  instance 2026-07-16 (env `dogfood`, broad → collector → LGTM, restricted
  → journal + Phoenix).

## Backend ADR

**Phoenix 18.0.0, local, pinned, loopback, behind the `llm-observability`
profile.** Executable spike gate passes (structural span-kind promotion is
a coded, documented criterion); MLflow fails on irreversible attribute
mutations; Langfuse Cloud measured with the owner's scoped key (exact
round-trip, but remote latency + tier limits; local tie-break per plan).
Journal remains the source of truth. Full evidence:
`docs/evidence/observability-backend-spike.md`.

## Validation commands (all green at handoff)

```bash
.venv/bin/ruff check .
.venv/bin/python -m pytest -m unit -q                      # host
.venv/bin/python -m pytest tests/suites/integration -q     # host subset
docker compose exec -T therapy uv run pytest -m integration -q
make test-e2e
uv lock --check
docker compose --profile observability --profile llm-observability config -q
make obs-canary-scan && make obs-fixture-hash
```

Live verifications on the dogfood stack: broad Prometheus carries route
templates only; Tempo trace names are templates (`GET
/api/sessions/{session_id}` for a concrete-ID request); zero canary hits in
any broad surface; Phoenix ACKs restricted exports against journaled IDs;
all six dashboards + alert rules provisioned.

Final gate run (2026-07-16, post O4.2): ruff clean; `uv lock --check` OK;
compose (all profiles) valid; canary gate pass; host unit 156 passed;
container integration 240 passed (one known pre-existing phase-4 ordering
flake passes on rerun); browser e2e 10 passed / 1 skipped including the new
telemetry queue/schema/flush/push coverage. A phase O3 cross-model audit is
running; its findings land as follow-up commits like the O0-O2 audits did.

## Rollout / rollback state

Rollout steps 1-4 and 6 are live (journal → Phoenix export → broad OTel →
LGTM profile → client endpoint gated by `THERAPY_CLIENT_TELEMETRY`).
Rollback: `THERAPY_OTEL_ENABLED=0`, `THERAPY_INTERACTION_BACKEND=journal`,
restart — never delete journal/WAL/backend volumes before capturing
diagnostics (runbooks). Telemetry rollback touches no product schema.

## Owner gates still open (named, per plan)

1. **O4.3 two-week dogfood review** — started 2026-07-16; owner reviews
   capture completeness, evaluator usefulness, false alerts, series counts,
   storage, CPU/RSS, TTFA, then tunes buckets/thresholds/retention.
2. **STT fixture review** — WER/CER claims stay disallowed until the owner
   approves the seeded reference transcripts.
3. **Alert notification channel** — Grafana rules evaluate; the single
   owner-approved content-free notification channel needs the owner's
   choice of destination.
4. **Promotion triggers** — untouched and unchanged: genuine data, exposure
   beyond loopback/tailnet, remote genuine-content export, and
   production-like deployment each require a new threat-model decision.

## Known limitations

- Realtime `provider_native` evidence is frame-boundary (Pipecat 1.5.0
  exposes no documented wire hook); the non-realtime path captures true
  native bodies. Recorded with the ADR seam note.
- Per-stage TTFA decomposition and several named per-surface instruments
  are deferred with explicit triggers (`observability-o3-coverage.md`).
- `service.instance.id` is random per process by plan §5.4; it does add
  restart-cardinality to runtime metrics (audit H-02 note) — bounded by
  LGTM's 30-day retention and reviewed at the dogfood gate.
- The O2 audit's remaining Medium findings (collector CPU caps, duplicate
  TTFT producers, default histogram buckets) are folded into the O4 dogfood
  tuning pass.

## Unrelated pre-existing worktree changes

None remain: the worktree started clean (post PR #13) and every change on
this branch belongs to the observability plan. The only non-plan edits are
`.vscode/settings.json` (IDE interpreter path, local-only/gitignored) and
`.env` (dogfood rollout block appended; owner-held file, not committed).
`tests/suites/integration/test_phase4_acceptance.py` showed one
pre-existing node/edge ordering flake (passes on rerun), untouched.
