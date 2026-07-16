# Observability backend acceptance spike (O0.3) — decision record

**Measured:** 2026-07-15/16 on macOS 15.7.7, Intel x86_64, Python 3.12.13  
**Scope:** private, single-user TheraPy; synthetic fixtures only; loopback services  
**Raw evidence:** `.local/obs-spike/results/{phoenix,mlflow}.json` and
`.local/obs-spike/logs/`; runbook `scripts/observability/backend_spike.md`  
**Fixture corpus SHA-256 (final run):**
`8198524ecf0027f87415ed95c0e39dff2828a09363c66db86adcda9e44c06479`

A first run against corpus `683b2279…` exposed two golden cases sharing
`(trace_id, span_id)` transport IDs — a fixture-generator defect, since each
case models a distinct provider attempt. The generator was fixed, the corpus
regenerated, and both local candidates re-measured end to end against the
corrected corpus. Numbers below are from the corrected run; the colliding-ID
run is retained in git history as duplicate-semantics evidence.

**Decision (ADR):** see "Selected backend" at the end of this document.

## Method and interpretation guardrails

- All 15 committed records were encoded as OpenInference `LLM` OTel spans and
  sent as one OTLP/HTTP protobuf batch. The original canonical JSON was also a
  scalar span attribute (`therapy.canonical_record`). The query was repeated 20
  times; p50/p95 use nearest-rank percentiles.
- The compact canonical record payload is 33,090 bytes. Disk amplification
  includes each product's migrated SQLite schema and other files under its
  storage directory, so it is deliberately an end-to-end small-corpus number.
- The corrected corpus contains **15 unique `(trace_id, span_id)` pairs** —
  one per golden case. The earlier collision (two derived cases inheriting
  their parents' IDs) was fixed in
  `scripts/observability/gen_interaction_fixtures.py` (`_rekey()`).
- “Canonical field losses” counts missing leaf paths in a parsed canonical
  envelope. “Attribute discrepancies” separately counts exported attribute-map
  removal or irreversible representation changes. A JSON string decoded into
  the same object is listed as a reversible transformation, not a field loss.
- Both outage probes terminated the server tree immediately before a one-span
  export. This proves a deterministic post-kill failure surface, not an
  in-flight TCP interruption.

---

## Candidate 1 — Phoenix 18.0.0

### A. Officially documented/versioned facts

- Phoenix 18.0.0 was released on 2026-07-14 and runs locally with
  `phoenix serve`; HTTP OTLP is `/v1/traces`. Its default local database is
  SQLite under `PHOENIX_WORKING_DIR`, unless `PHOENIX_SQL_DATABASE_URL`
  overrides it. ([18.0.0 package](https://pypi.org/project/arize-phoenix/18.0.0/),
  [terminal deployment](https://github.com/Arize-ai/phoenix/blob/arize-phoenix-v18.0.0/docs/phoenix/self-hosting/deployment-options/terminal.mdx),
  [v18 server source](https://github.com/Arize-ai/phoenix/blob/arize-phoenix-v18.0.0/src/phoenix/server/cli/commands/serve.py#L287-L300),
  [v18 database config](https://github.com/Arize-ai/phoenix/blob/arize-phoenix-v18.0.0/src/phoenix/config.py#L1076-L1084))
- `PHOENIX_TELEMETRY_ENABLED=false` disables Phoenix UI analytics pixels.
  `PHOENIX_ALLOW_EXTERNAL_RESOURCES=false` also prevents external UI resources.
  ([v18 privacy documentation](https://github.com/Arize-ai/phoenix/blob/arize-phoenix-v18.0.0/docs/phoenix/self-hosting/security/privacy.mdx#L21-L42))
- The v18 gRPC collector binds `[::]` rather than reusing the HTTP host setting.
  ([v18 gRPC server source](https://github.com/Arize-ai/phoenix/blob/arize-phoenix-v18.0.0/src/phoenix/server/grpc_server.py#L105-L112))
- The official client can retrieve spans with
  `Client.spans.get_spans(...)`/`get_spans_dataframe(...)`; the REST API also
  exposes cursor-paginated Phoenix JSON and OTLP-shaped span export. The latter
  is a reconstructed OTLP projection, not the original request envelope.
  ([v18 export documentation](https://github.com/Arize-ai/phoenix/blob/arize-phoenix-v18.0.0/docs/phoenix/tracing/how-to-tracing/importing-and-exporting-traces/extract-data-from-spans.mdx),
  [v18 REST source](https://github.com/Arize-ai/phoenix/blob/arize-phoenix-v18.0.0/src/phoenix/server/api/routers/v1/spans.py#L719-L1082))
- Persistent span insertion is keyed by span ID and uses conflict-ignore. The
  higher-level `Client.spans.log_spans()` endpoint instead rejects a batch when
  a pre-existing duplicate is found; duplicates inside the same queued ingest
  can still reach the conflict-ignore layer. ([v18 insertion source](https://github.com/Arize-ai/phoenix/blob/arize-phoenix-v18.0.0/src/phoenix/db/insertion/span.py#L137-L162),
  [v18 client source](https://github.com/Arize-ai/phoenix/blob/arize-phoenix-v18.0.0/packages/phoenix-client/src/phoenix/client/resources/spans/__init__.py#L586-L640))
- Retention is indefinite by default (`max_days=0`). Policies can cap age,
  trace count, or either on a cron schedule. Projects, traces, sessions,
  datasets, experiments, prompts, and individual spans also have deletion
  surfaces; deleting one span can orphan descendants. ([v18 retention docs](https://github.com/Arize-ai/phoenix/blob/arize-phoenix-v18.0.0/docs/phoenix/settings/data-retention.mdx),
  [v18 span deletion source](https://github.com/Arize-ai/phoenix/blob/arize-phoenix-v18.0.0/src/phoenix/server/api/routers/v1/spans.py#L1464-L1480))
- Evaluation UI: versioned datasets, CSV/JSONL export, human annotations,
  dataset experiments, evaluator annotations, comparisons, and baselines.
  Stored LLM spans can be reopened and rerun in Playground; this is single-call
  replay, not full application/tool/retrieval replay. ([datasets](https://github.com/Arize-ai/phoenix/blob/arize-phoenix-v18.0.0/docs/phoenix/datasets-and-experiments/concepts-datasets.mdx),
  [annotations](https://github.com/Arize-ai/phoenix/blob/arize-phoenix-v18.0.0/docs/phoenix/tracing/how-to-tracing/feedback-and-annotations/annotating-in-the-ui.mdx),
  [experiments](https://github.com/Arize-ai/phoenix/blob/arize-phoenix-v18.0.0/docs/phoenix/datasets-and-experiments/how-to-experiments/run-experiments.mdx#L324-L397),
  [span replay](https://github.com/Arize-ai/phoenix/blob/arize-phoenix-v18.0.0/docs/phoenix/prompt-engineering/overview-prompts/span-replay.mdx))
- Phoenix's SQLite engine uses WAL and sets `PRAGMA synchronous=OFF`.
  ([v18 engine source](https://github.com/Arize-ai/phoenix/blob/arize-phoenix-v18.0.0/src/phoenix/db/engines.py#L38-L44))

### B. TheraPy-specific inferences

- Phoenix's native OpenInference model and LLM-span Playground are the closest
  fit to the proposed restricted interaction/evaluation plane. The owned
  SQLite/FULL journal must remain the durable source of truth: Phoenix's
  SQLite durability setting is not an acceptable replacement.
- Pipecat/FastAPI integration is direct OTLP HTTP with no experiment-ID header.
  A production local launch still needs a network decision: v18's gRPC server
  binds `[::]` independently of the HTTP `--host`. This spike disabled the
  unused gRPC server in a documented launcher shim to preserve loopback-only
  scope.
- Conflict-ignore protects an already stored span from later overwrite, but
  Phoenix's OTLP 200 response did not surface the two fixture ID collisions or
  the exact resend. An owned exporter/journal must therefore make collision and
  delivery state visible instead of interpreting HTTP success as persistence.
- Phoenix offers stronger immediately relevant evaluation/replay workflow than
  MLflow, but exact end-to-end TheraPy replay still belongs in the owned harness.

### C. Measured results

**Invocation**

```text
.local/obs-spike/phoenix-venv/bin/python scripts/observability/backend_spike.py serve-phoenix --host 127.0.0.1 --port 62006
OTLP: http://127.0.0.1:62006/v1/traces
SQLite: .local/obs-spike/storage/phoenix/phoenix.db
```

Environment: `PHOENIX_TELEMETRY_ENABLED=false`,
`PHOENIX_ALLOW_EXTERNAL_RESOURCES=false`,
`PHOENIX_DISABLE_AGENT_ASSISTANT=true`,
`PHOENIX_AGENTS_DISABLE_WEB_ACCESS=true`, plus explicit host/port/working-dir/
SQLite URL. `lsof` showed only `127.0.0.1:62006`; the launcher disabled gRPC.

| Measurement | Result |
|---|---:|
| Package | `arize-phoenix==18.0.0` |
| OTel/OpenInference pins | `1.43.0` / `0.1.30` |
| First 15-span export | **11.837 ms**, exporter `SUCCESS` |
| Query latency, 20 repetitions | **p50 22.913 ms; p95 50.168 ms** |
| Idle server RSS | **361,025,536 B (344.30 MiB)**, one process |
| Disk after first ingest | **28,593,708 B (27.27 MiB)** |
| Disk amplification | **863.415×** (dominated by fixed schema overhead at this corpus size) |
| Disk after identical resend | 28,593,708 B; unchanged |
| Queried/matched records | **15 / 15** |
| Canonical leaf losses | **0** — every canonical envelope was field-exact |
| Attribute representation changes | **0** |
| Attribute-map discrepancies | **15** (`openinference.span.kind` promoted to Phoenix's top-level `span_kind` column; value preserved, position structural) |
| Transport-ID mismatches | **0** |
| Duplicate resend | exporter `SUCCESS`; **ignored, no row growth** (15 → 15) |
| Outage probe | exporter `FAILURE` in 4.575 ms |

All 15 canonical envelopes round-tripped field-exactly; no field was lost,
truncated, or reordered. The only attribute-map difference is Phoenix
consuming `openinference.span.kind` into its own `span_kind` column — the
value is preserved and recoverable, so this is structural, not content loss.
The earlier colliding-corpus run additionally showed conflict-ignore
(first-write-wins) semantics for spans sharing transport IDs, with OTLP still
returning success — collision/delivery evidence must stay owned.

Raw storage scan counts (corrected corpus):

| Content canary | Count | Content canary | Count |
|---|---:|---|---:|
| completion | 35 | memory_note | 44 |
| provider_error | 36 | retrieval_passage | 44 |
| system_prompt | 46 | tool_arguments | 4 |
| tool_definition | 4 | tool_result | 2 |
| typed_transcript | 68 | voice_transcript | 44 |

All 10 content canaries were present. All six forbidden canaries
(`api_key`, `auth_header`, `push_key`, `query_string`, `sdp`, `turn_key`) had
count **0**.

---

## Candidate 2 — MLflow 3.14.0

### A. Officially documented/versioned facts

- MLflow 3.14.0 runs a local tracking server with a SQLite backend and local
  artifact destination. Since 3.3, ordinary trace info and span JSON are stored
  in SQL; artifact storage remains relevant to run artifacts and trace binary
  attachments. ([3.14.0 package](https://pypi.org/project/mlflow/),
  [v3.14 tracking-store source](https://github.com/mlflow/mlflow/blob/v3.14.0/mlflow/store/tracking/__init__.py#L12-L23),
  [v3.14 CLI artifact flags](https://github.com/mlflow/mlflow/blob/v3.14.0/mlflow/utils/cli_args.py#L208-L232))
- MLflow 3.6+ exposes OTLP/HTTP `POST /v1/traces` for SQL stores. It requires
  `x-mlflow-experiment-id`, has no OTLP/gRPC route, and persists each request as
  one all-or-nothing `log_spans()` batch. ([v3.14 OTLP docs](https://github.com/mlflow/mlflow/blob/v3.14.0/docs/docs/genai/tracing/opentelemetry/ingest-shared.mdx),
  [v3.14 route source](https://github.com/mlflow/mlflow/blob/v3.14.0/mlflow/server/otel_api.py#L91-L227))
- `MLFLOW_DISABLE_TELEMETRY=true` or `DO_NOT_TRACK=true` disables usage
  telemetry. ([v3.14 usage tracking docs](https://github.com/mlflow/mlflow/blob/v3.14.0/docs/docs/community/usage-tracking.mdx),
  [v3.14 telemetry source](https://github.com/mlflow/mlflow/blob/v3.14.0/mlflow/telemetry/utils.py#L97-L114))
- `MlflowClient.search_traces(locations=[experiment_id], include_spans=True)`
  is the supported SDK query. The SDK searches trace metadata and fetches spans;
  callers should not hard-code its internal REST endpoints. ([v3.14 client source](https://github.com/mlflow/mlflow/blob/v3.14.0/mlflow/tracking/client.py#L1416-L1477),
  [v3.14 REST-store source](https://github.com/mlflow/mlflow/blob/v3.14.0/mlflow/store/tracking/rest_store.py#L575-L652))
- Trace ID is globally unique and span primary key is `(trace_id, span_id)`.
  SQL `log_spans` reuses the trace and upserts span rows, updating non-key
  columns. Replaying OTLP can also re-merge token/cost aggregates, so row-count
  idempotency does not imply aggregate idempotency. ([v3.14 models](https://github.com/mlflow/mlflow/blob/v3.14.0/mlflow/store/tracking/dbmodels/models.py#L720-L782),
  [v3.14 SQL store](https://github.com/mlflow/mlflow/blob/v3.14.0/mlflow/store/tracking/sqlalchemy_store.py#L5065-L5147))
- Trace archival is optional, not automatic deletion: finalized span payloads
  can move from SQL to an artifact repository while trace info/pointers remain.
  Archived full-text span filtering is reduced and appending spans is rejected.
  Hard deletion is separate through `delete_traces` and is irreversible.
  ([v3.14 archive docs](https://github.com/mlflow/mlflow/blob/v3.14.0/docs/docs/genai/tracing/observe-with-traces/archive-traces.mdx),
  [v3.14 delete API](https://github.com/mlflow/mlflow/blob/v3.14.0/mlflow/tracking/client.py#L1321-L1376))
- GenAI Evaluation Datasets require SQL and can be populated from traces.
  MLflow 3.14's OSS Review Queues are experimental and support pass/fail,
  categorical, numeric, and text assessments. Playground can issue edited chat
  requests through the AI Gateway, but the checked v3.14 surfaces do not
  establish historical trace-to-application replay. ([datasets](https://github.com/mlflow/mlflow/blob/v3.14.0/docs/docs/genai/datasets/index.mdx),
  [review queues](https://github.com/mlflow/mlflow/blob/v3.14.0/docs/docs/genai/assessments/review-queues.mdx),
  [playground API source](https://github.com/mlflow/mlflow/blob/v3.14.0/mlflow/server/js/src/experiment-tracking/pages/playground/api.ts#L31-L43))

### B. TheraPy-specific inferences

- OTLP/FastAPI integration is viable, but every exporter must obtain and carry
  an MLflow experiment ID. Phoenix's OpenInference-native presentation is a
  closer semantic match to the existing canonical interaction contract.
- Last-write-wins upsert is useful for corrections but unsafe as an implicit
  retry contract: an accidental ID collision replaces span content while OTLP
  still reports success. The owned journal needs immutable interaction IDs and
  explicit collision diagnostics.
- MLflow's datasets and new review queues are useful for evaluation, but the
  v3.14 OSS evidence is weaker for historical LLM-span replay and the review
  queue is newly experimental. Exact TheraPy reconstruction/replay remains an
  owned capability.
- The default local server started a job runner and seven Huey-related
  processes even though only trace ingestion/query was exercised. The
  documented `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false` could reduce this, but
  that alternative was not measured and may remove evaluation job capability.

### C. Measured results

**Invocation**

```text
.local/obs-spike/mlflow-venv/bin/mlflow server --host 127.0.0.1 --port 62007 --backend-store-uri sqlite:////Users/jsugg/dev/github/TheraPy/.local/obs-spike/storage/mlflow/mlflow.db --serve-artifacts --default-artifact-root mlflow-artifacts:/ --artifacts-destination /Users/jsugg/dev/github/TheraPy/.local/obs-spike/storage/mlflow/artifacts --workers 1
OTLP: http://127.0.0.1:62007/v1/traces
```

Environment: `MLFLOW_DISABLE_TELEMETRY=true`, `DO_NOT_TRACK=true`, and client
`MLFLOW_TRACKING_URI=http://127.0.0.1:62007`. A direct SDK check returned no
telemetry client. `lsof` showed only `127.0.0.1:62007`.

| Measurement | Result |
|---|---:|
| Package | `mlflow==3.14.0` |
| OTel/OpenInference pins | `1.43.0` / `0.1.30` |
| First 15-span export | **1053.846 ms**, exporter `SUCCESS` |
| Query latency, 20 repetitions | **p50 220.554 ms; p95 309.156 ms** |
| Idle server master RSS | **216,182,784 B (206.17 MiB)** |
| Idle full process-tree RSS | **1,987,334,144 B (1,895.27 MiB)**, 11 processes |
| Disk after first ingest | **925,696 B (0.883 MiB)** |
| Disk amplification | **27.952×** |
| Disk after identical resend | 925,696 B; unchanged |
| Queried/matched records | **15 / 15** |
| Canonical leaf losses | **0** — every canonical envelope was field-exact after SDK JSON decoding |
| Irreversible attribute representation changes | **5** (four provider error bodies and one structured assistant message decoded to objects, changing string type/whitespace) |
| Reversible JSON-decoded attributes | **67** |
| Transport-ID mismatches | **0** |
| Duplicate resend | exporter `SUCCESS`; **no row growth, no visible change** (15 → 15) |
| Outage probe | exporter `FAILURE` in 4.666 ms |

All 15 canonical envelopes round-tripped field-exactly after accounting for
the SDK's JSON decoding. However, MLflow irreversibly changed the stored
representation of five exact-evidence attributes — including four provider
error bodies, which the capture contract requires byte-exact. The earlier
colliding-corpus run additionally showed last-write-wins span upsert for
conflicting IDs with OTLP still returning success.

Raw storage scan counts (corrected corpus):

| Content canary | Count | Content canary | Count |
|---|---:|---|---:|
| completion | 55 | memory_note | 46 |
| provider_error | 21 | retrieval_passage | 46 |
| system_prompt | 48 | tool_arguments | 6 |
| tool_definition | 5 | tool_result | 4 |
| typed_transcript | 79 | voice_transcript | 46 |

All 10 content canaries were present. All six forbidden canaries had count
**0**.

---

## Candidate 3 — Langfuse Cloud Hobby

### A. Officially documented facts (verified 2026-07-15)

- Hobby is free, includes **50,000 units/month**, **30 days historical-data
  access**, and **2 users**. Units include traces, observations, and scores;
  experiments, judges, and annotations also consume units. ([pricing](https://langfuse.com/pricing),
  [billable units](https://langfuse.com/docs/administration/billable-units))
- Safest retention wording is “30-day accessible history; fixed,
  non-configurable Hobby retention,” not a guaranteed hard deletion exactly on
  day 30. Configurable retention policies are unavailable on Hobby/Core.
  ([retention](https://langfuse.com/docs/administration/data-retention),
  [support handbook](https://langfuse.com/handbook/support/how-to-answer-support-questions))
- UI/API deletion supports individual, batch, and filtered traces, including
  related observations/scores. Project deletion is irreversible. Trace deletion
  is documented as usually completing within about 15 minutes without a
  completion confirmation. ([deletion](https://langfuse.com/docs/administration/data-deletion))
- UI capabilities include versioned datasets/CSV import, annotation queues,
  prompt/model experiments with optional evaluators, session browsing, and
  reopening an individual generation in Playground. These docs do not prove
  deterministic whole-trace application replay. ([datasets](https://langfuse.com/docs/evaluation/experiments/datasets),
  [annotation queues](https://langfuse.com/docs/evaluation/evaluation-methods/annotation-queues),
  [UI experiments](https://langfuse.com/docs/evaluation/experiments/experiments-via-ui),
  [sessions](https://langfuse.com/docs/observability/features/sessions),
  [playground](https://langfuse.com/docs/prompt-management/features/playground))
- Filtered UI data can be exported as CSV/JSON and the public API states all
  platform data/features are API-accessible. Scheduled blob export is not
  available on Hobby/Core. ([UI export](https://langfuse.com/docs/api-and-data-platform/features/export-from-ui),
  [public API](https://langfuse.com/docs/api-and-data-platform/features/public-api),
  [blob export availability](https://langfuse.com/docs/api-and-data-platform/features/export-to-blob-storage))

### B. TheraPy-specific inferences

- Managed operation and near-zero local server footprint could be attractive,
  but remote latency, quota consumption, exact attribute reconstruction,
  duplicate behavior, canary persistence, outage behavior, and practical export
  remain unknown for this corpus.
- Hobby's fixed accessible-history window and lack of scheduled blob export
  make the owned journal/open export path more important, not less.
- A real measurement requires an owner-scoped test key. It must not use a
  personal production credential or trigger a signup from this spike.

### C. Measured results (2026-07-16, owner-provided scoped key)

The owner supplied `LANGFUSE_{PUBLIC,SECRET}_KEY`/`LANGFUSE_BASE_URL`, and
the identical corpus (`8198524e…`) ran via
`scripts/observability/backend_spike_langfuse.py` (raw:
`.local/obs-spike/results/langfuse.json`):

| Measurement | Result |
|---|---:|
| OTLP ingest (`/api/public/otel/v1/traces`, Basic auth) | **SUCCESS**, 15/15 spans; first export **1368.8 ms** |
| Queryable traces | **15 / 15** (async ingestion; polled) |
| Canonical envelope round-trip via public API | **exact**, 0 losses/transformations |
| Query latency (per-trace, remote + Hobby rate limits) | **p50 1146.8 ms; p95 25177.7 ms** (429 backoffs included) |
| Duplicate resend | accepted, **no new trace IDs** |
| Content canaries via API | all 10 present |
| Forbidden canaries via API | all 6 absent (API-level scan only — remote storage cannot be inspected) |
| Local footprint | none (managed) |

Measurement limits: retention/deletion completion, access controls, and
storage-level scans remain documented-only (no remote storage access);
Hobby-tier public-API rate limits materially shape query latency.

---

## Candidate ranking by the fixed decision order

“Unmeasured” is not treated as a passing result. Rankings combine the official
facts, explicit TheraPy inferences, and the two local runs above.

| Priority | Current evidence ordering | Why |
|---:|---|---|
| 1. Capture correctness/completeness | **Phoenix > MLflow; Langfuse unmeasured** | On the corrected corpus both retained all 15 records with field-exact canonical envelopes, but Phoenix stored every attribute representation exactly (its only difference is structural span-kind promotion), while MLflow irreversibly rewrote 5 exact-evidence attributes — including four provider error bodies — and JSON-decoded 67 more. |
| 2. Evaluation usefulness | **Phoenix > MLflow; Langfuse promising but unmeasured** | Phoenix has mature span-to-Playground replay plus datasets/annotations/experiments. MLflow has datasets and experimental review queues but no verified historical application replay. Langfuse documents a broad workflow but has no corpus evidence. |
| 3. Loss visibility | **Phoenix > MLflow; Langfuse unmeasured** | Neither OTLP path exposed collision behavior: both returned success. Phoenix preserves first write; MLflow silently overwrites the span row and can re-merge aggregates. Both require owned collision/delivery evidence. |
| 4. Pipecat/FastAPI integration fit | **Phoenix > MLflow; Langfuse unmeasured** | Both accept OTLP HTTP; Phoenix is OpenInference-native and needs no experiment header. MLflow requires experiment lifecycle/header plumbing. |
| 5. Resource footprint | **Phoenix > MLflow locally; Langfuse local footprint inferred low but unmeasured end-to-end** | Phoenix used 344.30 MiB in one process with ~10x faster export and query. MLflow's default server tree used 1.90 GiB across 11 processes despite a smaller SQLite file. |
| 6. Operational complexity | **Langfuse inferred simplest > Phoenix > MLflow** | Managed Langfuse avoids a local server but adds credential/quota/vendor operation. Phoenix needs the v18 gRPC bind mitigation. MLflow launched a 10-process tree and requires experiment/artifact/job configuration. |
| 7. Open export | **Phoenix > MLflow > Langfuse Hobby** | Phoenix exposes SDK, JSON, and OTLP-shaped span export. MLflow exposes SDK/SQL-backed trace access and archive. Langfuse has public API/manual JSON/CSV, but Hobby lacks scheduled blob export. |
| 8. Cost | **Phoenix > MLflow > Langfuse Hobby at this scale** | All have zero software/subscription price for this spike. Phoenix's local resource cost was materially below MLflow's default process tree. Langfuse is free only within the 50k-unit/month and history limits. |

## Selected backend (ADR)

**Selected: Phoenix `arize-phoenix==18.0.0`, local pinned image/package,
loopback-only, SQLite storage, deployed behind the opt-in
`llm-observability` Compose profile (plan O1.2 item 5).**

Rationale against the fixed ranking order:

1. **Capture correctness** — Phoenix is the only measured candidate that
   stores every exported attribute representation exactly; MLflow mutates
   provider error bodies, which the capture contract requires byte-exact.
2. **Evaluation usefulness** — datasets, annotations, experiments, and
   span-to-Playground replay are mature and OpenInference-native.
3. **Loss visibility** — neither candidate surfaces ID collisions over OTLP;
   this is compensated in owned code: the journal remains the source of
   truth, interaction IDs are immutable, and the export worker records
   explicit per-record ACK state (plan §5.3) instead of trusting HTTP 200.
4. **Fit** — direct OTLP HTTP, no experiment-header plumbing.
5. **Resources** — one process, 344 MiB RSS, ~10x faster export/query than
   MLflow's 11-process default tree.

Deployment conditions bound to this selection (from §10 of the plan and the
findings above):

- Pinned `version-18.0.0-nonroot`-pattern image (container) or
  `arize-phoenix==18.0.0` (package); upgrades re-run this spike's replay.
- `PHOENIX_TELEMETRY_ENABLED=false`, `PHOENIX_ALLOW_EXTERNAL_RESOURCES=false`,
  explicit `PHOENIX_DEFAULT_RETENTION_POLICY_DAYS=30` for synthetic data,
  dedicated persistent volume, resource caps, loopback/tailnet-only UI.
- The v18 gRPC collector binds `[::]` independently of `--host`: the
  deployment must disable or firewall it (the spike's launcher shim disables
  it); this is re-verified at every upgrade.
- Phoenix runs `PRAGMA synchronous=OFF`: it is a derived, disposable
  evaluation store. Nothing in the product or evaluation path may treat
  Phoenix persistence as durability; the owned journal keeps records until
  Phoenix ACK and survives Phoenix volume loss.
- Remote export stays off (`THERAPY_INTERACTION_REMOTE_EXPORT=0`); Phoenix is
  local, so this gate remains untouched.

**Langfuse Cloud** was subsequently measured (2026-07-16) with an
owner-provided scoped key: the identical corpus round-trips exactly and all
canary checks pass (section above). It does not displace Phoenix under the
fixed ranking: capture correctness ties, but remote query latency (p50
1.15 s / p95 25 s under Hobby rate limits) versus Phoenix's local 23/50 ms,
plus the plan's local tie-break (reproducibility/control/resource cost) and
the Hobby unit/retention limits keep the local candidate selected. LangSmith
was not tested — the plan requires it only before REJECTING hosted mode,
and hosted mode was measured and viable, just not selected.

**Executable gate reconciliation (audit F-03):** `backend_spike.py` now (a)
returns a nonzero exit code whenever `overall_pass` is false, and (b)
encodes the documented structural-promotion criterion — consuming
`openinference.span.kind` into a backend's own span-kind column is recorded
as `structural_promotions`, not content loss. Re-run results: **Phoenix
`overall_pass: true`** (15 structural promotions, 0 content losses);
**MLflow `overall_pass: false`** (its 5 irreversible representation
mutations are genuine capture-contract failures). The machine gate and this
decision record now agree.

**Config consequence:** `THERAPY_INTERACTION_BACKEND` gains the enum value
`phoenix` (alongside `journal`) when the O1.2 adapter lands; the default
remains `journal` until the O1 gate passes.
