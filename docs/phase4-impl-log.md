# Phase 4 implementation log

**Status:** all automated Phase 4 engineering and acceptance gates landed;
owner-confirmed multi-week dogfood insight remains the sole completion gate.

This log records implementation and verification evidence. Phase 4 must not be
called complete until the owner confirms one organic, true, non-obvious insight
in conversation after normal multi-week use.

| Workstream | Evidence | Status |
|---|---|---|
| W1 Property graph | Ordered v1 graph migration, evidence/lifecycle/tombstone/deletion tests. | Automated complete |
| W2 Distillation/inbox/graduation | Transactional session runs, strict extraction, quote verification, node/edge judgment. | Automated complete |
| W3 Graph context | Per-turn multilingual semantic graph + bounded episodic retrieval in Pipecat. | Automated complete |
| W4 Longitudinal insight | Durable queue, adjacent delivery, recap, UI/digest, exact conversational resolution. | Automated complete |
| W5 Proactivity | Persisted scheduler/ledger and push, greeting, check-in, digest delivery. | Automated complete |
| W6 Research KB | Local five-format ingest/OCR, versioned semantic index, grounding/citations, corpus UX/CLI. | Automated complete |
| W7 Review UI sovereignty | Accessible graph/list/audit/edit/delete/boundary/insight/corpus/data workspace. | Automated complete |
| W8 Crisis config | Strict structured environment config, safe fallback, docs, explicit API failure. | Automated complete |

## Remaining work

- Sole remainder: the manual north-star dogfood protocol at the end of this log.
- SER/Phase 3 and Phase 6/VPS work remain explicitly out of Phase 4 scope.

## 2026-07-15 — Phase A/B continuation checkpoint

- Added ordered knowledge schema migration v1 with pre-migration SQLite backup;
  converted legacy registries/statuses and normalized node/edge evidence.
- Replaced registries with Appendix A values plus explicit legacy mappings.
- Added identical node/edge proposal, confirmation, rejection, decay, source-
  deletion, and lifecycle-event paths. Confirmation now requires a proposal;
  direct owner statements use separate trusted APIs.
- Replaced plaintext/ID-based tombstones with local-key HMAC digests over
  collision-safe canonical node/edge identities. Legacy plaintext tombstones
  migrate to keyed endpoint-stable identities instead of being discarded.
  `never_store` now guards statements, quotes, aliases, edits, migrations, and
  legacy inbox rows and performs an audited purge of matching graph, incident
  edge metadata, inbox, summaries, and research records.
- Added session-scoped idempotent distillation runs; complete candidate
  validation; exact user-turn quote verification; alias-aware edge resolution;
  atomic promotion/evidence/inbox consumption; bounded retries; durable
  negative-judgment snapshots; separate node/edge graduation judgment.
- Retired flat-fact reads/writes after one-time graph migration. Added source-
  deletion modes: sanitize quotes plus deleted-source provenance, or remove
  derived evidence and unsupported claims.
- Decoupled summary, distillation, recap, and title work in the Pipecat
  finalizer so one provider failure cannot suppress other artifacts.
- Added deterministic candidate-parser fuzzing and an overlapping-finalizer
  concurrency regression proving one atomic evidence/inbox commit.

Evidence:

- `.venv/bin/ruff check src/therapy/knowledge src/therapy/memory/store.py tests/suites/integration/test_user_model.py tests/suites/integration/test_distill.py`
  — pass.
- `.venv/bin/python -m pytest tests/suites/integration/test_user_model.py tests/suites/integration/test_distill.py -q`
  — 20 passed.

Migration: existing partial graph DBs receive a timestamped
`therapy.db.pre-phase4-*.bak` before schema conversion. Tombstone key is stored
owner-locally as mode-0600 `tombstone.key`.

Remaining risk / next checkpoint: wire deletion modes and complete-store
export/restore through validated API/CLI; replace opening-only lexical context
with per-turn multilingual graph + episodic retrieval; implement durable
pending-insight delivery/confirmation.

## 2026-07-15 — Phase C/D checkpoint

- Added a per-turn `ContextAssembler` with versioned local multilingual E5
  embeddings, bounded semantic graph traversal, relevant episodic summaries,
  active goals/threads, deterministic precedence, and application-enforced
  `never_initiate`. Protected boundary values no longer leak into unsolicited
  model context. User-raised private topics unlock only matching private
  nodes/episodes; unrelated protected material remains excluded.
- Pipecat now replaces one bounded longitudinal context block on every user
  turn. Confirmation/rejection resolves only the exact insight most recently
  raised in that session, and no second insight is raised on the resolution
  turn.
- Added durable pending-insight delivery/snooze/dismiss/resolution history and
  independent owner-facing recaps.
- Added validated graph/node/edge/boundary/insight APIs and the vanilla-JS model
  workspace: list + SVG graph, accessible fallback, filters, provenance audit,
  edits/deletes, pending actions, and boundaries. Existing companion/WebRTC DOM
  hooks remain intact. Direct graph resolution and owner edits also synchronize
  the matching durable queue record, preventing stale pending actions.

Migration: no additional schema version; C/D use the v1 graph and normalized
evidence/lifecycle tables.

Remaining risk / next checkpoint: persistent four-channel proactivity and the
local research/OCR corpus.

## 2026-07-15 — Phase E checkpoint

- Schema migration v2 adds four opt-in channel settings, IANA timezone/quiet
  hours, persistent jobs with idempotency/attempts/results, subscriptions,
  in-app messages, and written digests.
- Added a bounded lifespan scheduler with restart recovery, overnight/DST quiet
  handling, delivery-time `never_initiate` rechecks, deterministic jittered
  retries, and observable ledger failures.
- Implemented encrypted Web Push with owner-local mode-0600 P-256 VAPID keys
  and minimal generic payloads; in-app greeting, scheduled check-in, and
  daily/weekly digest delivery; service-worker push/click handling; and PWA
  channel controls. All channels default off.

Evidence: `tests/suites/integration/test_outreach.py` covers defaults, IANA
validation, restart idempotency, DST quiet hours, delivery-time privacy,
minimal push payload, all local channels, and bounded retry.

Remaining risk / next checkpoint: five-format local research/OCR and complete
owner-data sovereignty.

## 2026-07-15 — Phase F/G checkpoint

- Schema migration v3 adds source documents, stable extraction/OCR blocks, and
  versioned semantic index generations; legacy lexical research tables migrate
  without source loss.
- Added bounded PDF/image/HTML/Markdown/text ingest, exact extension/MIME/magic
  validation, digital-PDF-first extraction, scanned-page/image Tesseract OCR,
  EXIF/orientation/preprocessing, confidence/bounding-box audit, low-confidence
  correction, stable page/heading/block citations, and active-content stripping.
- Production retrieval uses local `intfloat/multilingual-e5-small`, revision
  `fd1525a9fd15316a2d503bf26ab031a61d056e98`, dimension 384, with E5 task
  prefixes and deterministic chunk-policy reindexing. A real model smoke ranked
  English energy/planning passages first for Spanish and Portuguese queries.
- Research grounding now influences per-turn technique selection silently;
  psychoeducation is passage-only and carries exact source anchors.
- Added PWA/API/CLI corpus ingest, preview/correct, list, reindex, delete, export,
  restore; one complete whitelisted owner snapshot covers sessions/audio,
  graph/provenance/inbox/runs/embeddings, proactivity/subscriptions/messages,
  corpus/artifacts, tombstone key, and VAPID key. Restore validates fully and
  rolls back atomically; delete removes every owned store and backup.
- Session deletion exposes both settled modes and plainly discloses whether
  sanitized learned knowledge survives. Crisis contacts are strict JSON with
  an independent safe response fallback and explicit API diagnostics.
- Container OCR smoke recognized “Visible checklist supports planning” locally
  at 0.89 confidence with installed `eng`, `spa`, and `por` packs.

Remaining risk / next checkpoint: real server/agent/browser acceptance and the
complete isolated quality gate.

## 2026-07-15 — Phase H automated acceptance checkpoint

- Replaced direct-domain verification with two real isolated uvicorn processes.
  `make verify-longitudinal-loop` now drives HTTP agent turns and proves three
  finalized sessions -> proposed nodes/edge -> adjacent reflections -> explicit
  conversational confirmation -> confirmed nodes and edge; per-turn refreshed
  context; unsolicited/user-raised privacy behavior; research-grounded response
  plus exact citation; quiet-hour postponement; full-process restart idempotency;
  crisis config; and owner export.
- Deterministic doubles exist only at LLM/STT/TTS/embedding/OCR external test
  boundaries. Container browser acceptance still exercises the real FastAPI
  lifespan, Pipecat/WebRTC frame path, persistence, APIs, scheduler, service
  worker, and vanilla-JS UI.
- Browser E2E covers model graph/list/audit/edit/delete, type/status filters,
  node and edge pending resolution, boundary add/remove, greeting/digest
  presentation, corpus upload/preview/correction/delete, PWA
  export/delete/restore, push payload minimization, and notification-click
  navigation. Server acceptance covers PDF, image, HTML, Markdown, and text
  uploads plus local image/scanned-PDF OCR review and exact citations. It found
  and fixed SVG `hidden` handling, immediate typed-turn echo reconciliation,
  stale queue records, private-topic overexposure, and stale WebRTC ownership
  during destructive data operations. Peer-scoped disconnect now drains
  deterministic test finalizers without shutting down the reusable production
  gateway.
- Final automated evidence:
  - host: `pytest -q` — 209 passed, 2 skipped, 9 E2E deselected;
  - container unit/integration: 215 passed, 9 E2E deselected;
  - container Playwright: 9 passed;
  - full `ruff check .`, `uv lock --check`, `git diff --check` — pass;
  - Pipecat imports and pipeline/API integration — pass;
  - real local Tesseract and multilingual production-embedding smoke — pass.

Migrations: ordered schema version is now 3: v1 graph/provenance, v2
proactivity, v3 research/index. Each old partial graph receives a timestamped
pre-migration SQLite backup.

## Sole outstanding stopping condition — owner dogfood

Use TheraPy normally for several weeks without acceptance seeds or manual graph
promotion. The north-star gate passes only when Rowan organically raises a
text-derived cognitive, routine/energy, or relational insight that is:

1. supported by real evidence across the graduation floor;
2. genuinely true and non-obvious to Juan Pedro;
3. raised adjacent to the live topic, not from the review UI or a test route;
4. explicitly confirmed by Juan Pedro in that same conversational protocol.

Record the date, insight/claim identifiers, evidence-session count, and a
privacy-safe paraphrase here. Do not record sensitive quotes. Until that owner
confirmation is recorded, status remains **automated complete; Phase 4 not yet
manually complete**.
