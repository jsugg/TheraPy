# GOAL — The longitudinal self-knowledge loop

*(Phase 4 + the Appendix A user-model foundation it depends on)*

**Status:** partial engineering slice landed
**Implementation log:** [phase4-impl-log.md](phase4-impl-log.md)
**Owner:** Juan Pedro Sugg
**Date:** 2026-07-12
**Related:** [SPEC.md](SPEC.md) §3, §5, §9 (Phase 4), Appendix A · [ux-companion-spec.md](ux-companion-spec.md)

---

## Objective

Turn accumulated conversation into **confirmed, reflected-back self-knowledge**. Replace the flat-`facts` placeholder shipped in Phase 2 with the **Appendix A property-graph user model**, then layer on distillation/graduation, graph-aware context assembly, cross-session insight, proactive reflection channels, and the curated research KB.

Emotion is deliberately sourced later: the graph is built **emotion-source-agnostic** (its `source` field already admits `ser` alongside `conversation | user-stated`), so `ser`/Phase 3 slots into the same substrate with no rework. This goal builds everything the north-star loop needs *except* the emotional dimension.

## Definition of done

The SPEC's **north-star test, text-only variant**: after regular dogfooding, the assistant surfaces at least one **true, non-obvious self-insight** — cognitive, energy/routine, or relational (emotional patterns wait for `ser`) — that graduates observation → pattern → confirmed, and the owner confirms it in conversation. Supporting: the whole model is browsable/editable/deletable in the PWA, at least one proactive channel fires within boundaries + quiet hours, and the research KB both silently grounds technique choice and cites sources on demand.

## Scope

| In scope | Out of scope |
|----------|--------------|
| Appendix A property-graph user model (nodes + edges + claim lifecycle) | **`ser` / Phase 3** — per-turn emotion, emotion recap, emotion in review UI, retroactive re-analysis, two-layer emotion representation, ser-driven register modulation, avatar-register reactions |
| Distillation, observation inbox, graduation engine | **Phase 5** — P3 rehearsal, P4 daily-structure/OT, VPS migration |
| Graph-walk context assembly | VPS prerequisites deferred with it: **encryption at rest, real PWA auth, `hostwatch` launchd daemon** |
| Cross-session insight + reflections/recaps (text-derived) | Streaming SER, ambient devices (parked) |
| Proactivity engine — all four channels, quiet hours, `never_initiate` | |
| Research KB v1 — ingest, silent grounding, source-attributed psychoeducation | |
| Review-UI sovereignty (browse/edit/delete graph + pending-insights inbox) | |
| Crisis-resource configuration (small, non-VPS safety item) | |

## Workstreams & deliverables

**W1 — Property-graph user model** `knowledge/user_model.py` (framework-free)
- `nodes` + `edges` tables in SQLite; **extensible type registries** (config, not migration) for the 11 node types and the starter edge catalog.
- Claim-lifecycle fields on **both** nodes and edges: `statement` (canonical English), `quotes[]` (verbatim, language-tagged), `n_occurrences` + `n_sessions`, `status` (observation|pattern|confirmed), `source`, `first_seen`/`last_seen`/`user_edited`.
- **Migration**: existing v1 `facts` rows imported as `observation` nodes (no data loss); flat store retired.
- **Boundaries**: `never_store` checked before *any* write (incl. inbox); `never_initiate` exposed to context assembly + proactivity.
- Deletion → **tombstone** so distillation can't re-learn removed items; per-type **decay**.

**W2 — Distillation, inbox & graduation** `knowledge/distill.py`
- Freeform **observation inbox** written during conversation (zero schema pressure).
- Between-session `distill.py`: inbox + transcript → node/edge promotion, attaching quotes; discards the rest. Replaces the current disconnect-time flat-fact extraction in `memory/summarizer.py`.
- **Graduation engine**: mechanical floor (≥3 occurrences across ≥2 sessions) → eligible; LLM judgment → *propose*; **explicit user validation** → `confirmed`. `user-stated` starts confirmed.

**W3 — Graph-walk context assembly** (feeds `dialogue/policy.py` context)
- Always in context: identity + preferences + the `never_initiate` list.
- Active goals/threads + a **graph walk** from current topic/state to top-K relevant nodes **with their confirmed edges**. Priority confirmed > pattern > observation; inbox reaches the LLM only during distillation.
- Replaces the Phase-2 "distilled summaries + flat facts" continuity injection.

**W4 — Longitudinal insight** (P2's non-emotional half)
- Cross-session pattern queries over the graph/timeline; end-of-session **reflections/recaps** (text-derived).
- **Pending-insights inbox**: context-aware in-conversation raise when the topic is adjacent, otherwise queued — never derails the moment.

**W5 — Proactivity engine** `dialogue/proactive.py`
- Four channels, each individually configurable with **quiet hours**: **PWA push** (VAPID/web-push; the service worker already exists — add the push handler), **in-app greeting**, **scheduled check-in**, **written daily/weekly digest**.
- Consults `never_initiate` before every outreach. Serves reflection, never engagement (no streaks/guilt nudges, §4). In-container scheduler for check-ins/digest. *(Works over Tailscale HTTPS today — not VPS-gated.)*

**W6 — Research KB v1** `knowledge/research.py`
- Curated, user-owned ingest (drop in vetted papers/articles) → local chunk + embed (local embedding model; nothing cloud).
- **Silent grounding** (default): retrieved material shapes technique/framing invisibly. **Psychoeducation on demand**: answers from the literature **with source attribution**.
- *Stretch:* match confirmed patterns → suggest reading (feeds the digest).

**W7 — Review-UI sovereignty** (extends the Phase-2 review UI)
- Graph browsable as graph **and** as lists; nodes/edges editable and deletable; boundaries editable; pending-insights inbox surfaced. "The user's model of themselves — the assistant is its curator, not its owner."

**W8 — Crisis-resource config** (small, cross-cutting safety)
- Make the §4 crisis protocol's local hotlines/contacts **configurable** (protocol logic already exists in policy v1).

## Sequencing (dependency-ordered)

1. **W1** property graph + v1→graph migration — the foundation; everything else depends on it.
2. **W2** distillation + graduation — fills the graph.
3. **W3** context assembly **∥ W7** review UI — graph feeds conversation and becomes inspectable.
4. **W4** longitudinal insight — needs the graph plus several sessions of accumulated data.
5. **W5** proactivity **∥ W6** research KB **∥ W8** crisis config — build on the graph; largely independent, parallelizable.

## Invariants to hold (from SPEC + project memory)

- **Dependency/framework boundary**: all `knowledge/` code stays framework-free; only `agent.py` imports Pipecat. TheraPy → ser, never the reverse.
- **Local-first privacy (§8)**: graph, inbox, and corpus are all local; the cloud LLM receives distilled context + current conversation, never raw history or audio; `never_store` is bulletproof before any inbox write.
- **Emotion-source-agnostic**: the graph accepts `source = ser` later with no schema change.
- **No engagement mechanics (§4)**; proactivity gated by quiet hours + `never_initiate`.
- **Isolated test data** (`THERAPY_DATA_DIR`); acceptance/E2E never touch real `/data` — the Hardening 7–9 lesson.

## Acceptance criteria

- Property graph replaces flat facts; v1 facts migrated as observations; graph browse/edit/delete works in the PWA.
- A claim reaches `confirmed` **only** via explicit user validation; the graduation floor blocks single-session minting.
- Context assembly demonstrably uses the graph walk and respects `never_initiate`.
- A scripted longitudinal-loop verification
  (`scripts/verify_longitudinal_loop.py`, exposed as
  `make verify-longitudinal-loop`): seed multi-session data → a pattern
  graduates → it's reflected back → user confirmation flips status.
- ≥1 proactive channel fires within boundaries + quiet hours, gated by `never_initiate`.
- Research KB: a vetted doc ingested; silent grounding shifts technique choice; a psychoeducation answer cites its source.
- **North-star (text-only)**: one true, non-obvious self-insight surfaced and owner-confirmed.

## Risks

- **Pattern over-minting from sparse data** → mechanical floor (≥3/≥2) + LLM judgment + user confirmation.
- **Graph-walk relevance scoring is unspecified** (SPEC §10 leaves it to implementation) → ship a simple first cut, iterate against lived use.
- **Distillation quality on the local dev LLM** (gemma3:4b is weak at structured extraction) → schema-constrained output + validate on the target provider.
- **Pseudo-clinical drift (R3)** → observations phrased as data, not diagnosis; enforced in policy.
- **Privacy regressions** → `never_store` enforced at the single write path; tombstones prevent re-learning.

## Scoping decisions

The VPS-prerequisite safety items — encryption at rest, real PWA auth, `hostwatch` launchd — travel *with* Phase 5's VPS migration, so they are out. The only safety item kept in scope is the small, non-VPS crisis-resource config. Existing v1 facts are **migrated** into the graph (as observations) rather than discarded.
