# TheraPy — Project Specification

**Status:** v0.9 — phases 0–2 complete and human-accepted (phase-1 acceptance passed 2026-07-12); companion UX phases A–C shipped; Phase 4 has a partial engineering slice landed
**Owner:** Juan Pedro Sugg
**Date:** 2026-07-15
**Supersedes:** the 2024 `TheraPy` prototype (to be archived; no code carried over)

---

## 1. Vision

A voice-first personal assistant whose purpose is **self-understanding**: helping its user get to know themselves — emotional patterns, thought patterns, energy and routine patterns — through ongoing, emotionally aware conversation.

It draws on three practice traditions without claiming to be any of them:

- **CBT** — Socratic questioning, naming cognitive distortions, reframing, thought records.
- **Occupational therapy** — routines, transitions, sensory/energy regulation, daily structure.
- **Coaching** — goals, accountability, reflective check-ins.

The differentiator is the **longitudinal loop**: every conversation produces both a transcript and an emotion timeline (via [`jsugg/ser`](https://github.com/jsugg/ser)). Over weeks, the assistant can reflect real patterns back to the user — "you sound flattest on Mondays," "this topic consistently raises your arousal" — which no session-scoped chatbot can do.

### North star

> After a month of regular use, the user knows at least one true, non-obvious thing about themselves that they learned from the assistant's observations.

## 2. Target user & scope

- **v1 is single-user, for the owner (dogfooding).** No accounts, onboarding, or multi-tenancy.
- Designed with autistic adults in mind (explicit communication, predictability, no small-talk pressure), but generalization to other users is explicitly **out of scope for v1**.
- **Trilingual from day one: Spanish, English, Portuguese.** The user may code-switch; the assistant detects the utterance language and responds in kind.
- **Access: one PWA serving both a desktop web interface and an installable mobile interface, voice and text as equal citizens.** One conversation can mix modalities — speak, then type, then speak — and continue across devices. Voice runs over WebRTC. Desktop browsers reach it directly (localhost/tailnet); phones install it from the browser.
- Ambient/dedicated devices are **not** part of the current vision (explicitly parked; may or may not return).

## 3. Product pillars (phased, all in vision)

| # | Pillar | What it looks like |
|---|--------|--------------------|
| P1 | **Supportive companion** | Open-ended, low-pressure voice conversation that adapts to how the user sounds, not just what they say. |
| P2 | **Emotional awareness coach** | Uses the SER timeline to help the user notice and name emotional states; end-of-session reflections; longitudinal pattern reports. |
| P3 | **Social skill rehearsal** | Structured role-play (interview, small talk, difficult conversation) with feedback on content *and* delivery/prosody. |
| P4 | **Daily structure / executive aid** | Routines, transitions, decompression check-ins; OT-informed scaffolding. |

P1 is the foundation; P2 is the differentiator; P3–P4 build on both.

### Interaction model

No formal-therapy framing. The assistant is **always available**, and conversations range fluidly from a 30-second check-in to a deep reflective session — the user never has to declare which one they're starting. Session boundaries are inferred from conversation gaps and tagged by depth (check-in / conversation / deep session) so timelines and recaps stay coherent without ceremony.

The assistant is **adaptive and proactive**: it may open a conversation ("you mentioned the interview was today — how did it go?") within user-configured boundaries. All four channels are in scope, each individually configurable with quiet hours: **push notifications** (PWA push), **in-app greetings** (what it's been "thinking about" when you open the app), **scheduled check-ins** (e.g. Sunday-evening reflection), and a **written daily/weekly digest** of observed patterns and suggested topics. Proactivity serves growth and reflection, never engagement (see §4).

### The user model (knowledge layer)

The core long-term asset. The assistant continuously builds a structured, persistent model of the user — values, goals, known patterns (emotional, cognitive, energy/routine), preferences, ongoing threads — and **reasons over it**: it is distilled into every conversation's context, so the assistant always knows who it's talking to and where the user is trying to grow. Insights graduate from single observations → recurring patterns → confirmed self-knowledge (validated with the user in conversation). Fully local, inspectable, and editable by the user — it is *their* model of themselves.

## 4. Non-goals & safety boundaries

- **Not therapy, not a therapist replacement, no diagnoses.** Therapy-*informed* tool for self-knowledge.
- No clinical claims; no medication advice.
- **Crisis handling:** if conversation contains crisis language (self-harm, acute distress), the assistant stops coaching and surfaces human resources (configurable local hotlines/contacts). This exists even in the dogfood version.
- No engagement-maximizing patterns (streaks, guilt nudges). The tool serves reflection, not retention.

## 5. Architecture

Thin product app on a voice-agent framework; perception is pluggable; `ser` remains a product-agnostic library.

**The conversation is server-authoritative and modality-agnostic.** A conversation is a stream of turns; each turn carries modality metadata (voice/text) and language. STT and TTS are optional per-turn adapters: a typed turn skips STT, a voice reply is TTS-rendered text. This is what makes mid-conversation voice↔text switching and cross-device continuity native rather than bolted on. Response modality mirrors the input by default, with a user override.

```
voice in ──┬── VAD → STT (faster-whisper) ──┐
           │                                ├─→ turn ─┐
           └── SER adapter (per-turn) ──────┘         │   context ← user model + research KB
text in ────────────────────────────────────→ turn ───┼─→ dialogue policy → LLM ──┬─→ text out
                        │                             │                           └─→ TTS → voice out
                        └─→ session timeline (emotion + transcript, server storage)
```

### Components

| Component | Choice | Notes |
|-----------|--------|-------|
| Pipeline framework | **Pipecat** with `SmallWebRTCTransport` — settled by phase-0 spike ([framework-spike.md](framework-spike.md)) | P2P WebRTC from one self-contained Python process; no SFU/Redis footprint. Silero VAD, turn-taking, barge-in. Pipecat is confined to `therapy.integrations.pipecat`; the server depends on the framework-free `VoiceGateway` port. Revisit triggers are documented in the spike. |
| Clients | **Single PWA**: desktop web interface + installable mobile interface | WebRTC voice + text chat in one conversation view; PWA push for proactivity. Desktop browser and phone are the same client. No native app, no app stores. |
| STT | **faster-whisper** | Multilingual (es/en/pt native); shared model family with `ser`. |
| Emotion | **`jsugg/ser`** via adapter | Per-turn batch on the VAD-buffered utterance (ser is not realtime yet — that's fine; see §6). |
| LLM | **Cloud API (Claude), provider-swappable** | All prompts/policy behind a provider-agnostic interface; local (Ollama/vLLM) is a later option, not an MVP path. |
| TTS | **Kokoro** (stock local voice) | Fast, free, local; covers es/en/pt-br. A distinct "other" voice, deliberately not a self-clone. |
| Memory/store | **Local SQLite**: sessions + timelines relational; user model as a **property graph** (nodes + edges tables) | Graph-shaped model without graph-DB machinery; migration path stays open. Vector store only if/when retrieval demands it. |
| Research KB | **Local RAG over a curated corpus** | Papers/articles the user vets, chunked + embedded locally. See below. |
| Server | FastAPI backend serving the PWA + WebRTC | Server-authoritative conversation state. |
| Deployment | **Docker Compose from day one** | MVP runs on own machine, reached from mobile via **Tailscale**; target state is a **personal VPS** — containerization makes that a re-deploy, not a rewrite. VPS implies CPU-only or API fallbacks for heavy inference; keep model choices swappable. |

### Repo structure (new repo)

```
therapy/
├── pyproject.toml            # uv, Python 3.12+, same tooling stack as ser
├── src/therapy/
│   ├── voice/                # owned signaling DTOs/errors + VoiceGateway port
│   ├── integrations/
│   │   └── pipecat/
│   │       ├── runtime.py    # signaling, peer lifecycle, pipeline supervision
│   │       └── pipeline.py   # STT/TTS/relay processors, LLM provider factory
│   ├── perception/
│   │   ├── stt.py
│   │   └── emotion.py        # anti-corruption adapter around `ser`; owns EmotionFrame
│   ├── dialogue/
│   │   ├── policy.py         # system prompts, CBT/OT/coaching modes, guardrails, language handling
│   │   └── proactive.py      # assistant-initiated check-ins, within configured boundaries
│   ├── knowledge/
│   │   ├── user_model.py     # property-graph self-model: typed nodes + typed edges (Appendix A)
│   │   ├── distill.py        # inbox → nodes/edges promotion; graduation; context assembly
│   │   └── research.py       # curated corpus: ingest, embed, retrieve (silent grounding + psychoeducation)
│   ├── speech/tts.py         # Kokoro voice map (per-language voices)
│   ├── session/timeline.py   # merged emotion + transcript record; session-depth tagging; longitudinal queries
│   └── server/
│       ├── app.py            # FastAPI: WebRTC signaling + PWA serving; review UI from phase 2
│       └── static/           # PWA client (vanilla JS): web + mobile interface, manifest, service worker
├── tests/
└── docs/
```

**Dependency rule:** TheraPy → ser, never the reverse. ser's types stop at `perception/emotion.py`; TheraPy defines its own `EmotionFrame`.

### Research knowledge base

A second knowledge store, deliberately separate from the user model: the user model knows *the user*; the research corpus knows *the territory* (neurodiversity research, CBT/OT technique, current findings).

- **Curation is manual and user-owned:** the user drops in papers/articles they trust; nothing is auto-scraped. A personally vetted corpus is a feature — quality in this space varies wildly.
- **Two retrieval modes, both active:**
  - *Silent grounding (default):* retrieved material shapes which technique or framing the assistant picks, invisibly — no lecturing.
  - *Explicit psychoeducation:* when the user asks ("is this an autistic-burnout thing?") or when it clearly helps, the assistant answers from the literature **with source attribution**.
- **Later:** match the user model's confirmed patterns against the corpus to proactively suggest reading or conversation topics (feeds the digest channel).

### Persona & style

- **Stable identity, adaptive register.** One character — but tone, pace, energy, and directness modulate with the user's detected state (ser emotion frames + text cues): softer and slower when the user sounds low or overloaded, brisker and more energetic when they're up. The *register* adapts; the *character* never becomes someone else. Register selection is an explicit parameter in `dialogue/policy.py`, driven by perception — this is the most direct product use of ser's output.
- **Style arc: validate first, then challenge.** Lead with understanding; once something is genuinely understood (and the user model says the ground is stable), push back, question distortions, and hold the user accountable to their own stated goals. Challenge intensity is also register-gated — no confrontation when the user is dysregulated.

## 6. `ser` integration contract

- **Now (per-turn batch):** when VAD closes a user turn, the buffered utterance audio goes to ser's existing batch API; the resulting emotion classification is attached to that turn and injected into LLM context (e.g. "user's prosody: tense, fast"). Fits within turn-latency budget since utterances are seconds long.
- **Later (streaming):** when ser ships its streaming API, it slots in behind the same `EmotionFrame` iterator — frames just arrive more often. The per-turn shape defined here should inform ser's streaming API design.
- **Retroactive re-analysis:** raw utterance audio is retained (§8), so each ser version bump can re-run over the full archive and upgrade historical emotion data in place (raw layer is versioned per §6 — old and new analyses coexist).
- **Emotion representation is two-layered so ser's taxonomy can evolve freely:**
  - *Raw layer:* ser's native labels + scores are stored verbatim on every turn, tagged with the ser version/label-set id — historical data survives taxonomy changes and can be re-mapped later.
  - *Product layer:* a TheraPy-owned mapping (config, not code) renders raw labels into whatever each consumer needs: plain-language hints for the LLM context ("sounds tense, fast"), coarse valence/arousal for timeline visuals, and richer named emotions when the user wants to talk *about* emotions explicitly. New ser labels require a mapping entry, not a schema migration.
- **Cross-language caveat:** SER accuracy across es/en/pt must be validated; prosody-based emotion transfers across languages better than lexical, but most training data is English. Tracked as risk R2.

## 7. Multilinguality (es/en/pt)

- **Detection:** per-utterance language ID from Whisper, refined by word-level language ID over the transcript text (lingua-py, restricted to es/en/pt) — Whisper labels whole utterances only, which is too coarse for code-switched phrases.
- **Reply language (user-selectable):** the PWA exposes a compact language selector — **Auto · ES · EN · PT** — next to the mic/speaker toggles; the choice persists client-side (localStorage) and is re-sent on connect as a data-channel override (`reply_language`, `null` = auto), keeping conversation state server-authoritative.
  - **Auto (default):** reply in the **dominant** language of the last user phrase by word-level majority — *"Todo bien, me estoy sintiendo ok"* → Spanish reply (the lone "ok" doesn't flip it); *"Sí, todo bien… though… no… actually things have been hard lately"* → English reply. On a near-even mix or low-confidence detection, keep the language currently in use — no ping-ponging.
  - **Pinned (ES/EN/PT):** the therapist always replies — text and TTS voice — in the pinned language. The pin constrains replies only: the user may keep speaking or typing any language, and STT continues auto-detecting for transcripts and the timeline.
- **Code-switching:** mid-conversation switches are expected and supported; the session timeline records language per turn.
- **Prompts:** dialogue policy maintained once (English source), with language rendered at generation time by the LLM — no triplicate prompt files.
- **TTS:** chosen engine must produce natural es/en/pt (XTTS and Kokoro both cover these; quality per language to be evaluated — §10).

## 8. Data & privacy

- All persistent data (audio, transcripts, timelines, user model) stays **on infrastructure the user controls**: own machine for the MVP (mobile access via Tailscale — no public exposure, VPN identity doubles as auth), later a personal VPS (requires real authentication on the PWA and encryption at rest before migration).
- **Raw utterance audio is retained indefinitely** (decision, §10): it makes the timeline retroactively upgradable with each ser release and provides eval data for ser. It is the most sensitive data in the system — encrypted at rest is a hard prerequisite for the VPS migration, and audio never goes to any cloud service.
- Cloud LLM receives transcript text + emotion annotations — not raw audio. **Context depth:** the current conversation goes verbatim; everything older arrives only as distilled summaries + the structured user model. This is the accepted privacy trade-off of the "cloud, swappable" decision; the provider interface keeps a fully-local future open.
- Observability has two correlated planes: a restricted, full-fidelity interaction/evaluation journal and content-free operational logs, metrics, and traces. Once genuine data exists, the restricted plane stays on user-controlled infrastructure; only synthetic fixtures may enter managed backend comparisons. Broad telemetry never contains raw audio, conversation or model context, credentials, or SDP/ICE/network-endpoint identifiers.
- One-command export and delete of all stored personal data from day one (it's the owner's own data — make it inspectable).

## 9. Roadmap

| Phase | Deliverable | Acceptance | Status |
|-------|------------|-----------|--------|
| 0 | New repo scaffold (Docker from day one); **framework spike: Pipecat vs. LiveKit Agents**; archive old TheraPy | Spike verdict written down; stub server runs in compose | ✅ Done — verdict: Pipecat ([framework-spike.md](framework-spike.md)) |
| 1 | **Working voice+text loop** (P1): PWA (web + mobile) with WebRTC voice and text chat, mid-conversation switching, es/en/pt, barge-in; reachable from phone via Tailscale | Hold a natural 5-min mixed voice/text conversation in each language, from the phone | ✅ Done — human-accepted 2026-07-12; detailed hardening/evidence moved to [.local/working-notes/evidence/field-tests-2026-07.md](evidence/field-tests-2026-07.md). TTFA-vs-R1 under the target provider is deferred to a later tuning pass. |
| 2 | **Memory + timeline + review UI**: SQLite store, transcripts, session summaries, continuity, user-model v1; browse transcripts in the PWA | Assistant correctly references something from a previous session | ✅ Done — scripted verification green; continuity, reconnect-resume + transcript replay, and export/delete round-trip verified. |
| 3 | **ser integration** (P2 begins): per-turn emotion in context + timeline; emotion recap; review UI shows emotion alongside transcript (validates ser accuracy early) | Recap matches user's own read of the session | Not started |
| 4 | **Longitudinal insight + proactivity + research KB v1**: cross-session pattern queries, reflections; check-ins/digest channels; corpus ingest + both retrieval modes | North-star test: one true non-obvious self-insight | Partial engineering slice landed; not product-complete. See [phase4-goal.md](phase4-goal.md) and [phase4-impl-log.md](phase4-impl-log.md). |
| 5 | **Observability + evaluation foundation**: an owned, durable interaction journal for exact reconstruction and replay; correlated content-free operational telemetry across voice, web, storage, knowledge, outreach, and supervisors; dashboards, alerts, and runbooks | A versioned synthetic corpus round-trips and replays losslessly; leak tests keep restricted content and secrets out of broad telemetry; critical-path SLIs and actionable failure alerts stay within measured voice-latency and host-resource budgets | Not started |
| 6 | P3 rehearsal mode / P4 daily structure; VPS migration when uptime starts to matter | Prioritize based on lived usage | Not started |

Detailed July 2026 field-test and hardening chronology lives in [.local/working-notes/evidence/field-tests-2026-07.md](evidence/field-tests-2026-07.md).

## 10. Decisions log & open questions

**Settled (2026-07-09):**
- Audience: single-user dogfood first. LLM: cloud API (Claude/GPT/OpenAI-Compatible), provider-swappable. Languages: es/en/pt from day one.
- TTS: stock local voice (Kokoro) — deliberately not a self-clone.
- Interaction: no formal sessions; fluid check-ins ↔ deep conversations, always available, proactive within boundaries (§3).
- Emotion vocabulary: two-layer representation — raw ser labels stored verbatim, config-driven product mapping per consumer (§6).
- Review UI: early, phase 2–3, partly to validate ser accuracy.
- Clients: single PWA serving both the desktop web interface and the installable mobile interface; voice (WebRTC) and text are equal, switchable mid-conversation; conversation state is server-authoritative.
- Pipeline framework (settled 2026-07-09, phase-0 spike): **Pipecat with `SmallWebRTCTransport`** over LiveKit Agents — one self-contained Python process, no SFU/Redis; revisit triggers in [framework-spike.md](framework-spike.md).
- Hosting: own machine + Tailscale for MVP; personal VPS is the target state. Docker Compose from day one to make migration cheap. Ambient devices explicitly parked.
- Proactivity channels: all four (push, in-app, scheduled check-ins, digest), individually configurable with quiet hours.
- Research KB: curated local RAG, both modes — silent grounding by default, source-attributed psychoeducation on demand.

- Persona: stable identity, adaptive register (ser-driven); style arc = validate first, then challenge, with challenge intensity register-gated (§5).
- Cloud LLM context: current conversation verbatim + distilled past (summaries + user model) — never raw history (§8).
- Naming: **keep TheraPy** — archive the old repo, recreate fresh under the same name.

**Settled (2026-07-10):**
- Whisper model stays `small` on this host: measured `large-v3-turbo` (CPU int8, Intel Mac) — warm TTFA went from 9–13 s to 34–124 s in the dry run for marginal transcript gains. Meaningful STT quality improvement is gated on the VPS/GPU migration; `THERAPY_WHISPER_MODEL` stays the escape hatch.
- `scripts/hostwatch.py` exists as a host-side supervisor, but launchd wiring remains deferred until VPS migration; the VPS target removes the OrbStack-specific failure mode.

**Settled (2026-07-11):**
- Test layers: fast `pytest` (unit + API via TestClient) is the default gate; the headless `aiortc` phase scripts cover the server pipeline; a Playwright + headless-Chromium suite (`-m e2e`, opt-in) covers the real browser surface (installability, connect/transcript/resume) the others structurally cannot. Browser E2E runs against an isolated server instance so it never mutates real data — a lesson from field-test verification runs contaminating the owner's store (Hardening 7–9).
- Reply audio is the speaker toggle's job, not the input modality's: a typed turn getting an audio reply is fine (owner). Modality still mirrors by default (typed → silent, "no wasted synthesis"), but the dry run gates only on the typed turn's **text** reply arriving, not on silence.

- Reply language is user-selectable (Auto · ES · EN · PT) with **Auto = dominant language of the last phrase**, word-level majority via **lingua-py** (new dependency, phase-1-gated in pyproject); ties/low confidence keep the current language; a pin constrains replies only (STT stays auto); selection persists client-side (§7).

- Raw utterance audio: **kept indefinitely** (encrypted at rest before any VPS move). Each ser upgrade re-analyzes history — the emotional timeline gets retroactively smarter, and the archive doubles as eval data for ser itself.

- User model: **property graph** (nodes + edges in SQLite) with extensible type registries; node types include strengths, strategies (OT toolkit), people, CBT thought records; edges carry the claim lifecycle too; optional freeform observation inbox feeding distillation; canonical-English statements with original-language verbatim quotes; two-tier boundaries (`never_store` / `never_initiate`); counts instead of confidence floats; per-type decay. Full schema: Appendix A.

**Settled (2026-07-15):**
- Observability uses two correlated planes: a restricted full-fidelity interaction/evaluation journal, owned as the durable source of truth, and content-free operational telemetry. It measures safety, quality, and reliability — never engagement. Interaction-backend selection is gated by the same versioned synthetic corpus; no backend may weaken journal reconstruction, replay, or export.
- Before genuine personal data enters the observability system, re-run the threat model and approve storage, access, retention, export, and recovery controls under §8.

**Open:** none at the strategic level. Remaining refinement (exact table DDL, type-registry format, graph-walk relevance scoring) happens against implementation in phases 2–4.

## 11. Risks

| # | Risk | Mitigation |
|---|------|-----------|
| R1 | Latency of STT + LLM + TTS chain breaks conversational feel | Pipecat streaming end-to-end; measure time-to-first-audio from phase 1 |
| R2 | SER accuracy degrades on es/pt speech | Validate early with own recordings; treat emotion as a *hint* in prompts, never an assertion |
| R3 | Insight features drift into pseudo-clinical territory | Non-goals §4 enforced in policy layer; observations always phrased as data, not diagnosis |
| R4 | Scope: four pillars + three languages is a lot | Phases gate everything; P1 in three languages is the only phase-1 commitment |

---

## Appendix A — User model schema (v2: property graph)

**Paradigm (settled):** a **property graph**, persisted as nodes + edges tables in SQLite — full graph semantics without graph-DB operational weight at single-user scale; the shape keeps a real graph DB as a drop-in migration if ever needed. Plus an optional **freeform observation inbox** feeding the graph.

### Node types

Extensible by design: types live in a registry (config + UI rendering rule), so adding one is not a schema migration.

```yaml
identity_fact:   # slow-changing: languages, neurotype self-description, life context
value:           # what matters and why the user says it matters
goal:            # active | paused | achieved; horizon: week | quarter | life
pattern:         # kind: emotional | cognitive | energy | relational
preference:      # communication (directness tolerance), sensory, proactivity config, register defaults
thread:          # ongoing life situations ("the interview", "conflict with X")
person:          # minimal registry: alias, relationship — about *the user's relationships*, not dossiers on others
strength:        # what works, what the user is good at, wins — the anti-deficit ledger
strategy:        # coping/regulation/transition strategies, with evidence they work *for this user* (OT toolkit)
thought_record:  # CBT artifact: situation → automatic thought → emotion → evidence for/against → reframe
boundary:        # tier: never_store | never_initiate (see below)
```

### Edge types

Typed, directed relations; same extensible registry. Starter catalog:

```
involves        thread → person
triggers        thread | situation → pattern
soothes         strategy → pattern
works_for / failed_for   strategy → pattern | situation
supports / conflicts_with   pattern | value ↔ goal
instance_of     thought_record → pattern
about           inbox observation → any node
```

**Edges are claims too.** "Deadline pressure *triggers* catastrophizing" is an assertion that needs evidence — so edges carry the same lifecycle fields as nodes and must graduate the same way. This is the graph's core payoff: the *relationships* between patterns, goals, and situations become first-class, evidenced knowledge.

### Claim lifecycle (applies to nodes AND edges)

```
statement        canonical English (cross-language pattern matching, one coherent model)
quotes[]         user's verbatim words as evidence — original language, language-tagged
n_occurrences, n_sessions    counts, not a confidence float — honest and auditable
status           observation | pattern | confirmed
source           conversation | ser | user-stated (user-stated starts confirmed)
first_seen, last_seen, user_edited
```

- **Graduation — mechanical floor + judgment:** ≥3 occurrences across ≥2 sessions makes a claim *eligible* (protects against one intense conversation minting "patterns"); above the floor, LLM judgment decides what's worth proposing; only **explicit user validation** confirms.
- **Confirmation surfaces — context-aware:** raised in conversation when the topic is already adjacent; otherwise queued to a **pending-insights inbox** surfaced via digest and review UI. Never derails the moment.
- **Decay — per type:** identity/values never expire (revisited only on contradiction); threads go stale fast (~2 weeks); patterns flag at ~8 weeks unreinforced → re-validate or demote; strategies re-validate on each use. The model describes who the user *is*, not who they were.
- **Deletion → tombstone**, so distillation can't re-learn what the user removed.

### Observation inbox

Freeform jots during conversation — zero schema pressure, nothing lost to categorization friction. Between sessions, `distill.py` promotes inbox content into nodes/edges (attaching quotes as evidence) and discards the rest. Deliberately optional: if it proves useless in practice, it unplugs without touching the graph.

### Boundaries (two tiers)

- **`never_store`** — checked before *any* write, including the inbox.
- **`never_initiate`** — the assistant may hold it, and the user may raise it; the assistant never brings it up unprompted (checked at context assembly and by the proactivity engine).

### Context assembly (per conversation)

Always in context: identity + preferences + the `never_initiate` list. Then: active goals and threads, and a graph walk from the current topic/state to the top-K relevant nodes **with their confirmed edges** — the assistant doesn't just know your patterns, it knows how they connect. Priority: confirmed > pattern > observation; inbox content reaches the LLM only during distillation, never live.

### User sovereignty

The whole graph is browsable (as a graph and as lists), editable, and deletable in the review UI. It is the user's model of themselves — the assistant is its curator, not its owner.
