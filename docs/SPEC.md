# TheraPy — Project Specification

**Status:** v0.7 — phases 0–2 engineering complete; phase-1 human acceptance run pending
**Owner:** Juan Pedro Sugg
**Date:** 2026-07-09
**Supersedes:** the 2024 `TheraPy` prototype (to be archived; no code carried over)

---

## 1. Vision

A voice-first personal assistant whose purpose is **self-understanding**: helping its user get to know themselves — emotional patterns, thought patterns, energy and routine patterns — through ongoing, emotionally aware conversation.

It draws on three practice traditions without claiming to be any of them:

- **CBT (TCC)** — Socratic questioning, naming cognitive distortions, reframing, thought records.
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
| Pipeline framework | **Pipecat** with `SmallWebRTCTransport` — settled by phase-0 spike ([framework-spike.md](framework-spike.md)) | P2P WebRTC from one self-contained Python process; no SFU/Redis footprint. Silero VAD, turn-taking, barge-in. Only `agent.py` imports it; revisit triggers documented in the spike. |
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
│   ├── agent.py              # pipeline assembly — only file that imports Pipecat
│   │                         #   (STT/TTS/relay processors, LLM provider factory)
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
- One-command export and delete of all stored personal data from day one (it's the owner's own data — make it inspectable).

## 9. Roadmap

| Phase | Deliverable | Acceptance | Status |
|-------|------------|-----------|--------|
| 0 | New repo scaffold (Docker from day one); **framework spike: Pipecat vs. LiveKit Agents**; archive old TheraPy | Spike verdict written down; stub server runs in compose | ✅ Done — verdict: Pipecat ([framework-spike.md](framework-spike.md)) |
| 1 | **Working voice+text loop** (P1): PWA (web + mobile) with WebRTC voice and text chat, mid-conversation switching, es/en/pt, barge-in; reachable from phone via Tailscale | Hold a natural 5-min mixed voice/text conversation in each language, from the phone | ⏳ Engineering done incl. §7 reply-language selector; dry run green. Phone field tests + a cross-model review (2026-07-11) surfaced several defects (Hardening 7–10): resumed transcript not rendering, Spanish answered in English, a stale "Resume" button on an empty probe session, no install prompt (root cause was the service worker not taking control — `skipWaiting`/`clients.claim` — not the icons), and three render-race findings — all root-caused, fixed, regression-tested (incl. a browser E2E asserting Chrome's `getInstallabilityErrors == []` verdict), gates re-run green. Human acceptance run (clean, owner-only) still pending; on-device install confirmation and TTFA-vs-R1 to be recorded from it |
| 2 | **Memory + timeline + review UI**: SQLite store, transcripts, session summaries, continuity, user-model v1; browse transcripts in the PWA | Assistant correctly references something from a previous session | ✅ Done — scripted acceptance green (2026-07-10, re-verified 2026-07-11 post-fix): fact stated in session A recalled in session B; reconnect-resume + transcript replay; export/delete round-trip verified |
| 3 | **ser integration** (P2 begins): per-turn emotion in context + timeline; emotion recap; review UI shows emotion alongside transcript (validates ser accuracy early) | Recap matches user's own read of the session | Not started |
| 4 | **Longitudinal insight + proactivity + research KB v1**: cross-session pattern queries, reflections; check-ins/digest channels; corpus ingest + both retrieval modes | North-star test: one true non-obvious self-insight | Not started |
| 5 | P3 rehearsal mode / P4 daily structure; VPS migration when uptime starts to matter | Prioritize based on lived usage | Not started |

**Phase 1 — implemented so far** (commit `abff78b`): Pipecat pipeline behind `SmallWebRTCTransport` (Silero VAD, barge-in, per-turn TTFA logging); faster-whisper STT subclassed for per-utterance es/en/pt auto-detection; Kokoro TTS re-voiced per turn to the detected language; provider-agnostic LLM factory (Claude default; OpenRouter/Ollama for dev); typed turns over the WebRTC data channel into the same conversation context; vanilla-JS PWA (installable, offline shell) with mic/speaker toggles and modality mirroring; dialogue policy v1 (persona, register, validate-then-challenge, crisis protocol).

**Phase 1 — hardening (2026-07-10):** server-side reply-modality mirroring landed (typed turns skip TTS synthesis via `LLMConfigureOutputFrame`; the client's speaker toggle sends a `voice_replies` override); fixed the container image (opencv system libs — the phase-1 image had never booted); fixed VAD wiring (Pipecat ≥1.x ignores `TransportParams.vad_analyzer` on non-Daily transports — speech was never detected until `VADProcessor` became an explicit pipeline stage); added `scripts/phase1_dryrun.py`, a scripted WebRTC client that runs the trilingual voice+text conversation against the live server and measures TTFA without a microphone. Fully-local LLM verified: `THERAPY_LLM=ollama` runs `gemma3:4b` on the host (container reaches it via `host.docker.internal`); a per-switch system note (`language_switch_note`, dialogue/policy.py) now reminds the model of the user's current language — small local models otherwise keep replying in the conversation's dominant language after a switch.

**Phase 1 — dry-run evidence (2026-07-10):** `scripts/phase1_dryrun.py` passes against the live container: on a single WebRTC connection, spoken turns in es, en, and pt were each language-detected correctly and answered with the matching Kokoro voice; a typed turn got a text-only reply (no TTS audio reached the client — server-side mirroring verified); the next voice turn spoke again, and barge-in stopped the reply audio. TTFA (user stops speaking → first reply audio) across two clean runs: 3.3–15.5 s server-side, 8.3–22.6 s client-side including the 0.7 s VAD stop window; Kokoro's own synthesis TTFA is 1.3–1.9 s. The spread is dominated by the free-tier dev LLM (Kokoro's own synthesis TTFA is 0.5–2 s; whisper adds ~2–3 s on this CPU), so R1 must be re-measured with the target provider during the acceptance run. Known dev-provider quirk: the free OpenRouter meta-router sometimes ignores the reply-language instruction and occasionally emits junk replies ("User Safety: safe") — language detection itself was correct. With the fully-local `gemma3:4b` via Ollama the same dry run passes with correct reply language in all three turns (after the per-switch language note) and coherent, on-persona replies; TTFA is 12.7–20.7 s server-side, since the LLM shares the CPU with whisper and Kokoro.

**Phase 1 — reply-language selector (2026-07-10):** implemented per §7. `dialogue/language_choice.py` (framework-free) chooses the reply language: auto mode takes the word-level dominant language of the last phrase (lingua over the transcript, summed across code-switch sections; ties and undetectable phrases keep the current language), and a pin (es/en/pt) constrains replies and TTS voice only — STT keeps auto-detecting for transcripts. The PWA selector (Auto · ES · EN · PT) persists in localStorage and is replayed on connect as a `reply_language` data-channel override. Unit tests cover the two normative §7 phrases; the dry run exercises both normative phrases spoken, plus pin/unpin, end-to-end.

**Phase 1 — dry-run evidence (2026-07-10, final image, fully-local gemma3:4b):** all ten scenarios green on one connection — es/en/pt spoken turns each detected and answered in-language; typed turn silent (modality mirroring); barge-in stopped reply audio; both §7 normative code-switched phrases spoken and choosing correctly (minority "ok" kept Spanish; mid-phrase dominance flip switched to English); pin to pt answered a Spanish utterance in Portuguese voice+text with the transcript still tagged es (STT stays auto), and unpinning restored auto. Client-side TTFA 7.9–50.5 s (first turn includes cold model loading; all-CPU host sharing whisper, Kokoro, and the LLM).

**Phase 2 — acceptance evidence (2026-07-10, gemma3:4b local):** `scripts/phase2_acceptance.py` green end-to-end — session A stated a distinctive fact (a dog named Nebulosa) over a real WebRTC connection and disconnected; the background summarizer captured the fact; session B, a fresh connection, asked and the assistant answered "Se llama Nebulosa"; `python -m therapy.memory export` contained the fact; `delete` without `--yes` refused (exit 2) and `delete --yes` left zero sessions behind (verified via a fresh export and `/api/sessions`).

**Phase 2 — implemented (2026-07-10):** `memory/` package (framework-free): SQLite store under `THERAPY_DATA_DIR` — sessions, per-turn transcripts (language, modality, timestamps), pointers to archived raw utterance WAVs (written where STT already holds the buffered utterance; audio never leaves the host); session summaries + user-model v1 facts distilled at disconnect via the provider-agnostic summarizer (`THERAPY_LLM` convention); continuity injected at connect as distilled summaries + facts, never verbatim history (§8). `python -m therapy.memory export | delete --yes` is the one-command data round-trip. Review UI: session list + read-only transcript browser in the PWA (`/api/sessions`). `scripts/phase2_acceptance.py` automates the acceptance criterion (fact stated in session A, recalled in session B) plus the export/delete round-trip.

**Hardening (2026-07-10, evening):** the phone's "connecting → disconnected" failure was diagnosed to unreachable ICE candidates — the containerized pipeline advertises only container-internal IPs, and container→tailnet UDP is not guaranteed. Fixed with a **TURN relay** (`turn` compose service, coturn): the PWA fetches `/api/ice-config` and allocates a relay at `turn:<page-host>:3478`, which the pipeline reaches directly on the compose network; `scripts/netcheck.py --relay-only` regression-checks the path by offering relay candidates only. Reliability: `restart: unless-stopped` on both services, a compose healthcheck, an in-container **watchdog** (PID 1) that restarts uvicorn when the event loop hangs (a wedged loop passes process checks but fails HTTP), and **connection preemption** — a new WebRTC connection cancels the previous pipeline (each loads its own STT/TTS models; stacking them was the observed OOM cause), with the preempted session still summarized. UI concept direction (companion avatar, swappable skins) specced in [ux-companion-spec.md](ux-companion-spec.md).

**Hardening 2 (2026-07-10, night):** the Docker VM (OrbStack) wedged twice at the hypervisor level under memory pressure — every docker CLI call and port-forward hangs, which no in-container mechanism can heal. Mitigations: per-service `mem_limit` caps (a container OOM-kill + restart policy replaces VM exhaustion), `scripts/hostwatch.py` — a host-side supervisor with two-layer escalation (container restart while the daemon answers; provider restart + `compose up -d` when the daemon itself is wedged), and a service-worker fetch timeout so the PWA degrades to its cached shell instead of loading forever against a hung server. Regression tests cover the compose caps, the escalation logic, and both watchdogs.

**Hardening 3 (2026-07-10, first phone field test):** the phone connected via the TURN relay and surfaced four defects, all fixed. (1) *Reply-language drift:* the local model answered in English while the tag and TTS voice were correctly es/pt — the per-switch note was not enough; every user turn now carries a short reply-language anchor (switch note, pin note, or one-line reminder; deduped across fragments of one aggregated turn). (2) *STT losing most speech:* pipecat's VAD defaults (`min_volume .6`, `confidence .7`) gate quiet phone audio; now env-tunable and defaulted to `.3`/`.6`, with `stop_secs` 0.7→1.0 so natural mid-sentence pauses don't split an utterance. (3) *Quoted fragments flipping the language:* a phrase must have ≥4 detected words to switch the reply language — "muriendo de sueño" quoted inside an English sentence stays a quote (SPEC §7 ties/low-confidence rule, made concrete). (4) *Slow, mute connect:* the Start button yields instantly and returns on failure; ICE gathering is bounded at 3 s (browsers stall on the TURN tcp variant); the whisper model is loaded once per process instead of per connection (each copy cost ~1 GB and tens of seconds that blocked handshakes racing the load).

**Hardening 4 (2026-07-10, second phone field test):** a long English utterance over degraded audio made whisper hallucinate — one chunk decoded as repeated Korean stock phrases ("뵐게요. 뵐게요.") tagged `en`, corrupting the turn. Defenses in `run_stt`: segments failing whisper's own plausibility thresholds (`compression_ratio > 2.4`, `avg_logprob < -1.2`, `no_speech_prob > 0.6` — `perception.stt.plausible_segment`) are dropped; a detected language outside es/en/pt is treated as a hallucination signature and the audio is re-decoded anchored to the conversation's current language (recovering the content instead of discarding it); `condition_on_previous_text=False` stops one bad decode from poisoning the next. VAD `stop_secs` 1.0→1.2 for thinking pauses.

**Hardening 5 (2026-07-10, third phone field test):** (1) *Reconnect amnesia:* a 4-minute WebRTC drop finalized the session, and the reconnect opened a fresh one with an empty context — the bot denied the entire prior conversation. Now a connection arriving within `THERAPY_RESUME_WINDOW_SECS` (default 900) of the newest session's last activity **resumes** it: the session row is reopened (`MemoryStore.resume_candidate`/`reopen_session`), its turns re-enter the LLM context verbatim behind a reconnect marker (`policy.resume_note`/`rehydrate_messages` — within one session verbatim is correct; the summaries-only rule is cross-session), and STT/reply-language state restarts from the last user turn's language. Finalization is reconnect-safe: an in-flight finalize is cancelled by the resuming pipeline, and one scheduled after ownership changed no-ops (owner tokens) — the session is re-summarized in full when it finally ends. Test scripts pass `/api/offer?new_session=1` to keep scenarios isolated; the acceptance run now includes a deterministic resume scenario (same session id, no new row). (2) *History-view layout:* the author rule `main{display:flex}` overrides the UA's `[hidden]{display:none}`, so the "hidden" empty chat kept flexing and squeezed the transcript browser into half the viewport, clipping bubbles mid-line — fixed with `[hidden]{display:none !important}` (+ shell cache bump to v3), regression-tested in `tests/test_static.py`.

**Session management (2026-07-10):** the server is now the source of truth for the live chat view — on every connect the client sends `client_ready` and receives a `session` payload (`server/protocol.py`: id, resumed flag, last 40 turns) that it renders wholesale, so a reconnect or page reload shows the resumed transcript instead of an empty pane (verified in the acceptance run's replay assertion). The history browser grew management verbs: **＋ New conversation** (`/api/offer?new_session=1`), **▶ Continue** any past session explicitly (`/api/offer?session=<id>` — reopens and rehydrates it, no freshness window), and **🗑 per-session delete** (`DELETE /api/sessions/{id}`, cascading turns + archived audio; refuses a live session with 409 via the `server/live.py` ownership registry; wholesale wipe remains the deliberate CLI act). Sessions carry a **title**: auto-generated at finalization (3–6 words, in the session's dominant language, `summarizer.entitle`) and user-editable (✏️ → `PATCH /api/sessions/{id}`); auto-generation is fill-only (`ensure_title`), so a rename is never overwritten, and the list shows title + date + turn count instead of the raw summary blob.

**Hardening 6 (2026-07-10, fourth phone field test):** five fixes. (1) *Mid-turn disconnects under long speech:* faster-whisper's `transcribe()` is lazy — the decode ran where the segments generator was consumed, on the event loop; a long utterance blocked the loop (watchdog probe failure in the logs confirms), starving WebRTC keepalives until the connection dropped and swallowing any speech during the stall. The whole decode now runs inside `asyncio.to_thread` (`_transcribe_utterance`). (2) *Spanish greeting answered in English:* the ≥4-word switch hysteresis also blocked the *first* determination ("¡Hola! ¿Cómo estás?" is three words) — the cold-start default now yields to the first detectable phrase; hysteresis applies once a language is established (resumed sessions count as established). (3) *Replies tagged es but written in English (gemma):* per-turn anchors are now written in the target language — an English system note is itself English evidence to a small model; in the target language the note is both instruction and prime. (4) *German speech silently translated and tagged en:* the unsupported-language re-decode (a hallucination defense) steamrolled genuine foreign speech, making whisper translate it. Discriminator: a hallucinated decode dies in the plausibility filter (empty text); a surviving decode in an unsupported language is real speech and is now transcribed verbatim under its honest tag (`de`), without steering the reply language (out-of-scope languages stay out of scope — replies continue in the conversation's language). (5) *Start vs Resume ambiguity:* the connect button now reads "Resume conversation" when `GET /api/resumable` says connecting would resume (labeled by what the server will actually do).

**Re-verification on the shipped image (2026-07-10, post-hardening-6, gemma3:4b local):** both scripted gates re-run green on the final image — `phase1_dryrun.py` exit 0, all ten scenarios (client TTFA 9.2–32.5 s; the earlier 50 s tail was cold model loading, absent on a warm container), and `phase2_acceptance.py` exit 0 including the replay and reconnect-resume assertions. In-container `pytest` 86 green, `ruff` clean; `netcheck.py --relay-only` PASS from the host (note: it must run from a real client's network position — inside the container the TURN hostname resolves to the pipeline itself and no relay candidates gather). Data volume backed up before the acceptance wipe and restored after; dry-run test sessions pruned via `DELETE /api/sessions/{id}`.

**Hardening 7 (2026-07-11, fifth phone field test):** resuming a conversation from the phone showed an empty pane — the resumed transcript never rendered, and a spoken turn produced no visible transcript; only after the user typed did the history appear, with no reply. Root cause: the initial chat state was delivered *solely* over the WebRTC data channel (`client_ready` → `session` replay). Pipecat starts a 10 s timer when the peer connection reaches `connected` and, if the data channel has not opened by then, clears its outbound queue and disables further queueing (`smallwebrtc/connection.py`); on the phone over the TURN relay the data channel opened late (log: *"Data channel not established within 10s … disabling future queueing"*), so the replay was silently dropped and every server→client message with it. Fix: the `/api/offer` response now carries the resolved `session_id` + `resumed` flag (`server/app.py:_resolve_session`, which mirrors run_bot's resume choice and is the exact id the pipeline joins), and the client loads that transcript over HTTP (`GET /api/sessions/{id}`) on connect — reliable on the same network the offer already succeeded on. The data-channel replay is now a deduped fallback (`renderHistoryOnce`, guarded by `historyLoaded`), so a late replay can no longer wipe live transcripts. The compounding "no reply" was a slow local-LLM turn (gemma3:4b, ~20 s TTFB on the rehydrated context) cancelled by the user's next utterance (barge-in) before delivery — with the transcript now visible the prod-and-repeat loop is broken, and R1 stays a target-provider measurement. Regression tests: the `_resolve_session` resolution table (`test_review_api.py`) and the client's HTTP transcript load (`test_static.py`); shell cache bumped to v6.

**Hardening 8 (2026-07-11, sixth phone field test):** the owner spoke Spanish and got English replies — the turns were correctly tagged `es` and voiced in a Spanish voice, only the words were English (the tell that reply-language detection was right and the model itself drifted). Root cause was a real code fault, not just small-model drift: on connect the PWA replays its reply-language selector, and in **auto** mode (`null`) the handler compared auto's cold default (`en`) against the still-unset relay language and injected a `language_switch_note('en')` — *"The user is now speaking English. Reply entirely in English"* — as a system message placed **ahead of the user's first turn**. The model context captured it verbatim, a contradictory English anchor sitting above a Spanish greeting; gemma3:4b obeyed the leading instruction. Fix: auto asserts nothing on replay — only an explicit es/en/pt pin anchors the reply language (`dialogue/language_choice.reply_language_override_effect`, framework-free, unit-tested). Verified end to end: the first-turn LLM context now carries only the Spanish anchor and the es reply comes back in Spanish. The scripted dry run had passed all along because it *skipped* the connect-time selector replay the real client sends — the exact fidelity gap that let this reach the field — so `phase1_dryrun.py` now replays it on connect (the es turn returns English without the fix), closing the gap as an end-to-end regression. Gates re-run green on the rebuilt image: `pytest` 92, `ruff` clean, `phase1_dryrun.py` exit 0 (trilingual + §7 + pin/unpin, es answered in Spanish, no English anchor in the first-turn context), `phase2_acceptance.py` exit 0 (continuity, reconnect-resume + replay, export/delete). **Process fix:** a verification dry run must never run against the live container during an owner field test — v1 is single-user and a new connection preempts the previous pipeline, so the scripted client and the phone corrupted each other (a dry-run utterance surfaced in the owner's session, and resume latched onto the newer scripted session). Verification now runs only against an idle container.

**Hardening 9 (2026-07-11, fresh-install field test):** reinstalling from the phone surfaced two issues. (1) *"Resume conversation" with nothing to resume:* `MemoryStore.resume_candidate` returned the newest session within the resume window regardless of whether it held any conversation, so an empty session — a connectivity probe, a `netcheck` run, or a connect that dropped before the user spoke — put a Resume button on the landing screen with nothing behind it. It now requires the candidate to have at least one turn (`WHERE EXISTS … turns`), skipping empty ones; `/api/resumable` and the offer's `_resolve_session` inherit the guard. (2) *No install offer (Chrome/DuckDuckGo):* the web manifest declared only an SVG icon (`sizes:"any"`), which does not satisfy Chrome's installability check — it needs raster 192px and 512px icons — so no "Install app" was offered (DuckDuckGo has no PWA-install support, expected). Added 192/512 PNGs rendered from the existing mark plus a padded **maskable** 512 (so Android doesn't frame it on a white badge), kept the SVG for scalable contexts, pointed `apple-touch-icon` at the PNG, and cached the icons in the shell (v7). Regression tests: empty sessions are not resumable and a newer empty probe does not hide an older real session (`test_memory.py`); the manifest declares 192/512 PNG + a maskable icon and each file exists at its true pixel size (`test_static.py`). Gates green on the rebuilt image: `pytest` 95, `ruff` clean, `/api/resumable` returns null against the empty probe session, all icon files serve `200 image/png`.

**Browser E2E + install RCA correction (2026-07-11):** the install fix in Hardening 9 was shipped without browser-level proof — the only automated coverage was headless `aiortc` (server pipeline) and structural file checks, so nothing ever loaded the PWA in a real browser, which is how both the install issue and the stale Resume button reached the field. Added a Playwright + headless-Chromium suite (`tests/e2e/`, opt-in `-m e2e`) that runs against an isolated server (own temp data dir + port, so it never touches real `/data`) and asserts what only a browser can: (1) real installability — an active service worker, a manifest served as `application/manifest+json`, and 192/512/maskable PNG icons that decode at their true sizes; (2) the live flow — connect over WebRTC with a fake mic, a typed turn, its assistant transcript rendering, and the Start→Resume label logic. Both pass. This also **corrects the Hardening 9 RCA**: the app *does* meet Chrome's installability criteria (now proven), so the owner's "no banner" was not a broken manifest but Chrome's post-uninstall banner suppression — the app stays installable via the menu. The raster PNG icons remain the right change (SVG-only installability is version-flaky; PNGs make it unambiguous), but they were not the cause of that specific observation.

**Hardening 10 (2026-07-11, install still not offered + cross-model review):** the install fix (Hardening 9 + browser E2E) did NOT resolve the owner's phone — Chrome offered no install, menu or automatic. The first E2E gave false confidence: it asserted the structural prerequisites (active service worker, manifest, icons decode) but not Chrome's actual verdict. Queried Chromium directly via CDP `Page.getInstallabilityErrors`: **zero errors** (the app meets Chrome's criteria), but `controls_page: false` — the service worker never took control (no `skipWaiting`/`clients.claim`). On a device that has loaded many versions of this app, a stale worker stays in charge while the new one waits, which drops Chrome's install option. Fixed: the worker now `skipWaiting()`s on install and `clients.claim()`s on activate (cache v8), and the E2E now asserts `getInstallabilityErrors == []` and that the worker controls the page. If install still does not appear, the residue is Chrome's own state (a prior install/uninstall leaving stale records) — clearing the site's data in Chrome and reloading forces the new worker in. Separately, an independent cross-model review found three correctness races in the post-connect history fetch (a stale fetch rendering a preempted connection's session; live turns mis-ordered or duplicated before it resolved) and an offer/`run_bot` disagreement on an explicit-unknown session id. Root-fixed by carrying the resumed transcript in the `/api/offer` answer — rendered synchronously on connect, no fetch to race — and having `run_bot` join exactly the resolved session (`new_session` when None).

**Deferred (decided 2026-07-10):** running `scripts/hostwatch.py` as a login daemon (launchd plist) — the host-side supervisor exists and is tested; wiring it into launchd is deferred until the VPS migration decision, since the VPS replaces the OrbStack layer entirely.

**Phase 1 — outstanding:** the human acceptance run itself (5-min mixed voice/text conversation in each language, from the phone over Tailscale HTTPS); and TTFA validation against R1 under real network conditions.

## 10. Decisions log & open questions

**Settled (2026-07-09):**
- Audience: single-user dogfood first. LLM: cloud API (Claude), provider-swappable. Languages: es/en/pt from day one.
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

**Settled (2026-07-11):**
- Test layers: fast `pytest` (unit + API via TestClient) is the default gate; the headless `aiortc` phase scripts cover the server pipeline; a Playwright + headless-Chromium suite (`-m e2e`, opt-in) covers the real browser surface (installability, connect/transcript/resume) the others structurally cannot. Browser E2E runs against an isolated server instance so it never mutates real data — a lesson from field-test verification runs contaminating the owner's store (Hardening 7–9).

- Reply language is user-selectable (Auto · ES · EN · PT) with **Auto = dominant language of the last phrase**, word-level majority via **lingua-py** (new dependency, phase-1-gated in pyproject); ties/low confidence keep the current language; a pin constrains replies only (STT stays auto); selection persists client-side (§7).

- Raw utterance audio: **kept indefinitely** (encrypted at rest before any VPS move). Each ser upgrade re-analyzes history — the emotional timeline gets retroactively smarter, and the archive doubles as eval data for ser itself.

- User model: **property graph** (nodes + edges in SQLite) with extensible type registries; node types include strengths, strategies (OT toolkit), people, CBT thought records; edges carry the claim lifecycle too; optional freeform observation inbox feeding distillation; canonical-English statements with original-language verbatim quotes; two-tier boundaries (`never_store` / `never_initiate`); counts instead of confidence floats; per-type decay. Full schema: Appendix A.

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
