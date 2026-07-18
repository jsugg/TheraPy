# TheraPy Developer Quickstart

TheraPy is a local-first, voice-based, therapy-informed companion. It is two things
at once, and the second one is the part most contributors miss:

1. A **real-time voice loop** — mic → speech-to-text → LLM → text-to-speech → speaker,
   over WebRTC.
2. An **AI system with production-grade traceability** — every model interaction is
   captured as durable, replayable evidence so behavior can be reproduced and evaluated,
   not just observed.

The mental model to hold the whole time:

> **build → observe → capture evidence → replay → evaluate → ship**

Everything below serves that loop. For product scope and configuration reference see
[`../README.md`](../README.md); for the design contract and the numbered requirements
(SPEC §…) referenced throughout the code see [`SPEC.md`](SPEC.md).

---

## 0. TL;DR — running in 30 seconds

```bash
git clone https://github.com/jsugg/TheraPy && cd TheraPy
cp .env.example .env          # defaults are safe and fully local
make up                       # build + start the stack (Docker)
make status                   # wait until the therapy container is "healthy"
open http://localhost:8000    # or http://<machine-name>:8000 to install the PWA
```

To talk to a model you need **one** of:

- `ANTHROPIC_API_KEY=…` in `.env` (default provider), **or**
- a local Ollama (`THERAPY_LLM=ollama`, see [§5](#5-running-fully-local-offline)).

### The edit loop (read this once)

`src/`, `tests/`, and `scripts/` are **bind-mounted read-only** into the container, so
your edits are live — but the server process is not always reloaded automatically:

| You changed… | Do this |
| --- | --- |
| UI (JS/CSS/HTML under `src/`) | just reload the browser |
| Python | **`make restart`** — restarts uvicorn to pick up the code |
| A test | nothing — `make test` / `make e2e` read it directly |
| A dependency (`pyproject.toml`/`uv.lock`) | **`make rebuild`** |

> ⚠️ **The container is the supported environment.** `uv sync` on the host is only for
> IDE/tooling (types, autocomplete) and **fails on macOS x86_64** — some wheels have no
> Intel-mac build. Never rely on the host venv to *run* the app; use `make up`.

> ⚠️ If tests suddenly fail with `StopIteration` on *every* case, the bind-mounted
> `pyproject.toml`/`uv.lock` inode went stale after an edit. Fix: **`make restart`**.

---

## 1. The mental model (do not skip)

### Two telemetry planes

There are **two** deliberately separate telemetry paths. Contributors must know which is
which, because putting the wrong data in the wrong one is the single most damaging mistake
you can make in this codebase.

| | **Restricted interaction plane** | **Broad operations plane** |
| --- | --- | --- |
| **Purpose** | Reproduce & evaluate any AI interaction deterministically | Operate the service without sensitive content |
| **Contains** | Exact prompts, ordered messages, system instructions, memory context, retrieved passages, tool calls & outputs, provider-native requests, streamed deltas, completions, provider error bodies, transcripts | Latency, counts, durations, sizes, statuses, bounded enums, correlation IDs |
| **Store** | Application-owned SQLite **journal** (durable, replayable — the source of truth) | stdout JSON logs → OTel Collector → Grafana / metrics |
| **Export** | Optional → Phoenix (OpenInference / OTel GenAI) | OTLP → Collector |
| **Owner module** | `observability/journal.py`, `interactions.py`, `exporters.py` | `observability/logging.py`, `telemetry.py`, `metrics.py` |

> 🔒 **The rule:** never put prompts, transcripts, memories, retrieved documents, or tool
> payloads into the broad plane. Routing enforces this (`observability/routing.py`) and a
> canary gate proves it (`make obs-canary-scan`) — but it is your responsibility first.

### Current scope: synthetic-only

All defaults are safe: capture is **local**, and **nothing leaves the host**. Genuine user
data, public exposure, and remote export are *promotion triggers* that each require an
explicit new threat-model decision before the corresponding gate is flipped (see [§9](#9-privacy--security-therapy-grade)).

---

## 2. Architecture at a glance

```
                     Browser / PWA   http://localhost:8000
                            │
        WebRTC audio  +  DataChannel (events / live transcript)
                            │  ⇄  coturn TURN relay  (for phone-over-Tailscale)
                            ▼
                     FastAPI server            src/therapy/server
             ┌──────────────┼───────────────────────────────┐
             ▼              ▼                                ▼
      Pipecat pipeline   Memory + knowledge           Observability
      (live voice loop)  subsystem                    (two planes)
             │              │                                │
   Multilingual Whisper     SQLite sessions & data     Restricted plane
   STT (faster-whisper,     summarizer.complete()   →    → SQLite journal (truth)
   per-utterance ES/EN/PT)   (non-realtime LLM)          → optional export → Phoenix
        │                    retrieval / context
   realtime LLM              assembly                  Broad plane
   make_llm_service()                                    → stdout JSON
        │                                                → OTel Collector → Grafana
   Kokoro TTS  →  audio back to the browser
```

Two distinct LLM call shapes exist, and the codebase names them explicitly
(`observability/model.py::ProviderPath`):

- **Realtime** (`PIPECAT_LLM_SERVICE`) — the streaming voice reply, built by
  `integrations/pipecat/pipeline.py::make_llm_service`. Evidence includes stream/tool deltas.
- **Non-realtime** (`COMPLETION_CLIENT`) — single-shot completions for memory & knowledge
  (`memory.summarizer.complete`, `knowledge.distill`, …).

Every audited call site is registered in `LLM_BOUNDARY_MANIFEST` with the evidence it must
capture and its failure policy (capture failures are `FAIL_OPEN_WITH_GAP` — they never
break the user, they record a gap).

### Provider / component matrix

| Layer | Component | Options (default) | Swap point |
| --- | --- | --- | --- |
| Voice in | STT | faster-whisper, multilingual ES/EN/PT (`THERAPY_WHISPER_MODEL=small`) | `perception/stt.py` |
| Brain — realtime | streaming LLM | `anthropic` \| `openrouter` \| `ollama` (`THERAPY_LLM=anthropic`) | `integrations/pipecat/pipeline.py::make_llm_service` |
| Brain — non-realtime | completion LLM | same providers | `memory/summarizer.py`, `knowledge/distill.py` |
| Voice out | TTS | Kokoro (`THERAPY_VOICE_{EN,ES,PT}`) | `speech/tts.py` |
| Memory | storage | SQLite | `memory/store.py` |
| Retrieval | embeddings / context | local pipeline | `knowledge/` |
| Transport | WebRTC + TURN | coturn (`turn` compose service) | `compose.yaml` |
| Restricted telemetry | journal / export | SQLite journal → Phoenix | `observability/journal.py`, `exporters.py` |
| Broad telemetry | logs / metrics / traces | stdout JSON → Collector → Grafana | `observability/{logging,telemetry,metrics}.py` |

`THERAPY_LLM_MODEL` overrides the model for whichever provider is selected; the per-provider
defaults live in `make_llm_service` (the source of truth — don't hard-code model IDs elsewhere).

---

## 3. First-time setup

**Required**

- Docker (the runtime and the test bed)
- Python via `uv` (host tooling / IDE only — see the caveat in [§0](#0-tldr--running-in-30-seconds))
- A provider API key **or** a local Ollama
- A browser with microphone permission

**Optional (observability stacks)**

- Phoenix (restricted-plane trace inspection)
- Grafana + OTel Collector (broad-plane dashboards)

```bash
git clone https://github.com/jsugg/TheraPy && cd TheraPy
cp .env.example .env
# edit .env: set ANTHROPIC_API_KEY, or configure Ollama (see §5)
make up
make logs        # follow startup; Ctrl-C to detach (the stack keeps running)
```

Then open `http://localhost:8000` on the host, or `http://<machine-name>:8000` from another
device on the LAN/tailnet to install the PWA. `.env.example` is thoroughly commented — it is
the canonical reference for every knob (speech, VAD, sessions, crisis contacts, OCR, push, TURN).

---

## 4. Everyday commands

Everything routes through `make` (run `make help` for the live list). Pass extra pytest flags
with `ARGS`, e.g. `make test ARGS="-k memory -x"`.

**Container lifecycle**

| Command | Does |
| --- | --- |
| `make up` | build if needed + (re)start the stack in the background |
| `make restart` | restart the server to pick up **Python** edits (no rebuild) |
| `make rebuild` | clean image rebuild (use when **dependencies** change) |
| `make down` | stop the stack |
| `make status` | container status + health |
| `make logs` | follow server logs |
| `make shell` | interactive shell in the running container |

**Tests & quality**

| Command | Does |
| --- | --- |
| `make test` | unit + integration in the container (the real test bed) |
| `make test-unit` / `make test-integration` | just one suite |
| `make e2e` | all browser e2e (auto-installs Chromium + Firefox) |
| `make lint` | Ruff (host) |
| `make typecheck` | Pyright in the container (the supported runtime) |
| `make coverage` | full suite + coverage report + `COV_MIN` fail-under gate (default 80) |
| `make check` | **lint + typecheck + coverage** — the pre-push gate (see [§11](#11-contributor-checklist-runnable)) |
| `make hooks` | install the repo git hooks (`.githooks`) into your clone |

> There is **no CI**. `make check` run locally *is* the gate. Install the hooks
> (`make hooks`) so `check` runs on push automatically (bypass any hook with `--no-verify`).

**Observability gates**

| Command | Does |
| --- | --- |
| `make obs-canary-scan` | routing/secret canary gate over the fixture corpus (the leak check) |
| `make obs-fixture-hash` | reproducible identity of the observability fixture corpus |
| `make obs-fixtures` | regenerate golden interaction fixtures + canaries |
| `make obs-baseline` | telemetry off/on workload baseline against a running instance |
| `make obs-dashboards` | regenerate the six Grafana dashboards deterministically |

---

## 5. Running fully local (offline)

Ollama runs on the **host**; the server runs in the container and reaches it via
`host.docker.internal`:

```bash
# host
ollama serve
ollama pull pedrolucas/smollm3:3b-q4_k_m     # default local model — CPU-friendly es/en/pt

# .env
THERAPY_LLM=ollama
OLLAMA_BASE_URL=http://host.docker.internal:11434/v1
```

What is and isn't offline:

- **STT (faster-whisper) and TTS (Kokoro) are already local** and deterministic given the
  same model + audio — no network.
- **The LLM** is the only component that needs the network *unless* you use Ollama, which
  makes the whole loop offline.
- Both the **realtime** path (`make_llm_service`) and the **non-realtime** path
  (`summarizer.complete`) honor `THERAPY_LLM`, so switching provider switches both.

> ⚠️ **STT performance trap:** `THERAPY_WHISPER_MODEL` defaults to `small` for a reason.
> On CPU (no GPU), `large-v3-turbo` measured warm time-to-first-audio jumping from ~9–13s to
> 34–124s. Don't bump the model for "quality" on a CPU host — real STT gains are gated on the
> GPU/VPS migration.

---

## 6. Observability in practice — turning the planes on

Both planes are **off/local by default**. You opt in per plane.

### Restricted plane (interaction journal → optional Phoenix)

| Var | Default | Meaning |
| --- | --- | --- |
| `THERAPY_CAPTURE_MODE` | `runtime` | `disabled` \| `runtime` \| `evaluation` |
| `THERAPY_INTERACTION_BACKEND` | `journal` | `journal` \| `phoenix` |
| `THERAPY_INTERACTION_JOURNAL` | `$THERAPY_DATA_DIR/interaction-journal.sqlite3` | journal file path |
| `THERAPY_INTERACTION_RETENTION_DAYS` | `30` | unacknowledged records never expire |
| `THERAPY_INTERACTION_REMOTE_EXPORT` | `0` | **egress gate** — must be explicitly flipped |
| `THERAPY_OTLP_RESTRICTED_ENDPOINT` | — | e.g. `http://phoenix:6006` with the Phoenix profile |

```bash
docker compose --profile llm-observability up -d phoenix   # http://127.0.0.1:6006
```

### Broad plane (logs / metrics / traces → Grafana)

| Var | Default | Meaning |
| --- | --- | --- |
| `THERAPY_OTEL_ENABLED` | `0` | broad traces/metrics export |
| `THERAPY_OTLP_BROAD_ENDPOINT` | `http://localhost:4318` | Collector OTLP endpoint |
| `THERAPY_CLIENT_TELEMETRY` | `0` | browser-side telemetry (a third, front-end source) |
| `THERAPY_LOG_LEVEL` | `INFO` | |
| `THERAPY_ENVIRONMENT` | `development` | `development` \| `test` \| `dogfood` \| `vps-test` |

```bash
docker compose --profile observability up -d               # Collector + Grafana
# Grafana → http://127.0.0.1:3000
```

> Both UIs bind to `127.0.0.1` **only** — loopback by design; OTLP and scrape traffic stay
> internal. There are two profiles, and they map onto the two planes:
> `llm-observability` = Phoenix (restricted), `observability` = Collector + Grafana (broad).

---

## 7. Code ownership map

The observability package is intentionally **framework-free**. Learn the boundaries before
you touch it (each module's docstring cites the plan section it implements):

```
src/therapy/observability/
  config.py        Strict, frozen configuration (env → typed config)
  model.py         Vendor-neutral contracts (ProviderPath, LLM_BOUNDARY_MANIFEST, event kinds)
  context.py       Trace / interaction correlation context
  routing.py       Plane classification, denylist, canary scanning
  logging.py       Broad-plane JSON logging + third-party logger policy
  telemetry.py     Owned OTel bootstrap — the ONLY module importing the OTel SDK
  metrics.py       Logical instrument manifest with bounded attribute sets
  interactions.py  Canonical interaction record: frozen, typed, exactly serialized
  journal.py       Dedicated SQLite interaction journal (the evidence store)
  exporters.py     Backend-neutral interaction export
  capture.py       Interaction capture service + failure policy
  health.py        Component health snapshots + the readiness model
  replay.py        Deterministic, network-free replay of restricted journals
```

```
src/therapy/integrations/pipecat/
  observability.py  Pipecat-only telemetry adapter — the ONLY Pipecat-aware observability code
```

> 🚧 **Architectural boundary:** no observability *logic* belongs inside Pipecat-specific
> code. Pipecat is one pipeline framework among possible others; keep the contracts in
> `observability/` vendor-neutral and let `integrations/pipecat/observability.py` be a thin
> adapter. `telemetry.py` is the only place allowed to import the OTel SDK.

---

## 8. Evaluation & replay workflow

The differentiator. A change is not "done" because it runs — it's done when its captured
evidence still replays and evaluates cleanly.

```
edit code → run fixtures → capture journal → replay deterministically
          → compare metrics / evaluations → approve or investigate → ship
```

### Fixture corpus

```
tests/fixtures/observability/
  interactions/   provider requests, streams, tool calls, failures
  speech/         ES / EN / PT, silence, code-switching
  research/       documents, OCR, embeddings, retrieval
  behavior/       safety, policy, grounding
  pipecat/        pipeline-adapter fixtures
```

Regenerate goldens with `make obs-fixtures`; prove the corpus is byte-stable with
`make obs-fixture-hash`.

### Replay one captured interaction

`replay.py` is a library; the CLI is `scripts/observability/replay_interaction.py`. It
reconstructs and **verifies one interaction** (it requires the interaction id, not just the
journal):

```bash
docker compose exec therapy uv run --no-dev python \
  scripts/observability/replay_interaction.py \
  --journal /data/interaction-journal.sqlite3 \
  --interaction-id <id> [--json]
```

Exit code: `0` verified · `1` not verified · `2` error. Replay is **network-free,
provider-free, and deterministic** — that's what makes it usable as a regression oracle.

---

## 9. Privacy & security (therapy-grade)

The restricted plane may contain deeply sensitive content. Treat it accordingly.

- **Synthetic-only scope.** Defaults capture locally and nothing leaves the host.
  `THERAPY_INTERACTION_REMOTE_EXPORT=0` is the egress gate; genuine data, public exposure, or
  remote export are *promotion triggers* that need an explicit new threat-model decision — not
  a casual env flip.
- **Never commit real conversations.** Tests use synthetic fixtures only. Journals live under
  the data volume (`$THERAPY_DATA_DIR`, git-ignored) — never add one to a commit, an issue, or
  an upload.
- **Keep sensitive content out of the broad plane.** Prove it with `make obs-canary-scan`
  before you push.
- **UIs are loopback-only** (`127.0.0.1`). Keep them that way in dev; exposing them is a
  threat-model decision, not a convenience.

Your data is local-first (SPEC §8) and yours to inspect or destroy — export or wipe the
container's data volume directly (the `delete` is irreversible; it wipes `/data`):

```bash
docker compose exec therapy uv run --no-dev python -m therapy.memory export > therapy-data.json
docker compose exec therapy uv run --no-dev python -m therapy.memory delete --yes
```

---

## 10. Debugging playbook

**Always start here**

```bash
curl -s localhost:8000/health     # -> {"status":"ok","version":"…"}
make status                        # container health
```

**"Tests fail with `StopIteration` on every case"** — stale bind-mount inode after editing
`pyproject.toml`/`uv.lock`. → `make restart`.

**"Voice works but the LLM reply fails"**

1. Find the interaction in the journal and **replay it** ([§8](#8-evaluation--replay-workflow)).
2. Inspect the captured `PROVIDER_EVENT` / `TERMINAL_ERROR` evidence (the raw provider error body).
3. Follow the correlation id across planes.

**"Latency increased"** — Grafana (broad plane): STT latency, retrieval latency, LLM
time-to-first-token, TTS synthesis latency. Baseline the overhead with `make obs-baseline`. Quick server-side numbers without
Grafana: `docker compose logs therapy | grep TTFA` (client time-to-first-audio, per turn).

**"The model's behavior changed"** — Phoenix (restricted plane): diff the prompt, memory
context, retrieval context, and completion between a known-good captured interaction and the
new one.

---

## 11. Contributor checklist (runnable)

Each item maps to a real command — don't eyeball it:

- [ ] **`make check`** passes — lint + Pyright + suite w/ coverage floor (the pre-push gate)
- [ ] **`make e2e`** passes (if you touched voice/UI/transport)
- [ ] Voice loop verified — `docker compose exec therapy uv run --no-dev python scripts/verify_voice_text_loop.py`
- [ ] Memory continuity verified — `docker compose exec therapy uv run --no-dev python scripts/verify_memory_continuity.py`
- [ ] **`make obs-canary-scan`** passes — no sensitive data reachable by the broad plane
- [ ] **`make obs-fixture-hash`** unchanged (or intentionally regenerated with `make obs-fixtures`) — replay stays deterministic
- [ ] Metric attribute sets remain bounded (`observability/metrics.py`)
- [ ] `.env.example` updated if you added a knob

> ⚠️ v1 is **single-user**: a new WebRTC connection preempts the running pipeline. Don't run
> the voice/relay verifications against a container a browser tab (or another dev) is actively
> using — the connections evict each other and cross-contaminate results. See the reliability
> notes in [`../README.md`](../README.md).

> Coverage floor is ratcheted via `COV_MIN` (currently 80). If you add code, add tests —
> the floor only goes up.

---

## 12. Where to go next

- [`../README.md`](../README.md) — product overview, configuration, crisis-contact setup, WebRTC/PWA notes
- [`SPEC.md`](SPEC.md) — the design contract and the numbered requirements (SPEC §…) the code cites
- `.env.example` — every environment knob, commented
- `make help` — the authoritative, live list of developer commands
