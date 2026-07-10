# TheraPy

A voice-first personal assistant for **self-understanding** — CBT/OT-informed
coaching that listens to *how* you speak, not just what you say.

Speech emotion recognition is provided by [`ser`](https://github.com/jsugg/ser);
every conversation builds an emotional timeline and a personal knowledge graph
that compound into longitudinal self-insight.

> Not therapy, not a therapist replacement, no diagnoses. A therapy-*informed*
> tool for getting to know yourself.

## Status

Phases 0–2 engineering complete (framework spike: Pipecat, see
[`docs/framework-spike.md`](docs/framework-spike.md)). Phase 1 — the
trilingual voice+text loop (es/en/pt) — is implemented and dry-run green;
only the human acceptance conversation remains. One PWA serves both
interfaces: a **web interface** in any desktop browser and an installable
**mobile interface** on the phone. Speak or type in the same conversation,
switch mid-turn, barge-in supported. The reply language is user-selectable
(Auto · ES · EN · PT): auto follows the word-level dominant language of your
last phrase, a pin constrains replies only (SPEC §7). Phase 2 adds local
memory: every session is stored in SQLite under the data dir (transcripts +
raw utterance audio, never leaving the host), summarized at disconnect, and
distilled into user-model facts — new conversations open knowing the prior
context, and a 📖 history view in the PWA browses past transcripts. See
[`docs/SPEC.md`](docs/SPEC.md) for the full specification and roadmap.

## Configuration

Copy `.env.example` to `.env`. The LLM provider is swappable
(`THERAPY_LLM=anthropic | openrouter | ollama`); production default is the
Claude API, with OpenRouter free models or a local Ollama for development.
Whisper/Kokoro model weights download to `~/.cache` on first run
(~800 MB total).

Fully-local LLM via Ollama (host-side, so the container reaches it at
`host.docker.internal`):

```sh
ollama serve                # on the host
ollama pull gemma3:4b       # default model — decent es/en/pt, CPU-friendly
# .env: THERAPY_LLM=ollama
#       OLLAMA_BASE_URL=http://host.docker.internal:11434/v1
```

Dropped connections resume: reconnecting within
`THERAPY_RESUME_WINDOW_SECS` (default 15 min) continues the interrupted
session — same transcript, same context — instead of starting a new one.
Set it to `0` to make every connection a fresh session. The chat view
re-renders the resumed transcript on connect (server truth), and the 📖
history browser can start a fresh conversation, continue any past
session, rename it (titles are auto-generated from the topic at session
end), or delete one (turns + archived audio) outright.

The 2024 prototype lives at
[`jsugg/TheraPy-legacy`](https://github.com/jsugg/TheraPy-legacy) (archived);
no code was carried over.

## Development

```sh
uv sync            # install
uv run pytest      # test
uv run uvicorn therapy.server.app:app --reload   # dev server
docker compose up  # containerized
```

Phase-1 instrumented dry run (no microphone needed — a scripted WebRTC
client speaks synthesized es/en/pt utterances at the live server and checks
language detection, reply-language choice per SPEC §7 (both normative
code-switched phrases spoken, plus pin/unpin over the data channel),
typed-turn silence, and barge-in, reporting client-side time-to-first-audio;
exits non-zero if any scenario fails):

```sh
docker compose up -d
docker compose exec therapy uv run --no-dev python scripts/phase1_dryrun.py
docker compose logs therapy | grep TTFA   # server-side numbers (risk R1)
```

Latest dry-run result (2026-07-10, fully-local gemma3:4b): all ten
scenarios green — trilingual turns, typed-turn silence, barge-in, both
SPEC §7 normative code-switched phrases, pin/unpin. Client-side TTFA
7.9–50.5 s (first turn pays cold model loading; whisper, Kokoro, and the
LLM share this CPU) — to be re-measured with the target provider during
the acceptance run.

Phase-2 acceptance (continuity + data round-trip; runs a scripted
two-session conversation against the live server, then exercises
export/delete — **the delete step wipes the data volume**):

```sh
docker compose exec therapy uv run --no-dev python scripts/phase2_acceptance.py
```

Personal data is local-first (SPEC §8) and yours to inspect or destroy:

```sh
docker compose exec therapy uv run --no-dev python -m therapy.memory export > therapy-data.json
docker compose exec therapy uv run --no-dev python -m therapy.memory delete --yes
```

**Web interface (desktop):** open `http://localhost:8000` in any browser —
localhost is a secure context, so the microphone works out of the box.

**Mobile interface (phone):** join the Tailscale tailnet and open
`http://<machine-name>:8000` — install it from the browser menu (PWA).
Note: browsers require a secure context for microphone access on non-localhost
origins; enable Tailscale HTTPS (`tailscale serve`) or add the origin to the
browser's insecure-origin allowlist for the tailnet hostname.

One-time Tailscale setup on the host (interactive — needs the owner):

```sh
brew install --cask tailscale   # or App Store / pkg
tailscale up                    # browser login, joins the tailnet
tailscale cert                  # provision the machine's HTTPS cert (needs
                                # HTTPS + MagicDNS enabled in the admin console)
tailscale serve --bg 8000       # https://<machine>.<tailnet>.ts.net → :8000
```

Then install Tailscale on the phone, sign into the same tailnet, and open
the `https://…ts.net` URL — secure context, so the mic works.

**Phone voice path (TURN):** the pipeline runs inside Docker and only
advertises container-internal WebRTC candidates, which a phone can never
reach — so the compose stack ships a `turn` relay (coturn) and the PWA
allocates a relay at `turn:<page-host>:3478` automatically. Verify the
relay path from any client machine (this simulates the phone by offering
only relay candidates):

```sh
python scripts/netcheck.py --relay-only        # against http://localhost:8000
python scripts/netcheck.py --server http://<host>:8000 --relay-only
```

**Reliability (three layers):** both services restart automatically
(`unless-stopped`) and are memory-capped (`mem_limit`) so runaway memory
OOM-kills a container — which the restart policy heals — instead of
exhausting the Docker VM, which wedges at the hypervisor level and hangs
the docker CLI and every port-forward with it. Inside the container,
uvicorn runs under a watchdog that restarts it if the event loop hangs
(health probe failures), and a compose healthcheck surfaces liveness in
`docker compose ps`. A new WebRTC connection preempts the previous
pipeline — v1 is single-user, and stacked pipelines are how the container
used to run out of memory. Finally, for the VM-wedge case nothing inside
Docker can fix, a host-side supervisor escalates from container restart
to a full provider restart:

```sh
python3 scripts/hostwatch.py   # on the host; probes /health, restarts
                               # the container — or OrbStack itself when
                               # the docker CLI is wedged — then
                               # `docker compose up -d`
```

The PWA shell also degrades gracefully: service-worker fetches time out
after 8 s and fall back to the cached shell rather than loading forever.

Note for Intel Macs: `onnxruntime` (via `kokoro-onnx`) no longer publishes
macOS x86_64 wheels, so `uv sync` fails there — use `docker compose up`
instead (Linux wheels are available).

## License

MIT
