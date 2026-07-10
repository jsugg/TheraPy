# TheraPy

A voice-first personal assistant for **self-understanding** — CBT/OT-informed
coaching that listens to *how* you speak, not just what you say.

Speech emotion recognition is provided by [`ser`](https://github.com/jsugg/ser);
every conversation builds an emotional timeline and a personal knowledge graph
that compound into longitudinal self-insight.

> Not therapy, not a therapist replacement, no diagnoses. A therapy-*informed*
> tool for getting to know yourself.

## Status

Phase 0 done (framework spike: Pipecat, see
[`docs/framework-spike.md`](docs/framework-spike.md)). Phase 1 — the
trilingual voice+text loop (es/en/pt) — is implemented and awaiting its
acceptance run. One PWA serves both interfaces: a **web interface** in any
desktop browser and an installable **mobile interface** on the phone. Speak
or type in the same conversation, switch mid-turn, barge-in supported.
Per-utterance language detection picks the reply voice; time-to-first-audio
is instrumented. See [`docs/SPEC.md`](docs/SPEC.md) for the full
specification and roadmap.

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
client speaks synthesized es/en/pt utterances at the live server, checks
language detection, typed-turn silence, and barge-in, and reports
client-side time-to-first-audio):

```sh
docker compose up -d
docker compose exec therapy uv run --no-dev python scripts/phase1_dryrun.py
docker compose logs therapy | grep TTFA   # server-side numbers (risk R1)
```

Latest dry-run result (2026-07-10, free-tier dev LLM): es/en/pt each
detected and answered in-language on one connection; typed turn replied
silently (server-side modality mirroring); barge-in stopped reply audio.
TTFA 3.3–15.5 s server-side (8.3–22.6 s client-side incl. the 0.7 s VAD
stop window) across two clean runs — dominated by dev-LLM latency, to be
re-measured with the target provider during the acceptance run.

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

Note for Intel Macs: `onnxruntime` (via `kokoro-onnx`) no longer publishes
macOS x86_64 wheels, so `uv sync` fails there — use `docker compose up`
instead (Linux wheels are available).

## License

MIT
