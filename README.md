# TheraPy

A voice-first personal assistant for **self-understanding** — CBT/OT-informed
coaching that listens to *how* you speak, not just what you say.

Speech emotion recognition is provided by [`ser`](https://github.com/jsugg/ser);
every conversation builds an emotional timeline and a personal knowledge graph
that compound into longitudinal self-insight.

> Not therapy, not a therapist replacement, no diagnoses. A therapy-*informed*
> tool for getting to know yourself.

## Status

Phase 1 — trilingual voice+text loop (es/en/pt). One PWA (web + mobile):
speak or type in the same conversation, switch mid-turn, barge-in supported.
Per-utterance language detection picks the reply voice; time-to-first-audio
is instrumented. See [`docs/SPEC.md`](docs/SPEC.md) for the full
specification and roadmap, and [`docs/framework-spike.md`](docs/framework-spike.md)
for the pipeline-framework decision.

## Configuration

Copy `.env.example` to `.env`. The LLM provider is swappable
(`THERAPY_LLM=anthropic | openrouter | ollama`); production default is the
Claude API, with OpenRouter free models or a local Ollama for development.
Whisper/Kokoro model weights download to `~/.cache` on first run
(~800 MB total).

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

From a phone, join the Tailscale tailnet and open
`http://<machine-name>:8000` — install it from the browser menu (PWA).
Note: browsers require a secure context for microphone access on non-localhost
origins; enable Tailscale HTTPS (`tailscale serve`) or add the origin to the
browser's insecure-origin allowlist for the tailnet hostname.

## License

MIT
