# TheraPy

A voice-first personal assistant for **self-understanding** — CBT/OT-informed
coaching that listens to *how* you speak, not just what you say.

Speech emotion recognition is provided by [`ser`](https://github.com/jsugg/ser);
every conversation builds an emotional timeline and a personal knowledge graph
that compound into longitudinal self-insight.

> Not therapy, not a therapist replacement, no diagnoses. A therapy-*informed*
> tool for getting to know yourself.

## Status

Phase 0 — scaffold. See [`docs/SPEC.md`](docs/SPEC.md) for the full
specification and roadmap, and [`docs/framework-spike.md`](docs/framework-spike.md)
for the pipeline-framework decision.

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

## License

MIT
