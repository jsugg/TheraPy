# TheraPy

A voice-first personal assistant for **self-understanding** — CBT/OT-informed
coaching around what you say today, with a committed SER roadmap for how you
speak over time.

Speech emotion recognition is a committed roadmap layer via
[`ser`](https://github.com/jsugg/ser). TheraPy already reserves the product
boundary in `perception/emotion.py` and `session/timeline.py`, and local raw
utterance audio is retained so the SER adapter can analyze future turns and
retroactively re-analyze history once the integration lands.

> Not therapy, not a therapist replacement, no diagnoses. A therapy-*informed*
> tool for getting to know yourself.

## Status

Phases 0–2 engineering complete (framework spike: Pipecat, see
[`docs/framework-spike.md`](docs/framework-spike.md)). Phase 1 — the
trilingual voice+text loop (es/en/pt) — is implemented and **human-accepted**:
the owner held 5-min mixed voice/text conversations in each language from the
phone over Tailscale and confirmed on-device install. Many phone field tests
were folded back as fixes along the way (SPEC §9 Hardening 7–11); latency
tuning against the target LLM provider (risk R1) is deferred to a later pass.
One PWA serves both
interfaces: a **web interface** in any desktop browser and an installable
**mobile interface** on the phone. Speak or type in the same conversation,
switch mid-turn, barge-in supported. A persistent companion avatar ("Rowan",
swappable) shows live presence — listening, thinking, speaking, pushed from the
pipeline — with an optional fullscreen focus mode and push-to-talk. The reply language is user-selectable
(Auto · ES · EN · PT): auto follows the word-level dominant language of your
last phrase, a pin constrains replies only (SPEC §7). Phase 2 adds local
memory: every session is stored in SQLite under the data dir (transcripts +
raw utterance audio, never leaving the host), summarized at disconnect, and
distilled into user-model facts — new conversations open knowing the prior
context, and a 📖 history view in the PWA browses past transcripts. See
[`docs/SPEC.md`](docs/SPEC.md) for the full specification and roadmap.

Latest dry-run result (2026-07-10, shipped image, fully-local pedrolucas/smollm3:3b-q4_k_m):
all ten scenarios green — trilingual turns, typed-turn silence, barge-in,
both SPEC §7 normative code-switched phrases, pin/unpin. Client-side TTFA
9.2–32.5 s on a warm container (whisper, Kokoro, and the LLM share this
CPU) — to be re-measured with the target provider during verification.

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
ollama pull pedrolucas/smollm3:3b-q4_k_m       # default model — decent es/en/pt, CPU-friendly
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

### Crisis contacts

Crisis contacts are deliberately environment-only so safety configuration
cannot depend on insight, research, or proactivity. Set
`THERAPY_CRISIS_CONTACTS` to a JSON array of at most 20 objects containing
exactly `label` and `value` string fields; see `.env.example`. Invalid JSON or
schema is reported by `/api/crisis-resources`, while the live crisis protocol
continues with its built-in emergency-services/trusted-person fallback. Restart
after changing contacts. `THERAPY_CRISIS_RESOURCES` remains a legacy
plain-string fallback.

The 2024 prototype lives at
[`jsugg/TheraPy-legacy`](https://github.com/jsugg/TheraPy-legacy) (archived);
no code was carried over.

## Development

Developer setup, the test and verification commands, and the build→observe→replay→evaluate
workflow live in **[docs/dev-quick-start.md](docs/dev-quick-start.md)** — the canonical
developer guide. Run `make help` for the full command list.

## Running it & reliability

**Web interface (desktop):** open `http://localhost:8000` in any browser —
localhost is a secure context, so the microphone works out of the box.

**Mobile interface (phone):** join the Tailscale tailnet and open
`http://<machine-name>:8000` — install it from the browser menu (PWA).
Note: browsers require a secure context for microphone access on non-localhost
origins; enable Tailscale HTTPS (`tailscale serve`) or add the origin to the
browser's insecure-origin allowlist for the tailnet hostname.

Install it in **Chrome** — the browser E2E confirms the app meets Chrome's
installability criteria. Two gotchas: DuckDuckGo has no PWA-install support at
all, and after you *uninstall* a PWA, Chrome suppresses the automatic install
banner for that origin for a while — the app is still installable via the ⋮
menu → **Install app** (or the omnibox install icon), just not re-prompted
automatically.

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
python scripts/verify_relay_connectivity.py --relay-only        # against http://localhost:8000
python scripts/verify_relay_connectivity.py --server http://<host>:8000 --relay-only
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

## License

MIT
