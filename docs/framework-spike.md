# Phase-0 spike: Pipecat vs. LiveKit Agents

**Verdict: Pipecat with `SmallWebRTCTransport`.**
Date: 2026-07-09 · Status: decided — phase 1 is built on it (`agent.py`, PWA client); working voice+text loop confirms the choice in code

## Evaluation criteria (from SPEC)

1. Self-hosted on user-controlled infra: own machine + Tailscale now, small personal VPS later (§8)
2. WebRTC voice to a browser PWA (web + mobile), no native apps (§2)
3. Mixed voice + text in one conversation, switchable mid-conversation (§5)
4. Custom per-turn processing: ser emotion adapter needs the VAD-buffered utterance audio (§6)
5. Local model plugins: faster-whisper STT, Kokoro TTS (§5)
6. Single-user scale — operational simplicity dominates

## Findings

| Criterion | Pipecat | LiveKit Agents |
|---|---|---|
| Self-host footprint | **One Python process.** `SmallWebRTCTransport` is P2P WebRTC straight from browser to the pipeline — no media server, no extra services | Media server (SFU) + **Redis** (room state, job dispatch) + agent worker — three services before the app itself |
| Browser/PWA client | `pipecat-client-web` + small-webrtc transport; prebuilt dev client exists | Mature, polished JS/mobile SDKs (its strongest card) |
| Voice + text mixed | Data channel alongside audio (`on_client_message`) feeds text turns into the same pipeline | Also supported (data streams / text input) |
| Custom processors | **Pipeline-first is the whole design**: frame processors compose into a graph — the ser per-turn adapter, register parameter, and modality-agnostic turns map 1:1 | Clean high-level interface, but customization means working around the abstraction rather than with it |
| Local models | faster-whisper and Kokoro services available in the ecosystem | Plugin set exists but the ecosystem leans cloud-provider |
| Latency | ~50–100 ms behind LiveKit in published comparisons; negligible at conversational scale, and Tailscale P2P removes internet hops anyway | Slight edge, better under heavy packet loss |
| Multi-participant | Not native (irrelevant: single user by design) | Native room model (unused here) |

## Rationale

The decision reduces to criterion 6. LiveKit's advantages — SFU-grade media
routing, rooms, packet-loss resilience, multi-participant — solve problems
TheraPy does not have, and cost a Redis + media-server deployment TheraPy
would have to carry on a personal machine and VPS forever. Pipecat's
pipeline-first model is exactly the architecture the SPEC already describes:
turns flowing through composable processors, with perception (ser) and
register logic as first-class pipeline stages, all in one self-contained
Python process.

## Revisit triggers

- Multi-user or multi-participant scenarios enter the vision → LiveKit's room model becomes relevant
- P2P WebRTC proves unreliable across networks the VPS must serve → managed SFU reconsidered
- Native mobile apps replace the PWA → LiveKit's mobile SDKs weigh more

Migration cost is contained by design: only `agent.py` (server) and the
client transport layer know the framework (SPEC §5).

## Sources

- [Pipecat SmallWebRTCTransport docs](https://docs.pipecat.ai/api-reference/server/services/transport/small-webrtc)
- [pipecat-client-web-transports](https://github.com/pipecat-ai/pipecat-client-web-transports)
- [small-webrtc-prebuilt](https://github.com/pipecat-ai/small-webrtc-prebuilt)
- [LiveKit self-hosting overview](https://docs.livekit.io/transport/self-hosting/)
- [WebRTC.ventures framework comparison (2026)](https://webrtc.ventures/2026/03/choosing-a-voice-ai-agent-production-framework/)
- [Voice agent frameworks: LiveKit & Pipecat](https://www.arunbaby.com/ai-agents/0018-voice-agent-frameworks/)
- [Pipecat vs. LiveKit key differences](https://www.cekura.ai/blogs/pipecat-vs-livekit-the-real-difference)
