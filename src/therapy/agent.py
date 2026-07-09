"""Pipeline assembly — the only module that imports the voice-agent framework.

Framework choice (Pipecat vs. LiveKit Agents) is the phase-0 spike:
docs/framework-spike.md. Everything else in the package must stay
framework-agnostic so the verdict — or a later reversal — touches only
this file.
"""
