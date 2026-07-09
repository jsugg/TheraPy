"""Dialogue policy: persona, register, style arc, safety guardrails.

Stable identity, adaptive register (SPEC §5): one character whose tone,
pace, and directness modulate with the user's detected state — register
is an explicit parameter driven by perception. Style arc: validate first,
then challenge; challenge intensity is register-gated.

Safety (SPEC §4): therapy-informed, never therapy — no diagnoses; crisis
language stops coaching and surfaces human resources.

LLM access is provider-agnostic; context = current conversation verbatim
+ distilled past only (SPEC §8).
"""
