"""Distillation: observation inbox -> graph promotion; graduation; context.

Runs between sessions. Promotes freeform observations into nodes/edges
(attaching quotes as evidence), applies the graduation rules (mechanical
floor: >=3 occurrences across >=2 sessions makes a claim *eligible*;
LLM judgment proposes; only explicit user confirmation graduates), and
assembles per-conversation context: identity + preferences + boundaries,
active goals/threads, graph walk to top-K relevant nodes with their
confirmed edges. Phase 2.
"""
