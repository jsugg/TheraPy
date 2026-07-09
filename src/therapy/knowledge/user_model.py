"""Property-graph self-model: typed nodes + typed edges over SQLite.

Full schema: SPEC Appendix A. Node and edge types live in extensible
registries (adding a type is config, not a migration). Edges carry the
same claim lifecycle as nodes: statement (canonical English), verbatim
original-language quotes as evidence, occurrence/session counts,
status observation|pattern|confirmed, per-type decay, tombstones.

Boundaries are enforced here: `never_store` checked before any write;
`never_initiate` exposed to context assembly and the proactivity engine.
Phase 2.
"""
