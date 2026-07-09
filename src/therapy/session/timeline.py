"""Session timeline: merged transcript + emotion record; longitudinal queries.

Conversations are server-authoritative streams of modality-agnostic turns
(voice or text), language-tagged, with session boundaries inferred from
gaps and tagged by depth (check-in / conversation / deep session).
Raw utterance audio is retained (SPEC §8) so newer ser versions can
re-analyze history — the timeline gets retroactively smarter. Phase 2.
"""
