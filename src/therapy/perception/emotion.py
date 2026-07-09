"""Anti-corruption adapter around `ser` (speech emotion recognition).

ser's native types stop here: the rest of the app consumes only
`EmotionFrame`. The raw layer stores ser's verbatim labels tagged with the
ser version so historical audio can be re-analyzed by newer ser releases
(SPEC §6); the product layer maps raw labels per consumer via config.

Phase 3: per-turn batch analysis of the VAD-buffered utterance.
Later: ser streaming slots in behind the same iterator shape.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EmotionFrame:
    """One emotion observation for a span of user speech.

    raw_labels holds ser's native output verbatim (label -> score);
    ser_version identifies the label set for later re-mapping.
    """

    turn_id: str
    start_s: float
    end_s: float
    ser_version: str
    raw_labels: dict[str, float] = field(default_factory=dict)
