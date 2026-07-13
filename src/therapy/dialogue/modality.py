"""Reply-modality policy: mirror the input modality, user override wins.

Framework-free. SPEC §5: response modality mirrors the input by default —
a spoken turn gets a spoken reply, a typed turn gets a silent one — with a
user override on top. The pipeline (agent.py) consults this state to tell
the LLM/TTS stage whether to synthesize speech for the upcoming reply.
"""

VOICE = "voice"
TEXT = "text"


class ReplyModality:
    """Tracks input modality and speaker override; decides if replies speak."""

    def __init__(self) -> None:
        self._last_input: str = VOICE
        self._override: bool | None = None  # None = auto (mirror)

    @property
    def speak(self) -> bool:
        """Whether the next reply should be synthesized to speech."""
        if self._override is not None:
            return self._override
        return self._last_input == VOICE

    def note_turn(self, modality: str) -> bool:
        """Record a user turn's modality; returns the resulting `speak`."""
        if modality not in (VOICE, TEXT):
            raise ValueError(f"Unknown modality: {modality!r}")
        self._last_input = modality
        return self.speak

    def set_override(self, enabled: bool | None) -> bool:
        """Set (or clear, with None) the user's speaker override."""
        self._override = enabled
        return self.speak
