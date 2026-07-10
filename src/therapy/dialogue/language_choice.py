"""Reply-language choice (SPEC §7): Auto · ES · EN · PT.

Framework-free. The user picks the reply language in the PWA; the choice
arrives over the data channel as a `reply_language` override (`null` = auto)
and lives here as server-authoritative state.

- **Auto (default):** reply in the dominant language of the last user
  phrase — word-level majority via lingua (restricted to es/en/pt), because
  Whisper labels whole utterances only, which is too coarse for
  code-switched phrases. Near-even mixes and empty/undetectable phrases
  keep the language currently in use — no ping-ponging.
- **Pinned (es/en/pt):** replies (text and TTS voice) always come back in
  the pinned language. The pin constrains replies only: STT keeps
  auto-detecting for transcripts and the timeline, and the auto choice
  keeps tracking the user underneath, so unpinning restores it.
"""

from lingua import Language, LanguageDetectorBuilder

from therapy.perception.stt import DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES

_LINGUA_TO_CODE = {
    Language.ENGLISH: "en",
    Language.SPANISH: "es",
    Language.PORTUGUESE: "pt",
}

_detector = None


def _detector_instance():
    """Lazy singleton — the detector loads per-language models on demand."""
    global _detector
    if _detector is None:
        _detector = LanguageDetectorBuilder.from_languages(*_LINGUA_TO_CODE).build()
    return _detector


def word_language_counts(text: str) -> dict[str, int]:
    """Word counts per supported language for a (possibly code-switched) phrase.

    lingua segments the text into contiguous same-language sections; summing
    each section's word count gives the word-level majority SPEC §7 asks for
    without classifying single words out of context (hopeless for es/pt).
    """
    counts: dict[str, int] = {}
    for section in _detector_instance().detect_multiple_languages_of(text):
        code = _LINGUA_TO_CODE.get(section.language)
        if code:
            counts[code] = counts.get(code, 0) + section.word_count
    return counts


def dominant_language(text: str, current: str) -> str:
    """Dominant language of a phrase; ties and no-detection keep `current`."""
    counts = word_language_counts(text)
    if not counts:
        return current
    best = max(counts.values())
    winners = [code for code, count in counts.items() if count == best]
    if len(winners) == 1:
        return winners[0]
    return current


class ReplyLanguage:
    """Server-authoritative reply-language state: auto tracking + user pin."""

    def __init__(self, initial: str = DEFAULT_LANGUAGE) -> None:
        self._auto = initial
        self._pin: str | None = None

    @property
    def language(self) -> str:
        """The language the next reply must be written and voiced in."""
        return self._pin or self._auto

    @property
    def pinned(self) -> str | None:
        return self._pin

    def set_pin(self, code: str | None) -> str:
        """Pin the reply language (`None` = back to auto); returns the result."""
        if code is not None and code not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Unsupported reply language: {code!r}")
        self._pin = code
        return self.language

    def note_phrase(self, text: str) -> str:
        """Track a user phrase; returns the reply language for this turn.

        The auto choice updates even while pinned, so unpinning resumes
        from the user's actual current language, not a stale one.
        """
        self._auto = dominant_language(text, self._auto)
        return self.language
