"""STT configuration and language detection helpers (framework-free).

The actual pipeline service (a faster-whisper subclass with per-utterance
language auto-detection) lives in agent.py, the only framework-aware module
(SPEC §5). This module owns the domain knowledge it consumes: which
languages TheraPy speaks, model selection, and a lightweight text-language
heuristic for typed turns (no audio to detect from).
"""

import os
import re

SUPPORTED_LANGUAGES = ("en", "es", "pt")
DEFAULT_LANGUAGE = "en"


def whisper_model() -> str:
    """Multilingual faster-whisper model name (env-overridable)."""
    return os.environ.get("THERAPY_WHISPER_MODEL", "small")


def clamp_language(code: str | None, fallback: str) -> str:
    """Clamp a detected language code to the supported set."""
    if code:
        base = code.lower().split("-")[0]
        if base in SUPPORTED_LANGUAGES:
            return base
    return fallback


# Distinctive, high-frequency function words per language. Deliberately
# excludes words shared across es/pt (que, o, a, e, para, com…).
_STOPWORDS = {
    "en": {"the", "and", "is", "are", "you", "i", "it", "of", "to", "my", "was", "this"},
    "es": {"el", "los", "las", "es", "y", "yo", "esto", "pero", "muy", "hoy", "está", "qué"},
    "pt": {"os", "as", "é", "eu", "isso", "mas", "muito", "hoje", "está", "não", "você", "em"},
}

_WORD_RE = re.compile(r"[a-záéíóúàâãêõçñü]+", re.IGNORECASE)


def guess_language(text: str, fallback: str = DEFAULT_LANGUAGE) -> str:
    """Best-effort language guess for typed text.

    Good enough to pick a TTS voice for a typed turn; the LLM does its own
    (better) detection from the full message. Ties fall back to `fallback`
    so short ambiguous inputs stay in the conversation's current language.
    """
    words = {w.lower() for w in _WORD_RE.findall(text)}
    scores = {lang: len(words & stops) for lang, stops in _STOPWORDS.items()}
    best = max(scores, key=lambda lang: scores[lang])
    if scores[best] == 0 or list(scores.values()).count(scores[best]) > 1:
        return fallback
    return best
