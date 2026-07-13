"""STT configuration and language detection helpers (framework-free).

The actual pipeline service (a faster-whisper subclass with per-utterance
language auto-detection) lives in agent.py, the only framework-aware module
(SPEC §5). This module owns the domain knowledge it consumes: which
languages TheraPy speaks and model selection. Text-language detection for
typed turns lives in dialogue/language_choice.py (lingua, word-level).
"""

import os

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
