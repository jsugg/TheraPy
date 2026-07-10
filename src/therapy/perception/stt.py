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


def is_supported(code: str | None) -> bool:
    """Whether a detected language code is one TheraPy speaks."""
    return bool(code) and code.lower().split("-")[0] in SUPPORTED_LANGUAGES


def plausible_segment(
    no_speech_prob: float,
    avg_logprob: float,
    compression_ratio: float,
    no_speech_threshold: float = 0.6,
) -> bool:
    """Whisper hallucination filter for one transcription segment.

    Degraded audio makes Whisper emit repeated stock phrases, often in an
    unrelated language (a Korean "뵐게요. 뵐게요." surfaced in field testing
    mid-English-sentence). Repetition inflates the compression ratio and
    the decoder's own confidence drops — the thresholds are the ones
    Whisper itself uses to reject a decoding before falling back.
    """
    if no_speech_prob > no_speech_threshold:
        return False
    if avg_logprob < -1.2:
        return False
    return compression_ratio <= 2.4
