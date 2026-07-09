"""TTS configuration: Kokoro voices per language (framework-free).

Kokoro stock voices, deliberately not a self-clone (SPEC §10). One female
voice per language, chosen for naturalness; env-overridable. The pipeline
switches voice+language per turn based on the detected utterance language.
"""

import os

VOICE_BY_LANGUAGE = {
    "en": "af_heart",  # US English
    "es": "ef_dora",   # Spanish
    "pt": "pf_dora",   # Brazilian Portuguese
}

KOKORO_SAMPLE_RATE = 24_000


def voice_for(language: str) -> str:
    """Voice id for a supported language code, honoring env overrides."""
    override = os.environ.get(f"THERAPY_VOICE_{language.upper()}")
    return override or VOICE_BY_LANGUAGE[language]
