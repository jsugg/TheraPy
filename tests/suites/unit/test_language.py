from therapy.perception.stt import clamp_language, is_supported, plausible_segment
from therapy.speech.tts import VOICE_BY_LANGUAGE, voice_for


def test_clamp_language() -> None:
    assert clamp_language("es", "en") == "es"
    assert clamp_language("pt-BR", "en") == "pt"
    assert clamp_language("fr", "es") == "es"  # unsupported → fallback
    assert clamp_language(None, "pt") == "pt"


def test_is_supported() -> None:
    assert is_supported("es")
    assert is_supported("pt-BR")
    assert not is_supported("ko")
    assert not is_supported(None)


def test_plausible_segment_rejects_hallucination_signatures() -> None:
    ok = {"no_speech_prob": 0.1, "avg_logprob": -0.4, "compression_ratio": 1.4}
    assert plausible_segment(**ok)
    # Repeated stock phrases compress unnaturally well ("뵐게요. 뵐게요. 뵐게요.").
    assert not plausible_segment(**{**ok, "compression_ratio": 3.1})
    assert not plausible_segment(**{**ok, "avg_logprob": -1.6})
    assert not plausible_segment(**{**ok, "no_speech_prob": 0.9})


def test_voice_map_covers_all_languages(monkeypatch) -> None:
    assert set(VOICE_BY_LANGUAGE) == {"en", "es", "pt"}
    for lang in VOICE_BY_LANGUAGE:
        assert voice_for(lang)
    monkeypatch.setenv("THERAPY_VOICE_ES", "em_alex")
    assert voice_for("es") == "em_alex"


def test_genuine_foreign_speech_vs_hallucination() -> None:
    from therapy.perception.stt import genuine_foreign_speech

    # Plausible German decode: the user really speaks German — transcribe
    # honestly, never silently re-decode (whisper would translate it).
    assert genuine_foreign_speech("de", "Hallo, wie geht's dir?")
    # Unsupported detection whose decode died in the filter: hallucination.
    assert not genuine_foreign_speech("ko", "")
    # Supported languages take the normal path.
    assert not genuine_foreign_speech("es", "Hola, ¿cómo estás?")
    assert not genuine_foreign_speech(None, "whatever")
