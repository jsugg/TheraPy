from therapy.perception.stt import clamp_language
from therapy.speech.tts import VOICE_BY_LANGUAGE, voice_for


def test_clamp_language() -> None:
    assert clamp_language("es", "en") == "es"
    assert clamp_language("pt-BR", "en") == "pt"
    assert clamp_language("fr", "es") == "es"  # unsupported → fallback
    assert clamp_language(None, "pt") == "pt"


def test_voice_map_covers_all_languages(monkeypatch) -> None:
    assert set(VOICE_BY_LANGUAGE) == {"en", "es", "pt"}
    for lang in VOICE_BY_LANGUAGE:
        assert voice_for(lang)
    monkeypatch.setenv("THERAPY_VOICE_ES", "em_alex")
    assert voice_for("es") == "em_alex"
