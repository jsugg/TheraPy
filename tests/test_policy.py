from therapy.dialogue.policy import build_system_prompt, crisis_resources


def test_prompt_covers_trilingual_and_boundaries() -> None:
    prompt = build_system_prompt()
    for term in ("Spanish", "English", "Portuguese"):
        assert term in prompt
    assert "never diagnose" in prompt
    assert "validate" in prompt.lower() and "challenge" in prompt.lower()


def test_crisis_resources_env_override(monkeypatch) -> None:
    monkeypatch.setenv("THERAPY_CRISIS_RESOURCES", "Línea 135")
    assert crisis_resources() == "Línea 135"
    assert "Línea 135" in build_system_prompt()
