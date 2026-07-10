from therapy.dialogue.policy import (
    build_system_prompt,
    continuity_note,
    crisis_resources,
    language_pin_note,
    language_switch_note,
)


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


def test_language_switch_note_names_the_language() -> None:
    assert "Portuguese" in language_switch_note("pt")
    assert "English" in language_switch_note("en")
    # Unknown codes degrade to the code itself rather than raising.
    assert "fr" in language_switch_note("fr")


def test_language_pin_note_overrides_follow_the_user() -> None:
    note = language_pin_note("pt")
    assert "Portuguese" in note
    assert "regardless" in note


def test_continuity_note_empty_history_is_none() -> None:
    assert continuity_note([], []) is None


def test_continuity_note_renders_summaries_and_facts() -> None:
    note = continuity_note(
        [{"started_at": "2026-07-09T20:00:00+00:00", "summary": "Talked about work."}],
        [{"statement": "Has a dog named Bruno."}],
    )
    assert note is not None
    assert "2026-07-09" in note and "Talked about work." in note
    assert "Has a dog named Bruno." in note
