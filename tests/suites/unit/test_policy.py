import pytest

from therapy.dialogue.policy import (
    CrisisConfigurationError,
    build_system_prompt,
    continuity_note,
    crisis_contacts,
    crisis_resources,
    language_pin_note,
    language_switch_note,
    rehydrate_messages,
    resume_note,
)


def test_resume_note_forbids_regreeting() -> None:
    note = resume_note()
    assert "same ongoing conversation" in note
    assert "do not greet" in note


def test_rehydrate_messages_maps_turns_verbatim_and_caps() -> None:
    turns = [
        {"role": "user", "text": "Hola", "language": "es", "modality": "voice"},
        {"role": "assistant", "text": "Hola, ¿cómo estás?", "language": "es"},
        {"role": "user", "text": "", "language": "es"},  # empty → dropped
    ]
    messages = rehydrate_messages(turns)
    assert messages == [
        {"role": "user", "content": "Hola"},
        {"role": "assistant", "content": "Hola, ¿cómo estás?"},
    ]
    many = [{"role": "user", "text": f"t{i}"} for i in range(60)]
    capped = rehydrate_messages(many, limit=40)
    assert len(capped) == 40
    assert capped[0]["content"] == "t20"  # most recent turns win


def test_prompt_covers_trilingual_and_boundaries() -> None:
    prompt = build_system_prompt()
    for term in ("Spanish", "English", "Portuguese"):
        assert term in prompt
    assert "never diagnose" in prompt
    assert "validate" in prompt.lower()
    assert "challenge" in prompt.lower()


def test_crisis_resources_env_override(monkeypatch) -> None:
    monkeypatch.setenv("THERAPY_CRISIS_RESOURCES", "Línea 135")
    assert crisis_resources() == "Línea 135"
    assert "Línea 135" in build_system_prompt()


def test_crisis_contacts_are_strict_but_response_path_keeps_safe_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("THERAPY_CRISIS_CONTACTS", '[{"label":"Only label"}]')

    with pytest.raises(CrisisConfigurationError, match="label and value"):
        crisis_contacts()
    assert "emergency services" in crisis_resources()
    assert "emergency services" in build_system_prompt()


def test_language_switch_note_is_written_in_the_target_language() -> None:
    # The note is instruction AND language prime: an English note in the
    # context is itself English evidence pulling a small model to English.
    assert "português" in language_switch_note("pt")
    assert "español" in language_switch_note("es")
    assert "English" in language_switch_note("en")
    # Unknown codes degrade to the code itself rather than raising.
    assert "fr" in language_switch_note("fr")


def test_language_pin_note_overrides_follow_the_user() -> None:
    note = language_pin_note("pt")
    assert "português" in note
    assert "não importa" in note  # overrides follow-the-user, in Portuguese
    assert "regardless" in language_pin_note("en")


def test_reply_language_reminder_is_short_and_in_target_language() -> None:
    from therapy.dialogue.policy import reply_language_reminder

    note = reply_language_reminder("es")
    assert "español" in note
    assert len(note) < 100  # per-turn cost must stay negligible


def test_continuity_note_empty_history_is_none() -> None:
    assert continuity_note([], []) is None


def test_continuity_note_renders_summaries_and_facts() -> None:
    note = continuity_note(
        [{"started_at": "2026-07-09T20:00:00+00:00", "summary": "Talked about work."}],
        [{"statement": "Has a dog named Bruno."}],
    )
    assert note is not None
    assert "2026-07-09" in note
    assert "Talked about work." in note
    assert "Has a dog named Bruno." in note
