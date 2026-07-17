"""O3.2 finite transition evidence: reply language, modality, prompt builds,
and the raw-audio LLM-boundary guard. Broad values are enums/buckets only."""

from __future__ import annotations

import pytest


@pytest.fixture
def metric_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, float, dict]]:
    from therapy.observability import telemetry

    calls: list[tuple[str, float, dict]] = []
    monkeypatch.setattr(
        telemetry,
        "record_metric",
        lambda name, value, attrs=None: calls.append((name, value, attrs or {})),
    )
    return calls


def _transitions(calls: list, kind: str) -> list[str]:
    return [
        attrs["outcome"]
        for name, _, attrs in calls
        if name == "therapy_language_transitions_total" and attrs["kind"] == kind
    ]


def test_reply_language_transitions_only_on_state_change(metric_calls) -> None:
    from therapy.dialogue.language_choice import ReplyLanguage

    state = ReplyLanguage()
    state.set_pin("es")
    state.set_pin("es")  # no change -> no transition
    state.set_pin(None)
    with pytest.raises(ValueError, match="Unsupported reply language"):
        state.set_pin("de")

    assert _transitions(metric_calls, "reply_language") == [
        "pinned",
        "auto",
        "unsupported",
    ]


def test_auto_language_change_records_one_transition(metric_calls) -> None:
    from therapy.dialogue.language_choice import ReplyLanguage

    state = ReplyLanguage(initial="en", established=True)
    state.note_phrase("hola cómo estás hoy amigo mío querido")

    assert _transitions(metric_calls, "reply_language") == ["changed"]
    # The language code itself never becomes a metric value.
    for _, _value, attrs in metric_calls:
        assert "es" not in attrs.values()


def test_modality_transitions_are_finite_and_change_only(metric_calls) -> None:
    from therapy.dialogue.modality import TEXT, VOICE, ReplyModality

    modality = ReplyModality()
    modality.note_turn(VOICE)  # same as initial -> no transition
    modality.note_turn(TEXT)
    modality.set_override(True)
    modality.set_override(True)  # no change
    modality.set_override(None)
    with pytest.raises(ValueError, match="Unknown modality"):
        modality.note_turn("smoke-signal")

    assert _transitions(metric_calls, "modality") == [
        "changed",
        "pinned",
        "auto",
        "invalid",
    ]


def test_prompt_builds_record_bounded_sections(metric_calls, monkeypatch) -> None:
    monkeypatch.setenv(
        "THERAPY_CRISIS_CONTACTS", '[{"label": "Línea 135", "value": "135"}]'
    )
    from therapy.dialogue.policy import build_system_prompt, continuity_note

    build_system_prompt()
    continuity_note([], [])
    continuity_note(
        [{"summary": "s", "started_at": "2026-07-16T00:00:00Z"}],
        [{"statement": "f"}],
    )

    builds = [
        attrs
        for name, _, attrs in metric_calls
        if name == "therapy_prompt_builds_total"
    ]
    assert [b["outcome"] for b in builds] == ["success", "success", "success"]
    assert [b["sections"] for b in builds] == ["1-9", "0", "1-9"]


def test_llm_boundary_rejects_raw_audio_bytes() -> None:
    import asyncio

    from therapy.memory.summarizer import complete

    with pytest.raises(TypeError, match="text only"):
        asyncio.run(complete("system", b"\x00\x01raw-audio"))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="text only"):
        asyncio.run(complete(b"raw-audio", "user"))  # type: ignore[arg-type]
