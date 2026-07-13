import pytest

from therapy.dialogue.modality import TEXT, VOICE, ReplyModality


def test_mirrors_input_modality() -> None:
    m = ReplyModality()
    assert m.speak  # voice-first default
    assert m.note_turn(TEXT) is False
    assert m.note_turn(VOICE) is True


def test_override_wins_over_mirroring() -> None:
    m = ReplyModality()
    m.note_turn(TEXT)
    assert m.set_override(True) is True
    assert m.note_turn(TEXT) is True  # override sticks across turns
    assert m.set_override(False) is False
    assert m.note_turn(VOICE) is False


def test_clearing_override_returns_to_mirroring() -> None:
    m = ReplyModality()
    m.note_turn(TEXT)
    m.set_override(True)
    assert m.set_override(None) is False  # back to mirroring the typed turn


def test_rejects_unknown_modality() -> None:
    with pytest.raises(ValueError, match="telepathy"):
        ReplyModality().note_turn("telepathy")
