"""Reply-language choice (SPEC §7) — including the two normative examples."""

import pytest

from therapy.dialogue.language_choice import ReplyLanguage, dominant_language


# --- dominant_language: word-level majority ---------------------------------


def test_spec_example_minority_words_do_not_flip() -> None:
    # SPEC §7 normative: the lone "ok" must not flip a Spanish phrase.
    assert dominant_language("Todo bien, me estoy sintiendo ok", current="es") == "es"


def test_spec_example_mid_phrase_dominance_flip() -> None:
    # SPEC §7 normative: English majority after a Spanish opening → English.
    phrase = "Sí, todo bien… though… no… actually things have been hard lately"
    assert dominant_language(phrase, current="es") == "en"


def test_monolingual_phrases() -> None:
    assert dominant_language("I finally finished the project today", "es") == "en"
    assert dominant_language("Hoy fue un día muy largo en el trabajo", "en") == "es"
    assert dominant_language("Hoje foi um dia bem longo no trabalho", "en") == "pt"


def test_empty_or_undetectable_keeps_current() -> None:
    assert dominant_language("", "pt") == "pt"
    assert dominant_language("…", "pt") == "pt"
    assert dominant_language("42", "es") == "es"


# --- ReplyLanguage: auto tracking + pin -------------------------------------


def test_auto_follows_dominant_language() -> None:
    choice = ReplyLanguage(initial="en")
    assert choice.note_phrase("Hoy fue un día muy largo en el trabajo") == "es"
    assert choice.note_phrase("Todo bien, me estoy sintiendo ok") == "es"
    assert choice.note_phrase("Actually, let me switch to English now") == "en"


def test_pin_constrains_replies_regardless_of_input() -> None:
    choice = ReplyLanguage(initial="en")
    assert choice.set_pin("pt") == "pt"
    # User keeps speaking Spanish; replies stay pinned to Portuguese.
    assert choice.note_phrase("Hoy fue un día muy largo en el trabajo") == "pt"
    assert choice.pinned == "pt"


def test_unpin_restores_auto_from_latest_phrase() -> None:
    choice = ReplyLanguage(initial="en")
    choice.set_pin("pt")
    choice.note_phrase("Hoy fue un día muy largo en el trabajo")  # auto tracks es
    assert choice.set_pin(None) == "es"


def test_pin_rejects_unsupported_codes() -> None:
    choice = ReplyLanguage()
    with pytest.raises(ValueError):
        choice.set_pin("fr")
