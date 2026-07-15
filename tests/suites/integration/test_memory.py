import asyncio
import json
import wave
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from therapy.memory import MemoryStore
from therapy.memory.summarizer import LLMSummarizer, render_transcript


def test_session_lifecycle(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    first_session = store.create_session()
    user_turn_id = store.add_turn(
        first_session,
        "user",
        "voice",
        "es",
        "Hoy fue difícil.",
    )
    assistant_turn_id = store.add_turn(
        first_session,
        "assistant",
        "text",
        "en",
        "That sounds heavy.",
    )
    store.end_session(first_session, "The user said the day was difficult.")
    second_session = store.create_session()

    sessions = store.sessions()
    assert [session["id"] for session in sessions] == [second_session, first_session]
    assert sessions[1]["summary"] == "The user said the day was difficult."
    assert sessions[1]["ended_at"] is not None

    turns = store.session_turns(first_session)
    assert [turn["id"] for turn in turns] == [user_turn_id, assistant_turn_id]
    assert turns[0]["role"] == "user"
    assert turns[0]["modality"] == "voice"
    assert turns[0]["language"] == "es"
    assert turns[1]["role"] == "assistant"
    assert turns[1]["text"] == "That sounds heavy."


def test_resume_candidate_returns_recently_ended_newest_session(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path)
    session_id = store.create_session()
    store.add_turn(session_id, "user", "text", "en", "Hello.")
    store.end_session(session_id)

    assert store.resume_candidate(900.0) == session_id


def test_resume_candidate_returns_none_when_stale(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    session_id = store.create_session()
    store.add_turn(session_id, "user", "text", "en", "Hello.")
    store.end_session(session_id)

    assert store.resume_candidate(900.0, now=datetime.now(UTC) + timedelta(hours=2)) is None


def test_resume_candidate_uses_last_turn_for_unfinalized_session(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path)
    session_id = store.create_session()
    stale_started_at = (datetime.now(UTC) - timedelta(hours=2)).isoformat(
        timespec="microseconds"
    )
    with store._connect() as connection:
        with connection:
            connection.execute(
                "UPDATE sessions SET started_at = ? WHERE id = ?",
                (stale_started_at, session_id),
            )
    store.add_turn(session_id, "user", "text", "en", "Still here.")
    last_turn = store.session_turns(session_id)[-1]
    now = datetime.fromisoformat(str(last_turn["ts"])) + timedelta(minutes=5)

    assert store.resume_candidate(900.0, now=now) == session_id


def test_resume_candidate_empty_store_returns_none(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)

    assert store.resume_candidate(900.0) is None


def test_resume_candidate_zero_window_returns_none_for_fresh_session(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path)
    store.create_session()

    assert store.resume_candidate(0.0) is None


def test_resume_candidate_ignores_empty_session(tmp_path: Path) -> None:
    # A connectivity probe (netcheck), or a connect that dropped before the
    # user said anything, creates a session with no turns. It is not a
    # conversation to resume — offering it put a "Resume conversation" button
    # in front of the user with nothing behind it (field test 2026-07-11).
    store = MemoryStore(tmp_path)
    store.create_session()  # no turns

    assert store.resume_candidate(900.0) is None


def test_resume_candidate_skips_empty_session_for_the_real_one(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path)
    real = store.create_session()
    store.add_turn(real, "user", "text", "en", "Hello.")
    store.end_session(real)
    store.create_session()  # a newer empty probe must not hide the real one

    assert store.resume_candidate(900.0) == real


def test_reopen_session_clears_finalization_and_summary_history(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path)
    session_id = store.create_session()
    store.end_session(session_id, "The user checked in.")
    assert store.recent_summaries()[0]["summary"] == "The user checked in."

    store.reopen_session(session_id)

    session = store.sessions()[0]
    assert session["ended_at"] is None
    assert session["summary"] is None
    assert store.recent_summaries() == []


def test_audio_archival(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    session_id = store.create_session()
    audio = b"\x00\x00\x01\x00\xff\xff"
    turn_id = store.add_turn(
        session_id,
        "user",
        "voice",
        "en",
        "Audio turn",
        audio=audio,
        sample_rate=8000,
    )

    turn = store.session_turns(session_id)[0]
    assert turn["audio_path"] == f"audio/{session_id}/{turn_id}.wav"
    wav_path = tmp_path / str(turn["audio_path"])
    assert wav_path.is_file()

    with wave.open(str(wav_path), "rb") as wav_file:
        assert wav_file.getframerate() == 8000
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.readframes(3) == audio


def test_recent_summaries_chronological_with_limit(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    for summary in ("older", "middle", "newer"):
        session_id = store.create_session()
        store.end_session(session_id, summary)
    unsummarized_session = store.create_session()
    store.end_session(unsummarized_session)

    summaries = store.recent_summaries(limit=2)
    assert [summary["summary"] for summary in summaries] == ["middle", "newer"]
    assert all(set(summary) == {"started_at", "ended_at", "summary"} for summary in summaries)


def test_flat_fact_surface_is_retired_after_graph_migration(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)

    with pytest.raises(RuntimeError, match="flat facts are retired"):
        store.upsert_fact("The user prefers morning walks.")
    with pytest.raises(RuntimeError, match="flat facts are retired"):
        store.facts()


def test_export_all_json_round_trips(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    session_id = store.create_session()
    store.add_turn(session_id, "user", "text", "en", "I called Ana.")
    store.end_session(session_id, "The user said they called Ana.")

    snapshot = store.export_all()
    loaded = json.loads(json.dumps(snapshot))

    assert set(loaded) == {"exported_at", "sessions"}
    assert loaded["sessions"][0]["id"] == session_id
    assert loaded["sessions"][0]["turns"][0]["text"] == "I called Ana."


def test_delete_all_clears_tables_and_audio_but_store_still_works(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path)
    session_id = store.create_session()
    store.add_turn(
        session_id,
        "user",
        "voice",
        "en",
        "Audio turn",
        audio=b"\x00\x00",
    )
    store.end_session(session_id, "Summary")
    assert (tmp_path / "audio").is_dir()

    store.delete_all()

    assert store.sessions() == []
    assert store.session_turns(session_id) == []
    assert not (tmp_path / "audio").exists()
    new_session_id = store.create_session()
    assert store.sessions()[0]["id"] == new_session_id


def test_transcript_rendering_and_empty_summary_short_circuit() -> None:
    turns: list[dict[str, object]] = [
        {
            "role": "user",
            "modality": "voice",
            "language": "es",
            "text": "Me sentí mejor hoy.",
        },
        {
            "role": "assistant",
            "modality": "text",
            "language": "en",
            "text": "What helped?",
        },
    ]

    assert render_transcript(turns) == (
        "user (es, voice): Me sentí mejor hoy.\n"
        "assistant (en): What helped?"
    )
    assert asyncio.run(LLMSummarizer(provider="anthropic").summarize([])) == ""


def test_parse_facts_filters_noise() -> None:
    from therapy.memory.distill import parse_facts

    raw = "- Has a dog named Bruno.\n\n* Works as a nurse.\nNONE\n" + "x" * 250
    assert parse_facts(raw) == ["Has a dog named Bruno.", "Works as a nurse."]
    assert parse_facts("NONE") == []


def test_delete_session_removes_rows_and_audio_only_for_that_session(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path)
    doomed = store.create_session()
    doomed_turn = store.add_turn(
        doomed, "user", "voice", "es", "Bórrame.", audio=b"\x00\x00" * 160
    )
    kept = store.create_session()
    store.add_turn(kept, "user", "text", "en", "Keep me.")

    doomed_audio = tmp_path / "audio" / doomed
    assert doomed_audio.exists()
    assert store.has_session(doomed)

    assert store.delete_session(doomed) is True
    assert not store.has_session(doomed)
    assert store.session_turns(doomed) == []
    assert not doomed_audio.exists()
    assert doomed_turn not in [turn["id"] for turn in store.session_turns(kept)]
    assert store.has_session(kept)
    assert len(store.session_turns(kept)) == 1

    assert store.delete_session("nope") is False


def test_titles_set_ensure_and_legacy_migration(tmp_path: Path) -> None:
    import sqlite3

    # A database created before titles existed must gain the column.
    legacy = tmp_path / "therapy.db"
    with sqlite3.connect(legacy) as connection:
        connection.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                summary TEXT
            );
            INSERT INTO sessions (id, started_at) VALUES ('old', '2026-07-09T10:00:00+00:00');
            """
        )

    store = MemoryStore(tmp_path)
    assert store.sessions()[0]["title"] is None  # migrated, not crashed

    # ensure_title fills only the blank; a rename (set_title) always wins.
    store.ensure_title("old", "Generated title")
    assert store.sessions()[0]["title"] == "Generated title"
    store.ensure_title("old", "Second generation")
    assert store.sessions()[0]["title"] == "Generated title"
    assert store.set_title("old", "My own name") is True
    assert store.sessions()[0]["title"] == "My own name"
    assert store.set_title("nope", "x") is False


def test_clean_title_and_dominant_turn_language() -> None:
    from therapy.memory.summarizer import clean_title, dominant_turn_language

    assert clean_title('"Ansiedad por el trabajo."\nExtra line') == "Ansiedad por el trabajo"
    assert clean_title("   ") is None
    assert clean_title("x" * 200) == "x" * 80

    turns = [
        {"role": "user", "language": "es", "text": "Hola"},
        {"role": "assistant", "language": "en", "text": "Hi"},
        {"role": "user", "language": "es", "text": "Sigo"},
        {"role": "user", "language": "en", "text": "ok"},
    ]
    assert dominant_turn_language(turns) == "es"
    assert dominant_turn_language([]) == "en"
