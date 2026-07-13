import asyncio
import json
from pathlib import Path
import wave

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


def test_upsert_fact_dedupes_exact_statement(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.upsert_fact("The user prefers morning walks.")
    store.upsert_fact("The user prefers morning walks.")

    facts = store.facts()
    assert len(facts) == 1
    assert facts[0]["statement"] == "The user prefers morning walks."
    assert facts[0]["kind"] == "observation"
    assert facts[0]["n_occurrences"] == 2


def test_export_all_json_round_trips(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    session_id = store.create_session()
    store.add_turn(session_id, "user", "text", "en", "I called Ana.")
    store.end_session(session_id, "The user said they called Ana.")
    store.upsert_fact("The user knows Ana.", kind="relationship")

    snapshot = store.export_all()
    loaded = json.loads(json.dumps(snapshot))

    assert set(loaded) == {"exported_at", "sessions", "facts"}
    assert loaded["sessions"][0]["id"] == session_id
    assert loaded["sessions"][0]["turns"][0]["text"] == "I called Ana."
    assert loaded["facts"][0]["statement"] == "The user knows Ana."


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
    store.upsert_fact("A fact.")
    assert (tmp_path / "audio").is_dir()

    store.delete_all()

    assert store.sessions() == []
    assert store.session_turns(session_id) == []
    assert store.facts() == []
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
