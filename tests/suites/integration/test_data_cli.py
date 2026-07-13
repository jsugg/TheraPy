import json
from pathlib import Path

import pytest

from therapy.memory import MemoryStore
from therapy.memory.__main__ import main


def test_export_to_output_file_contains_seeded_session_and_fact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("THERAPY_DATA_DIR", str(tmp_path))
    store = MemoryStore()
    session_id = store.create_session()
    store.add_turn(session_id, "user", "text", "en", "I journaled today.")
    store.end_session(session_id, "The user journaled today.")
    store.upsert_fact("The user journals.", kind="habit")
    output_path = tmp_path / "export.json"

    assert main(["export", "--output", str(output_path)]) == 0

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["sessions"][0]["id"] == session_id
    assert payload["sessions"][0]["turns"][0]["text"] == "I journaled today."
    assert payload["facts"][0]["statement"] == "The user journals."


def test_delete_without_yes_returns_2_and_preserves_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("THERAPY_DATA_DIR", str(tmp_path))
    store = MemoryStore()
    session_id = store.create_session()
    store.add_turn(session_id, "assistant", "text", "en", "I'm listening.")

    assert main(["delete"]) == 2

    captured = capsys.readouterr()
    assert "without --yes" in captured.err
    assert MemoryStore().sessions()[0]["id"] == session_id
    assert MemoryStore().session_turns(session_id)[0]["text"] == "I'm listening."


def test_delete_yes_empties_the_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("THERAPY_DATA_DIR", str(tmp_path))
    store = MemoryStore()
    session_id = store.create_session()
    store.add_turn(session_id, "user", "text", "en", "Please forget this.")
    store.upsert_fact("A temporary fact.")

    assert main(["delete", "--yes"]) == 0

    captured = capsys.readouterr()
    assert "Deleted all TheraPy memory data." in captured.err
    empty_store = MemoryStore()
    assert empty_store.sessions() == []
    assert empty_store.session_turns(session_id) == []
    assert empty_store.facts() == []
