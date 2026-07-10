from collections.abc import Iterator
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from therapy.memory import MemoryStore
from therapy.server.app import _store, app


@pytest.fixture(autouse=True)
def isolated_data_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    monkeypatch.setenv("THERAPY_DATA_DIR", str(tmp_path))
    _store.cache_clear()
    yield
    _store.cache_clear()


def test_sessions_returns_seeded_session_with_turn_count(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    session_id = store.create_session()
    store.add_turn(session_id, "user", "text", "en", "I felt calm.")
    store.add_turn(session_id, "assistant", "text", "en", "What helped?")
    store.end_session(session_id, "The user felt calm.")

    with TestClient(app) as client:
        response = client.get("/api/sessions")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["sessions"]) == 1
    session = payload["sessions"][0]
    assert session["id"] == session_id
    assert session["ended_at"] is not None
    assert session["summary"] == "The user felt calm."
    assert session["turn_count"] == 2


def test_session_detail_returns_turns_in_order(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    session_id = store.create_session()
    first_turn_id = store.add_turn(
        session_id,
        "user",
        "voice",
        "es",
        "Me sentí bien.",
    )
    second_turn_id = store.add_turn(
        session_id,
        "assistant",
        "text",
        "en",
        "What felt different?",
    )

    with TestClient(app) as client:
        response = client.get(f"/api/sessions/{session_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["session"]["id"] == session_id
    assert [turn["id"] for turn in payload["turns"]] == [
        first_turn_id,
        second_turn_id,
    ]
    assert [turn["text"] for turn in payload["turns"]] == [
        "Me sentí bien.",
        "What felt different?",
    ]


def test_session_detail_returns_404_for_unknown_id() -> None:
    with TestClient(app) as client:
        response = client.get("/api/sessions/unknown")

    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found"}


def test_delete_session_endpoint_removes_one_session(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    doomed = store.create_session()
    store.add_turn(doomed, "user", "text", "en", "Erase this.")
    kept = store.create_session()

    with TestClient(app) as client:
        response = client.delete(f"/api/sessions/{doomed}")
        assert response.status_code == 200
        assert response.json() == {"deleted": doomed}
        remaining = client.get("/api/sessions").json()["sessions"]

    assert [session["id"] for session in remaining] == [kept]
    assert client.delete("/api/sessions/nope").status_code == 404


def test_delete_session_refuses_while_pipeline_is_live(tmp_path: Path) -> None:
    from therapy.server import live

    store = MemoryStore(tmp_path)
    session_id = store.create_session()
    token = live.claim(session_id)
    try:
        with TestClient(app) as client:
            assert client.delete(f"/api/sessions/{session_id}").status_code == 409
    finally:
        live.release(session_id, token)
    assert store.has_session(session_id)
