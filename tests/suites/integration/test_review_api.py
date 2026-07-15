from pathlib import Path

from therapy.memory import MemoryStore
from therapy.server.app import _resolve_session


def test_sessions_returns_seeded_session_with_turn_count(data_dir: Path, client) -> None:
    store = MemoryStore(data_dir)
    session_id = store.create_session()
    store.add_turn(session_id, "user", "text", "en", "I felt calm.")
    store.add_turn(session_id, "assistant", "text", "en", "What helped?")
    store.end_session(session_id, "The user felt calm.")

    response = client.get("/api/sessions")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["sessions"]) == 1
    session = payload["sessions"][0]
    assert session["id"] == session_id
    assert session["ended_at"] is not None
    assert session["summary"] == "The user felt calm."
    assert session["turn_count"] == 2


def test_session_detail_returns_turns_in_order(data_dir: Path, client) -> None:
    store = MemoryStore(data_dir)
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


def test_session_detail_returns_404_for_unknown_id(client) -> None:
    response = client.get("/api/sessions/unknown")

    assert response.status_code == 404
    assert response.json() == {"detail": "Session not found"}


def test_session_detail_flags_audio_without_leaking_path(data_dir: Path, client) -> None:
    store = MemoryStore(data_dir)
    session_id = store.create_session()
    store.add_turn(
        session_id, "user", "voice", "es", "Hola.", audio=b"\x01\x00" * 1600
    )
    store.add_turn(session_id, "assistant", "text", "es", "¿Cómo estás?")

    payload = client.get(f"/api/sessions/{session_id}").json()

    voice_turn, text_turn = payload["turns"]
    assert voice_turn["has_audio"] is True
    assert text_turn["has_audio"] is False
    # The host filesystem path is internal — it must not reach the client.
    assert "audio_path" not in voice_turn


def test_turn_audio_endpoint_serves_wav_or_404(data_dir: Path, client) -> None:
    store = MemoryStore(data_dir)
    session_id = store.create_session()
    voice_turn = store.add_turn(
        session_id, "user", "voice", "es", "Hola.", audio=b"\x01\x00" * 1600
    )
    text_turn = store.add_turn(session_id, "assistant", "text", "es", "Hola.")
    other = store.create_session()

    ok = client.get(f"/api/sessions/{session_id}/turns/{voice_turn}/audio")
    assert ok.status_code == 200
    assert ok.headers["content-type"] == "audio/wav"
    assert ok.content[:4] == b"RIFF"  # a real WAV came back
    # No archived audio for a text turn; unknown turn ids 404 too.
    assert (
        client.get(f"/api/sessions/{session_id}/turns/{text_turn}/audio").status_code
        == 404
    )
    assert (
        client.get(f"/api/sessions/{session_id}/turns/999999/audio").status_code == 404
    )
    # A real turn id requested under the wrong session must not resolve.
    assert (
        client.get(f"/api/sessions/{other}/turns/{voice_turn}/audio").status_code == 404
    )


def test_delete_session_endpoint_removes_one_session(data_dir: Path, client) -> None:
    store = MemoryStore(data_dir)
    doomed = store.create_session()
    store.add_turn(doomed, "user", "text", "en", "Erase this.")
    kept = store.create_session()

    response = client.delete(f"/api/sessions/{doomed}")
    assert response.status_code == 200
    result = response.json()
    assert result["deleted"] == doomed
    assert result["mode"] == "keep_knowledge"
    assert result["learned_knowledge_survives"] is True
    remaining = client.get("/api/sessions").json()["sessions"]

    assert [session["id"] for session in remaining] == [kept]
    assert client.delete("/api/sessions/nope").status_code == 404


def test_delete_session_refuses_while_pipeline_is_live(data_dir: Path, client) -> None:
    from therapy.server import live

    store = MemoryStore(data_dir)
    session_id = store.create_session()
    token = live.claim(session_id)
    try:
        assert client.delete(f"/api/sessions/{session_id}").status_code == 409
    finally:
        live.release(session_id, token)
    assert store.has_session(session_id)


def test_resumable_reflects_the_newest_session_freshness(data_dir: Path, client) -> None:
    assert client.get("/api/resumable").json() == {"session_id": None}

    store = MemoryStore(data_dir)
    session_id = store.create_session()
    store.add_turn(session_id, "user", "text", "es", "Hola.")
    store.end_session(session_id, "Greeted.")

    assert client.get("/api/resumable").json() == {"session_id": session_id}


def test_resolve_session_reports_what_the_connection_joins(tmp_path: Path) -> None:
    # The offer response carries this so the client loads the transcript over
    # HTTP instead of the data-channel replay, which pipecat drops when the
    # channel is slow to open on mobile (field test 2026-07-10).
    store = MemoryStore(tmp_path)
    resumable = store.create_session()
    store.add_turn(resumable, "user", "text", "es", "Hola.")
    store.end_session(resumable, "Greeted.")

    # Default connect resumes the recent session; new_session forces a fresh
    # one even though a resumable session exists.
    assert _resolve_session(store, new_session=False, explicit=None) == (
        resumable,
        True,
    )
    assert _resolve_session(store, new_session=True, explicit=None) == (None, False)

    # An explicit id is joined verbatim; an unknown one falls back to fresh.
    assert _resolve_session(store, new_session=False, explicit=resumable) == (
        resumable,
        True,
    )
    assert _resolve_session(store, new_session=False, explicit="nope") == (None, False)


def test_resolve_session_fresh_when_nothing_resumable(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    assert _resolve_session(store, new_session=False, explicit=None) == (None, False)


def test_rename_session_endpoint(data_dir: Path, client) -> None:
    store = MemoryStore(data_dir)
    session_id = store.create_session()

    ok = client.patch(
        f"/api/sessions/{session_id}", json={"title": "  Sueño y traba jo  "}
    )
    assert ok.status_code == 200
    assert ok.json()["title"] == "Sueño y traba jo"
    assert (
        client.patch(
            f"/api/sessions/{session_id}", json={"title": "  "}
        ).status_code
        == 422
    )
    assert client.patch("/api/sessions/nope", json={"title": "x"}).status_code == 404

    assert store.sessions()[0]["title"] == "Sueño y traba jo"
