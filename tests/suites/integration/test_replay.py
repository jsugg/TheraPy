"""Executable, network-free interaction journal replay integration tests."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import cast

import pytest

from therapy.observability.interactions import (
    InteractionRecord,
    InteractionRequest,
    InteractionResponse,
    JsonValue,
    Message,
    ProviderNative,
    RetrievedDocument,
    TranscriptTurn,
    Truncation,
)
from therapy.observability.journal import JournalStore
from therapy.observability.model import (
    InteractionEventKind,
    InteractionOperation,
    InteractionStatus,
    Provider,
)
from therapy.observability.replay import (
    ReplayVerificationError,
    reconstruct,
    replay_execute,
    replay_render,
)

_FIXTURE = Path("tests/fixtures/observability/interactions/ollama_success.json")


def _object(value: object, label: str) -> dict[str, object]:
    assert isinstance(value, dict), f"{label} must be an object"
    mapping = cast(dict[object, object], value)
    assert all(isinstance(key, str) for key in mapping), label
    return cast(dict[str, object], mapping)


def _json_object(value: object, label: str) -> dict[str, JsonValue]:
    return cast(dict[str, JsonValue], _object(value, label))


def _list(value: object, label: str) -> list[object]:
    assert isinstance(value, list), f"{label} must be a list"
    return cast(list[object], value)


def _text(value: object, label: str) -> str:
    assert isinstance(value, str), f"{label} must be text"
    return value


def _integer(value: object, label: str) -> int:
    assert isinstance(value, int), label
    assert not isinstance(value, bool), label
    return value


def _number(value: object, label: str) -> float:
    assert isinstance(value, int | float), label
    assert not isinstance(value, bool), label
    return float(value)


def _load_fixture(repo_root: Path) -> dict[str, object]:
    raw: object = json.loads((repo_root / _FIXTURE).read_text(encoding="utf-8"))
    return _object(raw, str(_FIXTURE))


def _request(payload: dict[str, object]) -> InteractionRequest:
    messages = tuple(
        Message(
            role=_text(row.get("role"), "message.role"),
            content=_text(row.get("content"), "message.content"),
        )
        for row in (
            _object(item, "request.messages item")
            for item in _list(payload.get("messages"), "request.messages")
        )
    )
    transcript = tuple(
        TranscriptTurn(
            role=_text(row.get("role"), "transcript.role"),
            language=_text(row.get("language"), "transcript.language"),
            modality=_text(row.get("modality"), "transcript.modality"),
            text=_text(row.get("text"), "transcript.text"),
        )
        for row in (
            _object(item, "request.transcript item")
            for item in _list(payload.get("transcript"), "request.transcript")
        )
    )
    retrieved = tuple(
        RetrievedDocument(
            source_type=_text(row.get("source_type"), "document.source_type"),
            source_id=_text(row.get("source_id"), "document.source_id"),
            anchor=_text(row.get("anchor"), "document.anchor"),
            score=_number(row.get("score"), "document.score"),
            rank=_integer(row.get("rank"), "document.rank"),
            text=_text(row.get("text"), "document.text"),
        )
        for row in (
            _object(item, "request.retrieved_documents item")
            for item in _list(
                payload.get("retrieved_documents"), "request.retrieved_documents"
            )
        )
    )
    tools = _list(payload.get("tools"), "request.tools")
    assert tools == [], "selected replay fixture must not declare tools"
    response_schema = payload.get("response_schema")
    assert response_schema is None
    truncation = _object(payload.get("truncation"), "request.truncation")
    memory_notes = tuple(
        _text(item, "request.memory_notes item")
        for item in _list(payload.get("memory_notes"), "request.memory_notes")
    )
    context_order = tuple(
        _text(item, "request.context_order item")
        for item in _list(payload.get("context_order"), "request.context_order")
    )
    return InteractionRequest(
        system_instructions=_text(
            payload.get("system_instructions"), "request.system_instructions"
        ),
        messages=messages,
        transcript=transcript,
        memory_notes=memory_notes,
        retrieved_documents=retrieved,
        parameters=_json_object(payload.get("parameters"), "request.parameters"),
        response_schema=None,
        context_order=context_order,
        truncation=Truncation(
            applied=bool(truncation.get("applied")),
            dropped_messages=_integer(
                truncation.get("dropped_messages"), "truncation.dropped_messages"
            ),
        ),
    )


def _started_record(fixture: dict[str, object]) -> InteractionRecord:
    record = _object(fixture.get("record"), "fixture.record")
    request = _object(record.get("request"), "fixture.record.request")
    native = _object(record.get("provider_native"), "fixture.record.provider_native")
    session_id = record.get("session_id")
    turn_id = record.get("turn_id")
    assert session_id is None or isinstance(session_id, str)
    assert turn_id is None or isinstance(turn_id, int)
    return InteractionRecord(
        interaction_id=_text(record.get("interaction_id"), "record.interaction_id"),
        trace_id=_text(record.get("trace_id"), "record.trace_id"),
        span_id=_text(record.get("span_id"), "record.span_id"),
        operation=InteractionOperation(
            _text(record.get("operation"), "record.operation")
        ),
        provider=Provider(_text(record.get("provider"), "record.provider")),
        requested_model=_text(record.get("requested_model"), "record.requested_model"),
        actual_model=_text(record.get("actual_model"), "record.actual_model"),
        prompt_template_version=_text(
            record.get("prompt_template_version"), "record.prompt_template_version"
        ),
        request=_request(request),
        response=InteractionResponse(),
        stream=(),
        error=None,
        provider_native=ProviderNative(
            request=_json_object(native.get("request"), "provider_native.request")
        ),
        language=_text(record.get("language"), "record.language"),
        modality=_text(record.get("modality"), "record.modality"),
        build_version=_text(record.get("build_version"), "record.build_version"),
        policy_version=_text(record.get("policy_version"), "record.policy_version"),
        config_version=_text(record.get("config_version"), "record.config_version"),
        started_at=_text(record.get("started_at"), "record.started_at"),
        completed_at=None,
        status=InteractionStatus.STARTED,
        session_id=session_id,
        turn_id=turn_id,
    )


def _journal_fixture(
    path: Path, repo_root: Path
) -> tuple[dict[str, object], dict[str, JsonValue]]:
    fixture = _load_fixture(repo_root)
    record_payload = _object(fixture.get("record"), "fixture.record")
    native = _object(
        record_payload.get("provider_native"), "fixture.record.provider_native"
    )
    record = _started_record(fixture)
    store = JournalStore(path)
    store.start_attempt(record)
    for sequence, raw_event in enumerate(
        _list(native.get("ordered_events"), "provider_native.ordered_events")
    ):
        store.append_stream_event(
            record.interaction_id,
            sequence,
            InteractionEventKind.PROVIDER_EVENT,
            f"fixture-{sequence}",
            _json_object(raw_event, "provider_native.ordered_events item"),
        )
    terminal_response = native.get("terminal_response")
    assert terminal_response is None or isinstance(terminal_response, dict)
    store.finish_success(
        record.interaction_id,
        {
            "response": _json_object(record_payload.get("response"), "record.response"),
            "provider_terminal": cast(JsonValue, terminal_response),
        },
        completed_at=_text(record_payload.get("completed_at"), "record.completed_at"),
    )
    loaded = store.load(record.interaction_id)
    assert loaded is not None
    row = _json_object(loaded.get("interaction"), "journal interaction row")
    store.close()
    return fixture, row


def test_fixture_attempt_round_trips_exact_requests(
    tmp_path: Path, repo_root: Path
) -> None:
    journal_path = tmp_path / "journal" / "interactions.sqlite3"
    fixture, row = _journal_fixture(journal_path, repo_root)
    interaction_id = _text(row.get("interaction_id"), "interaction_id")

    envelope = reconstruct(journal_path, interaction_id)
    rendered = replay_render(envelope)

    assert rendered.verified
    assert (
        rendered.canonical_json.encode()
        == _text(row.get("canonical_record_json"), "canonical_record_json").encode()
    )
    assert (
        rendered.canonical_request_json.encode()
        == _text(row.get("canonical_request_json"), "canonical_request_json").encode()
    )
    assert (
        rendered.provider_request_json.encode()
        == _text(row.get("provider_request_json"), "provider_request_json").encode()
    )
    fixture_record = _object(fixture.get("record"), "fixture.record")
    fixture_native = _object(
        fixture_record.get("provider_native"), "fixture.record.provider_native"
    )
    assert rendered.provider_request == _json_object(
        fixture_native.get("request"), "provider_native.request"
    )
    assert envelope.record["status"] == "succeeded"
    assert isinstance(envelope.record.get("terminal"), dict)

    executed: list[dict[str, JsonValue]] = []

    def fake_executor(request: dict[str, JsonValue]) -> str:
        executed.append(request)
        return "executed"

    assert replay_execute(envelope, fake_executor) == "executed"
    assert executed == [rendered.provider_request]


def test_tampered_canonical_row_fails_verification(
    tmp_path: Path, repo_root: Path
) -> None:
    journal_path = tmp_path / "tampered.sqlite3"
    fixture, row = _journal_fixture(journal_path, repo_root)
    fixture_record = _object(fixture.get("record"), "fixture.record")
    fixture_request = _object(fixture_record.get("request"), "fixture.record.request")
    original = _text(
        fixture_request.get("system_instructions"), "request.system_instructions"
    )
    with sqlite3.connect(journal_path) as connection:
        cursor = connection.execute(
            "UPDATE interactions SET canonical_record_json="
            "replace(canonical_record_json, ?, ?) WHERE interaction_id=?",
            (original, "tampered", row["interaction_id"]),
        )
        assert cursor.rowcount == 1

    envelope = reconstruct(
        journal_path, _text(row.get("interaction_id"), "interaction_id")
    )
    rendered = replay_render(envelope)
    assert rendered.checksum_verified is False
    assert rendered.verified is False

    executed: list[dict[str, JsonValue]] = []
    with pytest.raises(ReplayVerificationError, match="failed replay verification"):
        replay_execute(envelope, executed.append)
    assert executed == []


def test_tampered_provider_request_fails_exact_render(
    tmp_path: Path, repo_root: Path
) -> None:
    journal_path = tmp_path / "provider-tampered.sqlite3"
    _, row = _journal_fixture(journal_path, repo_root)
    with sqlite3.connect(journal_path) as connection:
        cursor = connection.execute(
            "UPDATE interactions SET provider_request_json=? WHERE interaction_id=?",
            ('{"model":"tampered"}', row["interaction_id"]),
        )
        assert cursor.rowcount == 1

    envelope = reconstruct(
        journal_path, _text(row.get("interaction_id"), "interaction_id")
    )
    rendered = replay_render(envelope)
    assert rendered.checksum_verified is True
    assert rendered.exact_match is False
    assert rendered.verified is False


def test_cli_emits_stdout_only_json_summary(
    tmp_path: Path, repo_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = pytest.importorskip("scripts.observability.replay_interaction")
    journal_path = tmp_path / "cli.sqlite3"
    _, row = _journal_fixture(journal_path, repo_root)

    result = cli.main(
        [
            "--journal",
            str(journal_path),
            "--interaction-id",
            _text(row.get("interaction_id"), "interaction_id"),
            "--json",
        ]
    )

    captured = capsys.readouterr()
    assert result == 0
    assert captured.err == ""
    assert json.loads(captured.out)["verified"] is True
