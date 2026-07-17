"""Complete CLI/API owner-data export, restore, and erasure."""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from therapy.data import DataSovereignty
from therapy.dialogue.outreach import ProactivityService
from therapy.knowledge.embeddings import EmbeddingMetadata
from therapy.knowledge.research import ResearchKB
from therapy.knowledge.user_model import UserModel
from therapy.memory import MemoryStore
from therapy.memory import __main__ as cli_module
from therapy.memory.__main__ import main

MetricCall = tuple[str, float, dict[str, str]]


class _CapturedSpan:
    """Minimal span double for finite migration-version assertions."""

    def __init__(self) -> None:
        self.attributes: dict[str, int | str] = {}

    def set_attribute(self, key: str, value: int | str) -> None:
        self.attributes[key] = value


@pytest.fixture
def metric_calls(monkeypatch: pytest.MonkeyPatch) -> list[MetricCall]:
    """Capture broad metric calls before the OTel manifest backend."""
    from therapy.observability import telemetry

    calls: list[MetricCall] = []

    def capture_metric(
        name: str, value: float, attrs: dict[str, str] | None = None
    ) -> None:
        calls.append((name, value, attrs or {}))

    monkeypatch.setattr(telemetry, "record_metric", capture_metric)
    return calls


@pytest.fixture
def broad_log() -> Iterator[io.StringIO]:
    """Capture fixed-schema broad events as their exported JSON form."""
    from therapy.observability.logging import BroadJsonFormatter

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(
        BroadJsonFormatter(service_version="0.1.0", environment="test")
    )
    logger = logging.getLogger("therapy.broad")
    previous_handlers = logger.handlers
    previous_level = logger.level
    previous_propagate = logger.propagate
    logger.handlers = [handler]
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    try:
        yield stream
    finally:
        logger.handlers = previous_handlers
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate


def _broad_events(stream: io.StringIO) -> list[dict[str, object]]:
    """Decode captured broad records."""
    return [json.loads(line) for line in stream.getvalue().splitlines() if line]


class ExportEmbedder:
    @property
    def metadata(self) -> EmbeddingMetadata:
        return EmbeddingMetadata("export-test", "v1", 2)

    @staticmethod
    def _vector(_text: str) -> np.ndarray:
        return np.asarray([1.0, 0.0], dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


def _seed(data_dir: Path) -> tuple[str, int]:
    store = MemoryStore(data_dir)
    session_id = store.create_session()
    store.add_turn(
        session_id,
        "user",
        "voice",
        "en",
        "I journaled today.",
        audio=b"\x00\x00\x01\x00",
    )
    store.end_session(session_id, "The user journaled today.")
    store.ensure_recap(session_id, "You journaled.")
    model = UserModel(data_dir)
    node_id = model.add_user_statement("identity_fact", "The user journals.")
    assert node_id is not None
    model.add_boundary("never_initiate", "private topic")
    research = ResearchKB(data_dir, embedder=ExportEmbedder())
    research.ingest("Journaling paper", "Doe 2025", "Journaling supports planning.")
    outreach = ProactivityService(data_dir, model=model, push_sender=lambda *_: None)
    outreach.update_settings(
        "greeting",
        enabled=True,
        timezone="UTC",
        quiet_start="22:00",
        quiet_end="08:00",
        schedule_time="18:00",
        schedule_day=6,
        frequency="weekly",
        topic=None,
    )
    return session_id, node_id


def test_cli_export_contains_all_tables_audio_corpus_and_private_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("THERAPY_DATA_DIR", str(tmp_path))
    session_id, _ = _seed(tmp_path)
    output = tmp_path / "owner-export.json"

    assert main(["export", "--output", str(output)]) == 0

    snapshot = json.loads(output.read_text(encoding="utf-8"))
    assert snapshot["format"] == "therapy-owner-data"
    assert snapshot["tables"]["sessions"][0]["id"] == session_id
    assert snapshot["tables"]["nodes"][0]["statement"] == "The user journals."
    assert snapshot["tables"]["research_documents"][0]["source_title"] == "Journaling paper"
    assert any(
        row["channel"] == "greeting" and row["enabled"] == 1
        for row in snapshot["tables"]["proactivity_settings"]
    )
    assert any(file["path"].endswith(".wav") for file in snapshot["files"])
    assert any(file["path"].startswith("research/sources/") for file in snapshot["files"])
    assert os.stat(output).st_mode & 0o777 == 0o600


def test_cli_delete_requires_confirmation_and_then_erases_every_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("THERAPY_DATA_DIR", str(tmp_path))
    _seed(tmp_path)
    (tmp_path / "therapy.db.pre-phase4-test.bak").write_bytes(b"private backup")

    assert main(["delete"]) == 2
    assert MemoryStore(tmp_path).sessions()
    assert main(["delete", "--yes"]) == 0

    assert "owner data and corpus" in capsys.readouterr().err
    assert MemoryStore(tmp_path).sessions() == []
    assert UserModel(tmp_path).nodes() == []
    assert ResearchKB(tmp_path, embedder=ExportEmbedder()).documents() == []
    assert all(not item["enabled"] for item in ProactivityService(tmp_path).settings())
    assert not (tmp_path / "audio").exists()
    assert not list((tmp_path / "research" / "sources").glob("*"))
    assert not list(tmp_path.glob("therapy.db*.bak"))


def test_cli_restore_round_trip_recovers_rows_audio_keys_and_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("THERAPY_DATA_DIR", str(tmp_path))
    session_id, node_id = _seed(tmp_path)
    vapid_public_key = ProactivityService(tmp_path).vapid.public_key()
    output = tmp_path / "snapshot.json"
    assert main(["export", "--output", str(output)]) == 0
    assert main(["delete", "--yes"]) == 0

    assert main(["restore", "--input", str(output)]) == 2
    assert main(["restore", "--input", str(output), "--yes"]) == 0

    store = MemoryStore(tmp_path)
    assert store.sessions()[0]["id"] == session_id
    assert store.turn_audio_path(session_id, store.session_turns(session_id)[0]["id"])
    assert UserModel(tmp_path).get_node(node_id)["statement"] == "The user journals."
    assert ResearchKB(tmp_path, embedder=ExportEmbedder()).documents()[0]["source_ref"] == "Doe 2025"
    assert ProactivityService(tmp_path).vapid.public_key() == vapid_public_key
    greeting = next(
        item for item in ProactivityService(tmp_path).settings() if item["channel"] == "greeting"
    )
    assert greeting["enabled"] is True


def test_invalid_restore_is_rejected_before_current_data_changes(tmp_path: Path) -> None:
    session_id, _ = _seed(tmp_path)
    service = DataSovereignty(tmp_path)
    snapshot = service.export_snapshot()
    snapshot["tables"]["nodes"][0]["unknown_column"] = "bad"

    with pytest.raises(ValueError, match="invalid columns"):
        service.restore_snapshot(snapshot)

    assert MemoryStore(tmp_path).sessions()[0]["id"] == session_id


def test_restore_rejects_incompatible_knowledge_schema(tmp_path: Path) -> None:
    session_id, _ = _seed(tmp_path)
    service = DataSovereignty(tmp_path)
    snapshot = service.export_snapshot()
    snapshot["knowledge_schema_version"] = 999

    with pytest.raises(ValueError, match="knowledge schema version"):
        service.restore_snapshot(snapshot)

    assert MemoryStore(tmp_path).sessions()[0]["id"] == session_id


@pytest.mark.parametrize("category", ["format", "schema", "table", "file"])
def test_restore_rejection_categories_are_finite_and_private(
    category: Literal["format", "schema", "table", "file"],
    tmp_path: Path,
    metric_calls: list[MetricCall],
    broad_log: io.StringIO,
) -> None:
    service = DataSovereignty(tmp_path)
    snapshot = service.export_snapshot()
    canary = f"submitted-{category}-secret"
    if category == "format":
        snapshot["format"] = canary
    elif category == "schema":
        snapshot["knowledge_schema_version"] = 999_999
        canary = "999999"
    elif category == "table":
        snapshot["tables"] = {canary: []}
    else:
        snapshot["files"] = [
            {"path": f"../{canary}", "size": 0, "content_base64": ""}
        ]

    with pytest.raises(ValueError, match="unsupported|snapshot"):
        service.restore_snapshot(snapshot)

    assert (
        "therapy_sovereignty_stages_total",
        1,
        {"operation": "restore", "stage": "validate", "outcome": "rejected"},
    ) in metric_calls
    event = next(
        item
        for item in _broad_events(broad_log)
        if item["event.name"] == "sovereignty.restore_rejected"
    )
    assert event["outcome"] == category
    assert event["outcome"] in {"format", "schema", "table", "file"}
    assert canary not in broad_log.getvalue()
    assert str(tmp_path) not in broad_log.getvalue()


def test_schema_migration_and_backup_emit_bounded_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metric_calls: list[MetricCall],
) -> None:
    from therapy.knowledge import schema
    from therapy.observability import telemetry

    spans: list[tuple[str, _CapturedSpan]] = []

    @contextmanager
    def capture_span(
        name: str, *, component: str, operation: str
    ) -> Iterator[_CapturedSpan]:
        assert component == "schema"
        assert operation in {"migrate", "backup"}
        span = _CapturedSpan()
        spans.append((name, span))
        yield span

    monkeypatch.setattr(telemetry, "broad_span", capture_span)

    fresh_database = tmp_path / "fresh.db"
    assert schema.migrate_database(fresh_database) is None

    migration_outcomes = {
        attrs["outcome"]
        for name, _, attrs in metric_calls
        if name == "therapy_schema_migrations_total"
    }
    duration_outcomes = {
        attrs["outcome"]
        for name, value, attrs in metric_calls
        if name == "therapy_schema_migration_seconds" and value >= 0
    }
    assert migration_outcomes == {"success"}
    assert duration_outcomes == {"success"}
    version_attributes = [
        span.attributes
        for name, span in spans
        if name in {"schema.migrate", "schema.migrate.step"}
    ]
    assert version_attributes
    for attributes in version_attributes:
        assert set(attributes) == {"schema.from_version", "schema.to_version"}
        assert all(
            isinstance(value, int) and 0 <= value <= schema.SCHEMA_VERSION + 1
            for value in attributes.values()
        )
    assert (
        "therapy_backups_total",
        1,
        {"outcome": "skipped"},
    ) in metric_calls

    legacy_database = tmp_path / "legacy.db"
    with sqlite3.connect(legacy_database) as connection:
        connection.execute("CREATE TABLE nodes (id INTEGER PRIMARY KEY)")
        connection.execute(
            """
            CREATE TABLE schema_migrations (
                component TEXT PRIMARY KEY,
                version INTEGER NOT NULL,
                applied_at TEXT NOT NULL
            )
            """
        )
        backup = schema._backup_if_needed(legacy_database, connection)

    assert backup is not None
    assert backup.is_file()
    assert (
        "therapy_backups_total",
        1,
        {"outcome": "created"},
    ) in metric_calls
    for name, _, attrs in metric_calls:
        if name == "therapy_schema_migrations_total":
            assert attrs["outcome"] in {"success", "error", "rolled_back"}
        elif name == "therapy_schema_migration_seconds":
            assert attrs["outcome"] in {"success", "error"}
        elif name == "therapy_backups_total":
            assert attrs["outcome"] in {"created", "skipped", "error"}


def test_schema_migration_and_backup_failures_are_critical_and_private(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metric_calls: list[MetricCall],
    broad_log: io.StringIO,
) -> None:
    from therapy.knowledge import schema

    migration_canary = "migration-secret-message"

    def fail_migration(_connection: sqlite3.Connection) -> None:
        raise RuntimeError(migration_canary)

    monkeypatch.setattr(
        schema, "MIGRATIONS", (schema.Migration(version=1, apply=fail_migration),)
    )
    with pytest.raises(RuntimeError, match=migration_canary):
        schema.migrate_database(tmp_path / "migration-failure.db")

    backup_database = tmp_path / "backup-failure.db"
    with sqlite3.connect(backup_database) as connection:
        connection.execute("CREATE TABLE nodes (id INTEGER PRIMARY KEY)")
    backup_canary = "backup-secret-message"

    def fail_backup(_source: Path, _destination: Path) -> None:
        raise OSError(backup_canary)

    monkeypatch.setattr(schema.shutil, "copy2", fail_backup)
    with pytest.raises(OSError, match=backup_canary):
        schema.migrate_database(backup_database)

    assert (
        "therapy_schema_migrations_total",
        1,
        {"outcome": "error"},
    ) in metric_calls
    assert (
        "therapy_schema_migrations_total",
        1,
        {"outcome": "rolled_back"},
    ) in metric_calls
    assert (
        "therapy_backups_total",
        1,
        {"outcome": "error"},
    ) in metric_calls
    events = _broad_events(broad_log)
    assert {
        (event["operation"], event["severity"], event["error.type"])
        for event in events
    } >= {
        ("migrate", "CRITICAL", "RuntimeError"),
        ("backup", "CRITICAL", "OSError"),
    }
    exported = broad_log.getvalue()
    assert migration_canary not in exported
    assert backup_canary not in exported
    assert str(tmp_path) not in exported


def test_sovereignty_success_stages_are_complete_and_bounded(
    tmp_path: Path, metric_calls: list[MetricCall]
) -> None:
    from therapy import data as data_module

    service = DataSovereignty(tmp_path)
    snapshot = json.loads(service.export_json())
    service.restore_snapshot(snapshot)
    service.delete_all()

    stage_attrs = [
        attrs
        for name, _, attrs in metric_calls
        if name == "therapy_sovereignty_stages_total"
    ]
    assert {
        attrs["stage"]
        for attrs in stage_attrs
        if attrs["operation"] == "export" and attrs["outcome"] == "success"
    } == {"tables", "files", "json"}
    assert {
        attrs["stage"]
        for attrs in stage_attrs
        if attrs["operation"] == "delete" and attrs["outcome"] == "success"
    } == data_module._DELETE_STAGES
    restored_tables = {
        attrs["stage"]
        for attrs in stage_attrs
        if attrs["operation"] == "restore"
        and attrs["stage"].startswith("table_")
        and attrs["outcome"] == "success"
    }
    assert restored_tables == {f"table_{table}" for table in data_module._TABLES}
    for attrs in stage_attrs:
        operation = attrs["operation"]
        if operation == "export":
            allowed_stages = data_module._EXPORT_STAGES
        elif operation == "restore":
            allowed_stages = data_module._RESTORE_STAGES
        else:
            assert operation == "delete"
            allowed_stages = data_module._DELETE_STAGES
        assert attrs["stage"] in allowed_stages
        assert attrs["outcome"] in {"success", "error", "timeout", "rejected"}
        assert str(tmp_path) not in json.dumps(attrs)


def test_invalid_snapshot_is_categorized_and_rollback_cleanup_preserves_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metric_calls: list[MetricCall],
    broad_log: io.StringIO,
) -> None:
    from therapy import data as data_module

    service = DataSovereignty(tmp_path)
    session_id = service.store.create_session()
    snapshot = service.export_snapshot()
    tables = snapshot["tables"]
    assert isinstance(tables, dict)
    sessions = tables["sessions"]
    assert isinstance(sessions, list)
    sessions.append(dict(sessions[0]))
    cleanup_canary = "cleanup-secret-message"
    original_rmtree = shutil.rmtree

    def fail_staged_cleanup(
        path: str | Path, ignore_errors: bool = False
    ) -> None:
        if Path(path).name.startswith(".restore-"):
            raise OSError(cleanup_canary)
        original_rmtree(path, ignore_errors=ignore_errors)

    monkeypatch.setattr(data_module.shutil, "rmtree", fail_staged_cleanup)
    with pytest.raises(ValueError, match="database constraints"):
        service.restore_snapshot(snapshot)

    assert service.store.sessions()[0]["id"] == session_id
    rollback_outcomes = [
        attrs["outcome"]
        for name, _, attrs in metric_calls
        if name == "therapy_sovereignty_rollbacks_total"
    ]
    assert rollback_outcomes == ["attempted", "succeeded"]
    assert (
        "therapy_sovereignty_stages_total",
        1,
        {"operation": "restore", "stage": "table_sessions", "outcome": "rejected"},
    ) in metric_calls
    events = _broad_events(broad_log)
    rejection = next(
        event
        for event in events
        if event["event.name"] == "sovereignty.restore_rejected"
    )
    assert rejection["outcome"] == "table"
    exported = broad_log.getvalue()
    assert cleanup_canary not in exported
    assert str(tmp_path) not in exported
    assert session_id not in exported


def test_partial_delete_is_critical_and_has_bounded_error_outcome(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metric_calls: list[MetricCall],
    broad_log: io.StringIO,
) -> None:
    service = DataSovereignty(tmp_path)
    canary = "delete-secret-message"

    def fail_research_delete() -> None:
        raise RuntimeError(canary)

    monkeypatch.setattr(service.research, "delete_all", fail_research_delete)
    with pytest.raises(RuntimeError, match=canary):
        service.delete_all()

    assert (
        "therapy_sovereignty_stages_total",
        1,
        {"operation": "delete", "stage": "proactivity", "outcome": "success"},
    ) in metric_calls
    assert (
        "therapy_sovereignty_stages_total",
        1,
        {"operation": "delete", "stage": "research", "outcome": "error"},
    ) in metric_calls
    event = next(
        item
        for item in _broad_events(broad_log)
        if item["event.name"] == "sovereignty.delete_partial"
    )
    assert event["severity"] == "CRITICAL"
    assert event["operation"] == "delete"
    assert event["outcome"] == "partial"
    assert event["error.type"] == "RuntimeError"
    exported = broad_log.getvalue()
    assert canary not in exported
    assert str(tmp_path) not in exported


def test_product_storage_inspection_is_bounded_and_failure_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metric_calls: list[MetricCall],
) -> None:
    from therapy.observability import capture

    (tmp_path / "therapy.db").write_bytes(b"db")
    (tmp_path / "therapy.db-wal").write_bytes(b"wal")
    (tmp_path / "audio" / "nested").mkdir(parents=True)
    (tmp_path / "audio" / "nested" / "turn.wav").write_bytes(b"audio")
    (tmp_path / "research").mkdir()
    (tmp_path / "research" / "source.bin").write_bytes(b"research")
    (tmp_path / "models" / "embeddings").mkdir(parents=True)
    (tmp_path / "models" / "embeddings" / "model.bin").write_bytes(b"model")
    (tmp_path / "therapy.db.pre-phase4-test.bak").write_bytes(b"backup")
    monkeypatch.setenv("THERAPY_DATA_DIR", str(tmp_path))

    capture._inspect_product_storage()

    sizes = {
        attrs["kind"]: value
        for name, value, attrs in metric_calls
        if name == "therapy_data_bytes"
    }
    assert sizes == {
        "db": 2,
        "wal": 3,
        "audio": 5,
        "research": 8,
        "model_cache": 5,
        "backups": 6,
    }
    assert any(name == "therapy_disk_free_bytes" for name, _, _ in metric_calls)
    assert (
        "therapy_storage_inspections_total",
        1,
        {"outcome": "success"},
    ) in metric_calls
    assert set(sizes) == {
        "db",
        "wal",
        "audio",
        "research",
        "model_cache",
        "backups",
    }

    metric_calls.clear()
    monkeypatch.setenv("THERAPY_DATA_DIR", str(tmp_path / "missing"))
    capture._inspect_product_storage()
    assert (
        "therapy_storage_inspections_total",
        1,
        {"outcome": "error"},
    ) in metric_calls
    for name, _, attrs in metric_calls:
        if name == "therapy_data_bytes":
            assert attrs["kind"] in sizes
        elif name == "therapy_storage_inspections_total":
            assert attrs["outcome"] in {"success", "error"}


def test_api_export_delete_and_restore_round_trip(data_dir: Path, client) -> None:
    session_id, _ = _seed(data_dir)
    exported = client.get("/api/data/export")
    assert exported.status_code == 200
    assert "attachment" in exported.headers["content-disposition"]

    assert client.request("DELETE", "/api/data", json={"confirmation": "no"}).status_code == 422
    assert client.request(
        "DELETE", "/api/data", json={"confirmation": "DELETE EVERYTHING"}
    ).status_code == 200
    assert MemoryStore(data_dir).sessions() == []

    restored = client.post(
        "/api/data/restore",
        files={"file": ("snapshot.json", exported.content, "application/json")},
    )
    assert restored.status_code == 200
    assert MemoryStore(data_dir).sessions()[0]["id"] == session_id


def test_research_cli_ingest_preview_correct_reindex_list_and_delete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("THERAPY_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(
        cli_module,
        "ResearchKB",
        lambda: ResearchKB(tmp_path, embedder=ExportEmbedder()),
    )
    source = tmp_path / "planning.md"
    source.write_text(
        "# Planning transitions\n\nA visible checklist reduces planning load.",
        encoding="utf-8",
    )

    assert main(
        [
            "research-ingest",
            str(source),
            "--title",
            "Planning transitions",
            "--ref",
            "Owner guide",
        ]
    ) == 0
    document_id = json.loads(capsys.readouterr().out)["document_id"]

    assert main(["research-list"]) == 0
    listed = json.loads(capsys.readouterr().out)["documents"]
    assert listed[0]["id"] == document_id

    assert main(["research-show", str(document_id)]) == 0
    document = json.loads(capsys.readouterr().out)["document"]
    anchor = document["blocks"][0]["anchor"]
    assert anchor == "section-planning-transitions-block-1"

    corrected = "A visible checklist can reduce planning load."
    assert main(
        ["research-correct", str(document_id), anchor, "--text", corrected]
    ) == 0
    assert json.loads(capsys.readouterr().out)["document"]["blocks"][0]["text"] == corrected

    assert main(["research-reindex", str(document_id)]) == 0
    assert json.loads(capsys.readouterr().out)["chunks_indexed"] == 1
    assert main(["research-delete", str(document_id)]) == 2
    assert main(["research-delete", str(document_id), "--yes"]) == 0
    assert json.loads(capsys.readouterr().out)["deleted"] == document_id
    assert ResearchKB(tmp_path, embedder=ExportEmbedder()).documents() == []
