"""One-command export, validated restore, and complete Phase 4 data erasure."""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal
from uuid import uuid4

from therapy.dialogue.outreach import ProactivityService
from therapy.knowledge.research import ResearchKB
from therapy.knowledge.schema import SCHEMA_VERSION, migrate_database
from therapy.knowledge.user_model import UserModel
from therapy.memory import MemoryStore

EXPORT_FORMAT: Final = "therapy-owner-data"
EXPORT_VERSION: Final = 1
MAX_RESTORE_BYTES: Final = 512 * 1024 * 1024

_TABLES: Final[tuple[str, ...]] = (
    "sessions",
    "turns",
    "nodes",
    "edges",
    "claim_evidence",
    "lifecycle_events",
    "node_aliases",
    "observation_inbox",
    "distillation_runs",
    "boundaries",
    "tombstones",
    "privacy_purge_events",
    "pending_insights",
    "insight_history",
    "semantic_embeddings",
    "proactivity_settings",
    "proactivity_jobs",
    "push_subscriptions",
    "in_app_messages",
    "digests",
    "research_documents",
    "research_blocks",
    "research_index",
)
_PRIVATE_FILES: Final = {"tombstone.key", "vapid-private.pem"}
_PRIVATE_PREFIXES: Final = ("audio/", "research/sources/")

type SovereigntyOperation = Literal["export", "restore", "delete"]
type SovereigntyOutcome = Literal["success", "error", "timeout", "rejected"]
type SnapshotRejectionCategory = Literal["format", "schema", "table", "file"]

_EXPORT_STAGES: Final = frozenset({"tables", "files", "json"})
_RESTORE_STAGES: Final = frozenset(
    {
        "validate",
        "stage_files",
        "backup_database",
        "backup_files",
        "database_clear",
        "replace_files",
        "refresh_keys",
        "rollback_database",
        "rollback_files",
        "rollback_refresh_keys",
        "cleanup_database_backup",
        "cleanup_files_backup",
        "cleanup_staged_files",
        *(f"table_{table}" for table in _TABLES),
    }
)
_DELETE_STAGES: Final = frozenset(
    {
        "proactivity",
        "research",
        "model",
        "sessions_audio",
        "private_files_backups",
        "wal_checkpoint",
        "vacuum",
    }
)
_SOVEREIGNTY_STAGES: Final[dict[SovereigntyOperation, frozenset[str]]] = {
    "export": _EXPORT_STAGES,
    "restore": _RESTORE_STAGES,
    "delete": _DELETE_STAGES,
}


class _SnapshotRejection(ValueError):
    """Validated owner-snapshot rejection with a finite broad category."""

    def __init__(self, category: SnapshotRejectionCategory, message: str) -> None:
        super().__init__(message)
        self.category = category


@contextmanager
def _sovereignty_stage(
    operation: SovereigntyOperation, stage: str
) -> Iterator[object | None]:
    """Trace and count one registered sovereignty stage without content labels."""
    if stage not in _SOVEREIGNTY_STAGES[operation]:
        raise ValueError("unregistered sovereignty stage")

    from therapy.observability.logging import emit_event
    from therapy.observability.telemetry import broad_span, record_metric

    outcome: SovereigntyOutcome = "success"
    with broad_span(
        f"sovereignty.{operation}.{stage}",
        component="sovereignty",
        operation=operation,
    ) as span:
        try:
            yield span
        except _SnapshotRejection as exc:
            outcome = "rejected"
            setter = getattr(span, "set_attribute", None)
            if callable(setter):
                setter("rejection.category", exc.category)
            emit_event(
                "sovereignty.restore_rejected",
                severity=logging.WARNING,
                component="sovereignty",
                operation="restore",
                outcome=exc.category,
            )
            raise
        except TimeoutError:
            outcome = "timeout"
            raise
        except Exception:
            outcome = "error"
            raise
        finally:
            record_metric(
                "therapy_sovereignty_stages_total",
                1,
                {"operation": operation, "stage": stage, "outcome": outcome},
            )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _encode(value: object) -> object:
    if isinstance(value, bytes):
        return {"$base64": base64.b64encode(value).decode("ascii")}
    if value is None or isinstance(value, (str, int, float)):
        return value
    raise TypeError(f"Unsupported SQLite value: {type(value).__name__}")


def _decode(value: object) -> object:
    if isinstance(value, dict) and set(value) == {"$base64"}:
        payload = value["$base64"]
        if not isinstance(payload, str):
            raise ValueError("invalid base64 database value")
        try:
            return base64.b64decode(payload, validate=True)
        except ValueError as exc:
            raise ValueError("invalid base64 database value") from exc
    if (
        value is None
        or isinstance(value, (str, int, float))
        and not isinstance(value, bool)
    ):
        return value
    raise ValueError("snapshot row contains unsupported value")


def _allowed_file(relative: str) -> bool:
    return relative in _PRIVATE_FILES or relative.startswith(_PRIVATE_PREFIXES)


class DataSovereignty:
    """Coordinate every personal store without leaking framework concerns inward."""

    def __init__(
        self,
        data_dir: Path | None = None,
        *,
        store: MemoryStore | None = None,
        model: UserModel | None = None,
        research: ResearchKB | None = None,
        proactivity: ProactivityService | None = None,
    ) -> None:
        self.data_dir = Path(
            data_dir or os.getenv("THERAPY_DATA_DIR", str(Path("./data")))
        )
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "therapy.db"
        migrate_database(self.db_path)
        self.store = store or MemoryStore(self.data_dir)
        self.model = model or UserModel(self.data_dir)
        self.research = research or ResearchKB(self.data_dir)
        self.proactivity = proactivity or ProactivityService(
            self.data_dir, model=self.model
        )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=30000")
            yield connection
        finally:
            connection.close()

    def export_snapshot(self) -> dict[str, object]:
        """Return complete inspectable personal rows and owner-owned source/audio files."""
        with _sovereignty_stage("export", "tables"):
            with self._connect() as connection:
                tables = {
                    table: [
                        {key: _encode(value) for key, value in dict(row).items()}
                        for row in connection.execute(f"SELECT * FROM {table}")
                    ]
                    for table in _TABLES
                }
        files: list[dict[str, object]] = []
        with _sovereignty_stage("export", "files"):
            for path in sorted(self.data_dir.rglob("*")):
                if not path.is_file() or path.is_symlink():
                    continue
                relative = path.relative_to(self.data_dir).as_posix()
                if not _allowed_file(relative):
                    continue
                payload = path.read_bytes()
                files.append(
                    {
                        "path": relative,
                        "size": len(payload),
                        "content_base64": base64.b64encode(payload).decode("ascii"),
                    }
                )
        return {
            "format": EXPORT_FORMAT,
            "version": EXPORT_VERSION,
            "knowledge_schema_version": SCHEMA_VERSION,
            "exported_at": _utc_now(),
            "tables": tables,
            "files": files,
        }

    def export_json(self) -> bytes:
        """Encode a deterministic UTF-8 owner snapshot."""
        with _sovereignty_stage("export", "json"):
            return json.dumps(
                self.export_snapshot(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")

    def _validate_snapshot(
        self, snapshot: Mapping[str, object]
    ) -> tuple[dict[str, list[dict[str, object]]], list[tuple[str, bytes]]]:
        with _sovereignty_stage("restore", "validate"):
            if (
                snapshot.get("format") != EXPORT_FORMAT
                or snapshot.get("version") != EXPORT_VERSION
            ):
                raise _SnapshotRejection(
                    "format", "unsupported TheraPy export format/version"
                )
            if snapshot.get("knowledge_schema_version") != SCHEMA_VERSION:
                raise _SnapshotRejection(
                    "schema", "unsupported knowledge schema version"
                )
            raw_tables = snapshot.get("tables")
            if not isinstance(raw_tables, dict) or set(raw_tables) != set(_TABLES):
                raise _SnapshotRejection(
                    "table", "snapshot does not contain the complete table set"
                )
            with self._connect() as connection:
                columns = {
                    table: {
                        str(row["name"])
                        for row in connection.execute(f"PRAGMA table_info({table})")
                    }
                    for table in _TABLES
                }
            tables: dict[str, list[dict[str, object]]] = {}
            for table in _TABLES:
                raw_rows = raw_tables[table]
                if not isinstance(raw_rows, list):
                    raise _SnapshotRejection(
                        "table", f"snapshot table {table} must be a list"
                    )
                rows: list[dict[str, object]] = []
                for raw_row in raw_rows:
                    if (
                        not isinstance(raw_row, dict)
                        or not set(raw_row) <= columns[table]
                    ):
                        raise _SnapshotRejection(
                            "table", f"snapshot table {table} has invalid columns"
                        )
                    try:
                        rows.append(
                            {
                                str(key): _decode(value)
                                for key, value in raw_row.items()
                            }
                        )
                    except ValueError as exc:
                        raise _SnapshotRejection(
                            "table", "snapshot table contains an invalid value"
                        ) from exc
                tables[table] = rows
            raw_files = snapshot.get("files")
            if not isinstance(raw_files, list):
                raise _SnapshotRejection("file", "snapshot files must be a list")
            files: list[tuple[str, bytes]] = []
            total = 0
            seen: set[str] = set()
            for item in raw_files:
                if not isinstance(item, dict):
                    raise _SnapshotRejection(
                        "file", "snapshot file entry must be an object"
                    )
                relative = item.get("path")
                encoded = item.get("content_base64")
                if (
                    not isinstance(relative, str)
                    or relative in seen
                    or not _allowed_file(relative)
                    or Path(relative).is_absolute()
                    or ".." in Path(relative).parts
                    or not isinstance(encoded, str)
                ):
                    raise _SnapshotRejection(
                        "file", "snapshot contains unsafe file path"
                    )
                try:
                    payload = base64.b64decode(encoded, validate=True)
                except ValueError as exc:
                    raise _SnapshotRejection(
                        "file", "snapshot file is not valid base64"
                    ) from exc
                if item.get("size") != len(payload):
                    raise _SnapshotRejection(
                        "file", "snapshot file size does not match payload"
                    )
                total += len(payload)
                if total > MAX_RESTORE_BYTES:
                    raise _SnapshotRejection(
                        "file", "snapshot files exceed restore size limit"
                    )
                seen.add(relative)
                files.append((relative, payload))
            return tables, files

    def restore_snapshot(self, snapshot: Mapping[str, object]) -> dict[str, int]:
        """Validate fully, then replace personal rows/files with rollback backups."""
        tables, files = self._validate_snapshot(snapshot)
        token = uuid4().hex
        database_backup = self.data_dir / f".pre-restore-{token}.db"
        files_backup = self.data_dir / f".pre-restore-{token}-files"
        staged = self.data_dir / f".restore-{token}"
        database_backup_ready = False
        files_backup_ready = False
        primary_error: Exception | None = None
        try:
            with _sovereignty_stage("restore", "stage_files"):
                staged.mkdir(mode=0o700)
                for relative, payload in files:
                    destination = staged / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    descriptor = os.open(
                        destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                    )
                    with os.fdopen(descriptor, "wb") as handle:
                        handle.write(payload)
            with _sovereignty_stage("restore", "backup_database"):
                with (
                    self._connect() as source,
                    sqlite3.connect(database_backup) as target,
                ):
                    source.backup(target)
                database_backup_ready = True
            with _sovereignty_stage("restore", "backup_files"):
                files_backup.mkdir(mode=0o700)
                self._backup_private_files(files_backup)
                files_backup_ready = True
            with self._connect() as connection:
                with connection:
                    with _sovereignty_stage("restore", "database_clear"):
                        for table in reversed(_TABLES):
                            connection.execute(f"DELETE FROM {table}")
                    for table in _TABLES:
                        with _sovereignty_stage("restore", f"table_{table}"):
                            try:
                                for row in tables[table]:
                                    if not row:
                                        continue
                                    names = tuple(row)
                                    placeholders = ",".join("?" for _ in names)
                                    connection.execute(
                                        f"INSERT INTO {table} ({','.join(names)}) "
                                        f"VALUES ({placeholders})",
                                        tuple(row[name] for name in names),
                                    )
                            except sqlite3.IntegrityError as exc:
                                raise _SnapshotRejection(
                                    "table",
                                    "snapshot table rows violate database constraints",
                                ) from exc
            with _sovereignty_stage("restore", "replace_files"):
                self._remove_private_files()
                for path in staged.rglob("*"):
                    if not path.is_file():
                        continue
                    destination = self.data_dir / path.relative_to(staged)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    path.replace(destination)
            with _sovereignty_stage("restore", "refresh_keys"):
                self._refresh_keys()
            return {
                "rows_restored": sum(len(rows) for rows in tables.values()),
                "files_restored": len(files),
            }
        except Exception as exc:
            primary_error = exc
            from therapy.observability.logging import emit_event
            from therapy.observability.telemetry import record_metric

            record_metric(
                "therapy_sovereignty_rollbacks_total",
                1,
                {"outcome": "attempted"},
            )
            rollback_errors: list[Exception] = []
            if database_backup_ready:
                try:
                    with _sovereignty_stage("restore", "rollback_database"):
                        with (
                            sqlite3.connect(database_backup) as source,
                            self._connect() as target,
                        ):
                            source.backup(target)
                except Exception as rollback_error:
                    rollback_errors.append(rollback_error)
            if files_backup_ready:
                try:
                    with _sovereignty_stage("restore", "rollback_files"):
                        self._remove_private_files()
                        self._restore_private_files(files_backup)
                except Exception as rollback_error:
                    rollback_errors.append(rollback_error)
            try:
                with _sovereignty_stage("restore", "rollback_refresh_keys"):
                    self._refresh_keys()
            except Exception as rollback_error:
                rollback_errors.append(rollback_error)
            if rollback_errors:
                record_metric(
                    "therapy_sovereignty_rollbacks_total",
                    1,
                    {"outcome": "failed"},
                )
                emit_event(
                    "sovereignty.restore_rollback_failed",
                    severity=logging.CRITICAL,
                    component="sovereignty",
                    operation="restore",
                    outcome="error",
                    error_type=type(rollback_errors[0]).__name__,
                )
            else:
                record_metric(
                    "therapy_sovereignty_rollbacks_total",
                    1,
                    {"outcome": "succeeded"},
                )
            raise
        finally:
            cleanup_error: Exception | None = None
            cleanup_stages = (
                ("cleanup_database_backup", database_backup, False),
                ("cleanup_files_backup", files_backup, True),
                ("cleanup_staged_files", staged, True),
            )
            for stage, path, is_directory in cleanup_stages:
                try:
                    with _sovereignty_stage("restore", stage):
                        if is_directory:
                            if path.exists():
                                shutil.rmtree(path)
                        else:
                            path.unlink(missing_ok=True)
                except Exception as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
            if cleanup_error is not None and primary_error is None:
                raise cleanup_error

    def _backup_private_files(self, destination: Path) -> None:
        for path in self.data_dir.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(self.data_dir).as_posix()
            if _allowed_file(relative):
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)

    def _restore_private_files(self, source: Path) -> None:
        for path in source.rglob("*"):
            if path.is_file():
                destination = self.data_dir / path.relative_to(source)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)

    def _remove_private_files(self) -> None:
        shutil.rmtree(self.data_dir / "audio", ignore_errors=True)
        shutil.rmtree(self.data_dir / "research", ignore_errors=True)
        for filename in _PRIVATE_FILES:
            (self.data_dir / filename).unlink(missing_ok=True)

    def _refresh_keys(self) -> None:
        # Existing services may survive an API restore; refresh their keyed digest state.
        self.model.reload_tombstone_key()

    def _delete_private_files_and_backups(self) -> None:
        """Remove non-database private artifacts and migration backups."""
        self._remove_private_files()
        for backup in self.data_dir.glob("therapy.db*.bak"):
            backup.unlink(missing_ok=True)

    def _checkpoint_wal(self) -> None:
        """Truncate the product database WAL after destructive erasure."""
        with self._connect() as connection:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def _vacuum(self) -> None:
        """Rewrite the product database after destructive erasure."""
        with self._connect() as connection:
            connection.execute("VACUUM")

    def delete_all(self) -> None:
        """Erase sessions/audio, graph/provenance, outreach, corpus, keys, and backups."""
        from therapy.observability.logging import emit_event

        stages = (
            ("proactivity", self.proactivity.delete_all),
            ("research", self.research.delete_all),
            ("model", self.model.delete_all),
            ("sessions_audio", self.store.delete_all),
            ("private_files_backups", self._delete_private_files_and_backups),
            ("wal_checkpoint", self._checkpoint_wal),
            ("vacuum", self._vacuum),
        )
        succeeded = 0
        for stage, action in stages:
            try:
                with _sovereignty_stage("delete", stage):
                    action()
            except Exception as exc:
                if succeeded:
                    emit_event(
                        "sovereignty.delete_partial",
                        severity=logging.CRITICAL,
                        component="sovereignty",
                        operation="delete",
                        outcome="partial",
                        error_type=type(exc).__name__,
                    )
                raise
            succeeded += 1


__all__ = ["DataSovereignty", "EXPORT_FORMAT", "EXPORT_VERSION", "MAX_RESTORE_BYTES"]
