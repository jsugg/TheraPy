"""One-command export, validated restore, and complete Phase 4 data erasure."""

from __future__ import annotations

import base64
import json
import os
import shutil
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Final
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
        with self._connect() as connection:
            tables = {
                table: [
                    {key: _encode(value) for key, value in dict(row).items()}
                    for row in connection.execute(f"SELECT * FROM {table}")
                ]
                for table in _TABLES
            }
        files: list[dict[str, object]] = []
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
        return json.dumps(
            self.export_snapshot(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def _validate_snapshot(
        self, snapshot: Mapping[str, object]
    ) -> tuple[dict[str, list[dict[str, object]]], list[tuple[str, bytes]]]:
        if (
            snapshot.get("format") != EXPORT_FORMAT
            or snapshot.get("version") != EXPORT_VERSION
        ):
            raise ValueError("unsupported TheraPy export format/version")
        if snapshot.get("knowledge_schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported knowledge schema version")
        raw_tables = snapshot.get("tables")
        if not isinstance(raw_tables, dict) or set(raw_tables) != set(_TABLES):
            raise ValueError("snapshot does not contain the complete table set")
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
                raise ValueError(f"snapshot table {table} must be a list")
            rows: list[dict[str, object]] = []
            for raw_row in raw_rows:
                if not isinstance(raw_row, dict) or not set(raw_row) <= columns[table]:
                    raise ValueError(f"snapshot table {table} has invalid columns")
                rows.append(
                    {str(key): _decode(value) for key, value in raw_row.items()}
                )
            tables[table] = rows
        raw_files = snapshot.get("files")
        if not isinstance(raw_files, list):
            raise ValueError("snapshot files must be a list")
        files: list[tuple[str, bytes]] = []
        total = 0
        seen: set[str] = set()
        for item in raw_files:
            if not isinstance(item, dict):
                raise ValueError("snapshot file entry must be an object")
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
                raise ValueError("snapshot contains unsafe file path")
            try:
                payload = base64.b64decode(encoded, validate=True)
            except ValueError as exc:
                raise ValueError("snapshot file is not valid base64") from exc
            if item.get("size") != len(payload):
                raise ValueError("snapshot file size does not match payload")
            total += len(payload)
            if total > MAX_RESTORE_BYTES:
                raise ValueError("snapshot files exceed restore size limit")
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
        staged.mkdir(mode=0o700)
        try:
            for relative, payload in files:
                destination = staged / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                descriptor = os.open(
                    destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                )
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
            with self._connect() as source, sqlite3.connect(database_backup) as target:
                source.backup(target)
            files_backup.mkdir(mode=0o700)
            self._backup_private_files(files_backup)
            with self._connect() as connection:
                with connection:
                    for table in reversed(_TABLES):
                        connection.execute(f"DELETE FROM {table}")
                    for table in _TABLES:
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
            self._remove_private_files()
            for path in staged.rglob("*"):
                if not path.is_file():
                    continue
                destination = self.data_dir / path.relative_to(staged)
                destination.parent.mkdir(parents=True, exist_ok=True)
                path.replace(destination)
            self._refresh_keys()
            return {
                "rows_restored": sum(len(rows) for rows in tables.values()),
                "files_restored": len(files),
            }
        except Exception:
            if database_backup.exists():
                with (
                    sqlite3.connect(database_backup) as source,
                    self._connect() as target,
                ):
                    source.backup(target)
            if files_backup.exists():
                self._remove_private_files()
                self._restore_private_files(files_backup)
            self._refresh_keys()
            raise
        finally:
            database_backup.unlink(missing_ok=True)
            shutil.rmtree(files_backup, ignore_errors=True)
            shutil.rmtree(staged, ignore_errors=True)

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

    def delete_all(self) -> None:
        """Erase sessions/audio, graph/provenance, outreach, corpus, keys, and backups."""
        self.proactivity.delete_all()
        self.research.delete_all()
        self.model.delete_all()
        self.store.delete_all()
        self._remove_private_files()
        for backup in self.data_dir.glob("therapy.db*.bak"):
            backup.unlink(missing_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            connection.execute("VACUUM")


__all__ = ["DataSovereignty", "EXPORT_FORMAT", "EXPORT_VERSION", "MAX_RESTORE_BYTES"]
