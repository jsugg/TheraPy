"""Dedicated SQLite interaction journal (plan §5.3).

Source of truth for every LLM attempt. Design points:

- Separate owner-only database; never the product DB or a backend DB.
- `journal_mode=WAL`, `synchronous=FULL`, foreign keys, busy timeout,
  short transactions, periodic passive checkpoints with bounded manual
  escalation, `integrity_check` off the hot path.
- One bounded async writer serializes writes. Pre-dispatch and terminal
  writes commit immediately and durably; stream deltas use bounded group
  commit (`THERAPY_INTERACTION_GROUP_COMMIT_MS`).
- State transitions are monotonic and idempotent: duplicate identical
  writes succeed silently; conflicting duplicates raise `JournalConflict`.
- Recovery marks stale nonterminal rows `incomplete` only after checking
  persisted events; it never invents success.
- SHA-256 checksums over canonical UTF-8 JSON detect corruption/dedup and
  never substitute for content.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import stat
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType
from typing import TypedDict

from therapy.observability.interactions import (
    InteractionRecord,
    JsonValue,
    canonical_json,
    checksum,
    require_json_object,
)
from therapy.observability.model import (
    INTERACTION_STATUS_TRANSITIONS,
    PAYLOAD_ENCODING,
    TERMINAL_INTERACTION_STATUSES,
    InteractionEventKind,
    InteractionStatus,
)

JOURNAL_SCHEMA_VERSION = 2


class InteractionJournalRow(TypedDict):
    """Typed persisted interaction row returned by `JournalStore.load`."""

    interaction_id: str
    schema_version: int
    payload_encoding: str
    key_version: str | None
    nonce: bytes | None
    trace_id: str
    span_id: str
    operation: str
    provider: str
    status: str
    started_at: str
    completed_at: str | None
    canonical_request_json: str
    canonical_record_json: str
    provider_request_json: str
    terminal_json: str | None
    checksum: str
    next_sequence: int
    created_at: str
    updated_at: str


class InteractionEventRow(TypedDict):
    """Typed persisted interaction event returned by `JournalStore.load`."""

    sequence: int
    kind: str
    observed_at: str
    payload_json: str
    checksum: str


class LoadedInteraction(TypedDict):
    """One interaction row and its ordered event rows."""

    interaction: InteractionJournalRow
    events: list[InteractionEventRow]


def _row_str(row: sqlite3.Row, key: str) -> str:
    value = row[key]
    if not isinstance(value, str):
        raise JournalError(f"journal column {key!r} is not text")
    return value


def _row_int(row: sqlite3.Row, key: str) -> int:
    value = row[key]
    if not isinstance(value, int):
        raise JournalError(f"journal column {key!r} is not an integer")
    return value


def _row_optional_str(row: sqlite3.Row, key: str) -> str | None:
    value = row[key]
    if value is not None and not isinstance(value, str):
        raise JournalError(f"journal column {key!r} is not nullable text")
    return value


def _row_optional_bytes(row: sqlite3.Row, key: str) -> bytes | None:
    value = row[key]
    if value is None:
        return None
    if not isinstance(value, bytes):
        raise JournalError(f"journal column {key!r} is not nullable bytes")
    return value

#: v2 (2026-07-16): `canonical_record_json` persists the COMPLETE §5.2
#: pre-dispatch envelope so exact reconstruction never depends on scattered
#: columns; the row checksum covers it plus the terminal.

_DDL = """
CREATE TABLE IF NOT EXISTS interactions(
  interaction_id TEXT PRIMARY KEY,
  schema_version INTEGER NOT NULL,
  payload_encoding TEXT NOT NULL CHECK(payload_encoding = 'json-v1'),
  key_version TEXT NULL,
  nonce BLOB NULL,
  trace_id TEXT NOT NULL,
  span_id TEXT NOT NULL,
  operation TEXT NOT NULL,
  provider TEXT NOT NULL,
  status TEXT NOT NULL,
  started_at TEXT NOT NULL,
  completed_at TEXT NULL,
  canonical_request_json TEXT NOT NULL,
  canonical_record_json TEXT NOT NULL DEFAULT '',
  provider_request_json TEXT NOT NULL,
  terminal_json TEXT NULL,
  checksum TEXT NOT NULL,
  next_sequence INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS interaction_events(
  interaction_id TEXT NOT NULL REFERENCES interactions ON DELETE CASCADE,
  sequence INTEGER NOT NULL,
  kind TEXT NOT NULL,
  observed_at TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  checksum TEXT NOT NULL,
  PRIMARY KEY(interaction_id, sequence)
);

CREATE TABLE IF NOT EXISTS interaction_exports(
  interaction_id TEXT NOT NULL REFERENCES interactions ON DELETE CASCADE,
  backend TEXT NOT NULL,
  state TEXT NOT NULL,
  attempts INTEGER NOT NULL,
  next_attempt_at TEXT NULL,
  last_error_type TEXT NULL,
  acknowledged_at TEXT NULL,
  backend_record_id TEXT NULL,
  PRIMARY KEY(interaction_id, backend)
);

CREATE TABLE IF NOT EXISTS journal_metadata(
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_interactions_status
  ON interactions(status, updated_at);
"""


class JournalError(RuntimeError):
    """Journal unavailable or corrupt; capture policy decides what happens."""


class JournalConflict(JournalError):
    """A conflicting duplicate write — visible by design (§5.3)."""


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class JournalHealth:
    """Content-free capture-health snapshot (plan O1.2 item 3)."""

    total_records: int
    nonterminal_records: int
    incomplete_records: int
    unexported_records: int
    oldest_unexported_at: str | None
    wal_bytes: int
    last_checkpoint_at: str | None
    db_bytes: int


class JournalStore:
    """Synchronous core; always driven by the single async writer or tests."""

    def __init__(self, path: Path, *, busy_timeout_ms: int = 5000) -> None:
        self.path = path
        self._busy_timeout_ms = busy_timeout_ms
        self._last_checkpoint_at: str | None = None
        path.parent.mkdir(parents=True, exist_ok=True)
        self._restrict_permissions(path.parent, directory=True)
        # check_same_thread=False: one async writer serializes thread-pool access.
        self._conn = sqlite3.connect(
            path,
            isolation_level=None,
            timeout=busy_timeout_ms / 1000,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._configure()
        self._migrate()
        self._restrict_permissions(path, directory=False)

    @staticmethod
    def _restrict_permissions(path: Path, *, directory: bool) -> None:
        try:
            mode = stat.S_IRWXU if directory else (stat.S_IRUSR | stat.S_IWUSR)
            os.chmod(path, mode)
        except OSError:
            pass  # non-POSIX targets; documented as best-effort

    def _configure(self) -> None:
        cur = self._conn
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=FULL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")

    def _migrate(self) -> None:
        # executescript() issues an implicit COMMIT, so DDL runs outside the
        # short-transaction helper (the connection is in autocommit mode).
        self._conn.executescript(_DDL)
        with self._tx():
            row = self._conn.execute(
                "SELECT value FROM journal_metadata WHERE key='schema_version'"
            ).fetchone()
            if row is None:
                self._conn.execute(
                    "INSERT INTO journal_metadata(key, value) VALUES(?, ?)",
                    ("schema_version", str(JOURNAL_SCHEMA_VERSION)),
                )
                return
            stored = int(row["value"])
            if stored > JOURNAL_SCHEMA_VERSION:
                raise JournalError("journal schema is newer than this build")
            if stored < 2:
                # v1 -> v2: add the complete-envelope column. v1 rows keep
                # an empty envelope and are reported as legacy by
                # `reconstruct()`; they were all synthetic dev records.
                columns = {
                    r[1]
                    for r in self._conn.execute(
                        "PRAGMA table_info(interactions)"
                    ).fetchall()
                }
                if "canonical_record_json" not in columns:
                    self._conn.execute(
                        "ALTER TABLE interactions ADD COLUMN "
                        "canonical_record_json TEXT NOT NULL DEFAULT ''"
                    )
            self._conn.execute(
                "UPDATE journal_metadata SET value=? WHERE key='schema_version'",
                (str(JOURNAL_SCHEMA_VERSION),),
            )

    def _tx(self):
        return _Transaction(self._conn)

    # -- lifecycle ---------------------------------------------------------

    def start_attempt(self, record: InteractionRecord) -> None:
        """Immediate durable pre-dispatch commit (§5.3)."""
        if record.status is not InteractionStatus.STARTED:
            raise JournalConflict("start_attempt requires status=started")
        payload = record.to_json_dict()
        request = require_json_object(payload.get("request"), "interaction.request")
        provider_native = require_json_object(
            payload.get("provider_native"), "interaction.provider_native"
        )
        provider_request = require_json_object(
            provider_native.get("request"), "interaction.provider_native.request"
        )
        canonical_request = canonical_json(request)
        canonical_record = canonical_json(payload)
        provider_request_json = canonical_json(provider_request)
        digest = checksum(payload)
        now = _utcnow()
        try:
            with self._tx():
                self._conn.execute(
                    """
                    INSERT INTO interactions(
                      interaction_id, schema_version, payload_encoding,
                      key_version, nonce, trace_id, span_id, operation,
                      provider, status, started_at, completed_at,
                      canonical_request_json, canonical_record_json,
                      provider_request_json,
                      terminal_json, checksum, next_sequence,
                      created_at, updated_at
                    ) VALUES (?,?,?,NULL,NULL,?,?,?,?,?,?,NULL,?,?,?,NULL,?,0,?,?)
                    """,
                    (
                        record.interaction_id,
                        record.schema_version,
                        PAYLOAD_ENCODING,
                        record.trace_id,
                        record.span_id,
                        record.operation.value,
                        record.provider.value,
                        record.status.value,
                        record.started_at,
                        canonical_request,
                        canonical_record,
                        provider_request_json,
                        digest,
                        now,
                        now,
                    ),
                )
        except sqlite3.IntegrityError:
            existing = self._conn.execute(
                "SELECT checksum FROM interactions WHERE interaction_id=?",
                (record.interaction_id,),
            ).fetchone()
            if existing is not None and existing["checksum"] == digest:
                return  # identical duplicate: idempotent success
            raise JournalConflict(
                f"conflicting duplicate start for {record.interaction_id}"
            ) from None

    def append_stream_event(
        self,
        interaction_id: str,
        sequence: int,
        kind: InteractionEventKind,
        observed_at: str,
        payload: dict[str, JsonValue],
    ) -> None:
        with self._tx():
            self._append_one(interaction_id, sequence, kind, observed_at, payload)

    def append_stream_events(
        self,
        items: list[tuple[str, int, InteractionEventKind, str, dict[str, JsonValue]]],
    ) -> None:
        """True group commit (§5.3): one transaction for the whole batch.

        A conflict anywhere rolls the batch back and surfaces to every
        member — a partially persisted group is never reported as success.
        """
        with self._tx():
            for interaction_id, sequence, kind, observed_at, payload in items:
                self._append_one(interaction_id, sequence, kind, observed_at, payload)

    def _append_one(
        self,
        interaction_id: str,
        sequence: int,
        kind: InteractionEventKind,
        observed_at: str,
        payload: dict[str, JsonValue],
    ) -> None:
        payload_text = canonical_json(payload)
        digest = checksum(payload)
        row = self._conn.execute(
            "SELECT status, next_sequence FROM interactions WHERE interaction_id=?",
            (interaction_id,),
        ).fetchone()
        if row is None:
            raise JournalConflict(f"unknown interaction {interaction_id}")
        status = InteractionStatus(row["status"])
        if status in TERMINAL_INTERACTION_STATUSES:
            raise JournalConflict(
                f"stream event after terminal state for {interaction_id}"
            )
        try:
            self._conn.execute(
                """
                INSERT INTO interaction_events(
                  interaction_id, sequence, kind, observed_at,
                  payload_json, checksum
                ) VALUES (?,?,?,?,?,?)
                """,
                (interaction_id, sequence, kind.value, observed_at, payload_text, digest),
            )
        except sqlite3.IntegrityError:
            existing = self._conn.execute(
                "SELECT checksum FROM interaction_events "
                "WHERE interaction_id=? AND sequence=?",
                (interaction_id, sequence),
            ).fetchone()
            if existing is not None and existing["checksum"] == digest:
                return  # idempotent duplicate
            raise JournalConflict(
                f"conflicting duplicate sequence {sequence} for {interaction_id}"
            ) from None
        new_status = (
            InteractionStatus.STREAMING
            if status is InteractionStatus.STARTED
            else status
        )
        self._conn.execute(
            "UPDATE interactions SET next_sequence=MAX(next_sequence, ?), "
            "status=?, updated_at=? WHERE interaction_id=?",
            (sequence + 1, new_status.value, _utcnow(), interaction_id),
        )

    def _finish(
        self,
        interaction_id: str,
        new_status: InteractionStatus,
        terminal: dict[str, JsonValue],
        completed_at: str,
    ) -> None:
        terminal_text = canonical_json(terminal)
        with self._tx():
            row = self._conn.execute(
                "SELECT status, terminal_json, checksum FROM interactions "
                "WHERE interaction_id=?",
                (interaction_id,),
            ).fetchone()
            if row is None:
                raise JournalConflict(f"unknown interaction {interaction_id}")
            status = InteractionStatus(row["status"])
            if status in TERMINAL_INTERACTION_STATUSES:
                if status is new_status and row["terminal_json"] == terminal_text:
                    return  # idempotent duplicate terminal
                raise JournalConflict(
                    f"conflicting terminal write for {interaction_id}: "
                    f"{status.value} -> {new_status.value}"
                )
            if new_status not in INTERACTION_STATUS_TRANSITIONS[status]:
                raise JournalConflict(
                    f"illegal transition {status.value} -> {new_status.value}"
                )
            digest = checksum({"base": row["checksum"], "terminal": terminal})
            self._conn.execute(
                "UPDATE interactions SET status=?, terminal_json=?, "
                "completed_at=?, checksum=?, updated_at=? WHERE interaction_id=?",
                (
                    new_status.value,
                    terminal_text,
                    completed_at,
                    digest,
                    _utcnow(),
                    interaction_id,
                ),
            )

    def finish_success(
        self,
        interaction_id: str,
        terminal: dict[str, JsonValue],
        completed_at: str | None = None,
    ) -> None:
        self._finish(
            interaction_id,
            InteractionStatus.SUCCEEDED,
            {"kind": "success", **terminal},
            completed_at or _utcnow(),
        )

    def finish_error(
        self,
        interaction_id: str,
        error: dict[str, JsonValue],
        completed_at: str | None = None,
    ) -> None:
        """Immediate exact provider-error commit."""
        self._finish(
            interaction_id,
            InteractionStatus.FAILED,
            {"kind": "error", **error},
            completed_at or _utcnow(),
        )

    def mark_incomplete(
        self,
        interaction_id: str,
        reason: str,
        completed_at: str | None = None,
    ) -> None:
        """Explicit terminal gap — never silent (§5.3)."""
        self._finish(
            interaction_id,
            InteractionStatus.INCOMPLETE,
            {"kind": "incomplete", "reason": reason},
            completed_at or _utcnow(),
        )

    # -- recovery / maintenance --------------------------------------------

    def recover(self) -> int:
        """Mark nonterminal rows from a previous process `incomplete`.

        Checks persisted events first so the terminal record reflects how
        far the stream truly got; never invents success.
        """
        recovered = 0
        rows = self._conn.execute(
            "SELECT interaction_id FROM interactions WHERE status IN (?, ?)",
            (InteractionStatus.STARTED.value, InteractionStatus.STREAMING.value),
        ).fetchall()
        for row in rows:
            interaction_id = row["interaction_id"]
            last = self._conn.execute(
                "SELECT MAX(sequence) AS last_seq, COUNT(*) AS events "
                "FROM interaction_events WHERE interaction_id=?",
                (interaction_id,),
            ).fetchone()
            self.mark_incomplete(
                interaction_id,
                reason=(
                    f"recovered_after_restart:persisted_events={last['events']}"
                    f":last_sequence={last['last_seq'] if last['last_seq'] is not None else 'none'}"
                ),
            )
            recovered += 1
        return recovered

    def checkpoint(self, *, wal_escalation_bytes: int = 16 * 1024 * 1024) -> None:
        """Passive checkpoint; one bounded TRUNCATE escalation when the WAL
        has grown past the threshold."""
        self._conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
        if self._wal_bytes() > wal_escalation_bytes:
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self._last_checkpoint_at = _utcnow()

    def integrity_check(self) -> bool:
        row = self._conn.execute("PRAGMA integrity_check").fetchone()
        return bool(row) and row[0] == "ok"

    def apply_retention(
        self, retention_days: int, *, require_ack_backend: str | None
    ) -> int:
        """Delete expired records; unacknowledged rows ignore expiry (§4)."""
        cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()
        with self._tx():
            if require_ack_backend is None:
                cursor = self._conn.execute(
                    "DELETE FROM interactions WHERE started_at < ? "
                    "AND status IN (?, ?, ?)",
                    (cutoff, *(s.value for s in TERMINAL_INTERACTION_STATUSES)),
                )
            else:
                cursor = self._conn.execute(
                    """
                    DELETE FROM interactions WHERE started_at < ?
                    AND status IN (?, ?, ?)
                    AND interaction_id IN (
                      SELECT interaction_id FROM interaction_exports
                      WHERE backend = ? AND acknowledged_at IS NOT NULL
                    )
                    """,
                    (
                        cutoff,
                        *(s.value for s in TERMINAL_INTERACTION_STATUSES),
                        require_ack_backend,
                    ),
                )
            return cursor.rowcount

    # -- export/replay ------------------------------------------------------

    def pending_exports(self, backend: str, limit: int = 32) -> list[str]:
        """Terminal interactions not yet acknowledged by `backend`."""
        now = _utcnow()
        rows = self._conn.execute(
            """
            SELECT i.interaction_id FROM interactions i
            LEFT JOIN interaction_exports e
              ON e.interaction_id = i.interaction_id AND e.backend = ?
            WHERE i.status IN (?, ?, ?)
              AND (e.acknowledged_at IS NULL)
              AND (e.next_attempt_at IS NULL OR e.next_attempt_at <= ?)
            ORDER BY i.started_at
            LIMIT ?
            """,
            (
                backend,
                *(s.value for s in TERMINAL_INTERACTION_STATUSES),
                now,
                limit,
            ),
        ).fetchall()
        return [row["interaction_id"] for row in rows]

    def record_export_attempt(
        self,
        interaction_id: str,
        backend: str,
        *,
        acknowledged: bool,
        backend_record_id: str | None = None,
        error_type: str | None = None,
        next_attempt_at: str | None = None,
    ) -> None:
        """Idempotent ACK recording; an existing ACK never regresses."""
        with self._tx():
            row = self._conn.execute(
                "SELECT acknowledged_at, attempts FROM interaction_exports "
                "WHERE interaction_id=? AND backend=?",
                (interaction_id, backend),
            ).fetchone()
            if row is not None and row["acknowledged_at"] is not None:
                return
            state = "acknowledged" if acknowledged else "pending"
            ack_at = _utcnow() if acknowledged else None
            if row is None:
                self._conn.execute(
                    """
                    INSERT INTO interaction_exports(
                      interaction_id, backend, state, attempts,
                      next_attempt_at, last_error_type, acknowledged_at,
                      backend_record_id
                    ) VALUES (?,?,?,?,?,?,?,?)
                    """,
                    (
                        interaction_id,
                        backend,
                        state,
                        1,
                        next_attempt_at,
                        error_type,
                        ack_at,
                        backend_record_id,
                    ),
                )

            else:
                self._conn.execute(
                    """
                    UPDATE interaction_exports SET state=?, attempts=attempts+1,
                      next_attempt_at=?, last_error_type=?, acknowledged_at=?,
                      backend_record_id=?
                    WHERE interaction_id=? AND backend=?
                    """,
                    (
                        state,
                        next_attempt_at,
                        error_type,
                        ack_at,
                        backend_record_id,
                        interaction_id,
                        backend,
                    ),
                )

    def export_attempts(self, interaction_id: str, backend: str) -> int:
        """Return the durable attempt count for one backend export."""
        row = self._conn.execute(
            "SELECT attempts FROM interaction_exports "
            "WHERE interaction_id=? AND backend=?",
            (interaction_id, backend),
        ).fetchone()
        if row is None:
            return 0
        attempts: object = row["attempts"]
        if not isinstance(attempts, int):
            raise JournalError("export attempt count is not an integer")
        return attempts

    # -- read/replay ----------------------------------------------------------

    def load(self, interaction_id: str) -> LoadedInteraction | None:
        """Load and validate one interaction plus its ordered events."""
        row = self._conn.execute(
            "SELECT * FROM interactions WHERE interaction_id=?", (interaction_id,)
        ).fetchone()
        if row is None:
            return None
        events = self._conn.execute(
            "SELECT sequence, kind, observed_at, payload_json, checksum "
            "FROM interaction_events WHERE interaction_id=? ORDER BY sequence",
            (interaction_id,),
        ).fetchall()
        interaction = InteractionJournalRow(
            interaction_id=_row_str(row, "interaction_id"),
            schema_version=_row_int(row, "schema_version"),
            payload_encoding=_row_str(row, "payload_encoding"),
            key_version=_row_optional_str(row, "key_version"),
            nonce=_row_optional_bytes(row, "nonce"),
            trace_id=_row_str(row, "trace_id"),
            span_id=_row_str(row, "span_id"),
            operation=_row_str(row, "operation"),
            provider=_row_str(row, "provider"),
            status=_row_str(row, "status"),
            started_at=_row_str(row, "started_at"),
            completed_at=_row_optional_str(row, "completed_at"),
            canonical_request_json=_row_str(row, "canonical_request_json"),
            canonical_record_json=_row_str(row, "canonical_record_json"),
            provider_request_json=_row_str(row, "provider_request_json"),
            terminal_json=_row_optional_str(row, "terminal_json"),
            checksum=_row_str(row, "checksum"),
            next_sequence=_row_int(row, "next_sequence"),
            created_at=_row_str(row, "created_at"),
            updated_at=_row_str(row, "updated_at"),
        )
        loaded_events = [
            InteractionEventRow(
                sequence=_row_int(event, "sequence"),
                kind=_row_str(event, "kind"),
                observed_at=_row_str(event, "observed_at"),
                payload_json=_row_str(event, "payload_json"),
                checksum=_row_str(event, "checksum"),
            )
            for event in events
        ]
        return LoadedInteraction(interaction=interaction, events=loaded_events)

    def verify_checksums(self, interaction_id: str) -> bool:
        """Corruption check over the WHOLE row: the pre-dispatch canonical
        envelope, the chained terminal digest, and every event payload."""
        import json as _json

        data = self.load(interaction_id)
        if data is None:
            return False
        row = data["interaction"]
        envelope_text = row.get("canonical_record_json") or ""
        if envelope_text:
            try:
                start_digest = checksum(_json.loads(envelope_text))
            except ValueError:
                return False
            if row["terminal_json"] is None:
                if row["checksum"] != start_digest:
                    return False
            else:
                try:
                    terminal = _json.loads(row["terminal_json"])
                except ValueError:
                    return False
                chained = checksum({"base": start_digest, "terminal": terminal})
                if row["checksum"] != chained:
                    return False
        for event in data["events"]:
            payload = _json.loads(event["payload_json"])
            if checksum(payload) != event["checksum"]:
                return False
        return True

    def reconstruct(self, interaction_id: str) -> dict[str, JsonValue] | None:
        """Exact §5.2 reconstruction: the pre-dispatch canonical envelope
        with the persisted stream and terminal folded back in.

        Returns None for unknown IDs; raises `JournalError` for legacy v1
        rows without a stored envelope (visible, never invented)."""
        import json as _json

        data = self.load(interaction_id)
        if data is None:
            return None
        row = data["interaction"]
        envelope_text = row.get("canonical_record_json") or ""
        if not envelope_text:
            raise JournalError(
                f"{interaction_id} is a legacy v1 row without a stored envelope"
            )
        envelope: dict[str, JsonValue] = _json.loads(envelope_text)
        envelope["status"] = row["status"]
        envelope["completed_at"] = row["completed_at"]
        stream: list[JsonValue] = []
        native_events: list[JsonValue] = []
        for event in data["events"]:
            payload = _json.loads(event["payload_json"])
            entry = {
                "sequence": event["sequence"],
                "observed_at": event["observed_at"],
                "kind": event["kind"],
                **(payload if isinstance(payload, dict) else {"payload": payload}),
            }
            if event["kind"] in ("stream_delta", "tool_delta"):
                stream.append(entry)
            else:
                native_events.append(entry)
        envelope["stream"] = stream
        native = envelope.get("provider_native")
        if isinstance(native, dict):
            ordered_events = native.get("ordered_events")
            if ordered_events is None:
                ordered_events = []
            if not isinstance(ordered_events, list):
                raise JournalError("provider ordered_events is not an array")
            native["ordered_events"] = ordered_events + native_events
        if row["terminal_json"] is not None:
            terminal = require_json_object(
                _json.loads(row["terminal_json"]), "interaction terminal"
            )
            envelope["terminal"] = terminal
            response = terminal.get("response")
            error = terminal.get("error")
            if isinstance(response, dict):
                envelope["response"] = response
            if isinstance(error, dict):
                envelope["error"] = error
        return envelope

    def iter_interaction_ids(self) -> Iterator[str]:
        for row in self._conn.execute(
            "SELECT interaction_id FROM interactions ORDER BY started_at"
        ):
            yield row["interaction_id"]

    def health(self) -> JournalHealth:
        total = self._conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
        nonterminal = self._conn.execute(
            "SELECT COUNT(*) FROM interactions WHERE status IN (?, ?)",
            (InteractionStatus.STARTED.value, InteractionStatus.STREAMING.value),
        ).fetchone()[0]
        incomplete = self._conn.execute(
            "SELECT COUNT(*) FROM interactions WHERE status=?",
            (InteractionStatus.INCOMPLETE.value,),
        ).fetchone()[0]
        unexported_row = self._conn.execute(
            """
            SELECT COUNT(*) AS n, MIN(i.started_at) AS oldest
            FROM interactions i
            LEFT JOIN interaction_exports e
              ON e.interaction_id = i.interaction_id
              AND e.acknowledged_at IS NOT NULL
            WHERE e.interaction_id IS NULL
            """
        ).fetchone()
        return JournalHealth(
            total_records=total,
            nonterminal_records=nonterminal,
            incomplete_records=incomplete,
            unexported_records=unexported_row["n"],
            oldest_unexported_at=unexported_row["oldest"],
            wal_bytes=self._wal_bytes(),
            last_checkpoint_at=self._last_checkpoint_at,
            db_bytes=self.path.stat().st_size if self.path.exists() else 0,
        )

    def _wal_bytes(self) -> int:
        wal = self.path.with_name(self.path.name + "-wal")
        try:
            return wal.stat().st_size
        except OSError:
            return 0

    def close(self) -> None:
        self._conn.close()


class _Transaction:
    """Short IMMEDIATE transaction; keeps writer lock windows small."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def __enter__(self) -> sqlite3.Connection:
        self._conn.execute("BEGIN IMMEDIATE")
        return self._conn

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self._conn.execute("COMMIT")
        elif self._conn.in_transaction:
            # a failed write (e.g. disk full) may already have auto-rolled
            # back; never mask the original error with a rollback error
            try:
                self._conn.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass


@dataclass(frozen=True, slots=True)
class _WriteOp:
    method: str
    args: tuple[object, ...]
    kwargs: dict[str, object]
    future: asyncio.Future[None]
    #: durable ops complete their future only after commit; grouped ops
    #: (stream deltas) may resolve on enqueue-flush.
    durable: bool
    stream_item: tuple[
        str, int, InteractionEventKind, str, dict[str, JsonValue]
    ] | None = None


class AsyncJournalWriter:
    """The single bounded writer task (§5.3).

    `start_attempt`, `finish_success`, `finish_error`, and `mark_incomplete`
    await the durable commit. `append_stream_event` participates in bounded
    group commit: it awaits queue admission (backpressure), and the writer
    flushes batches every `group_commit_ms` or when the batch fills.
    """

    def __init__(
        self,
        store: JournalStore,
        *,
        queue_size: int = 256,
        group_commit_ms: int = 50,
    ) -> None:
        self._store = store
        self._queue: asyncio.Queue[_WriteOp | None] = asyncio.Queue(maxsize=queue_size)
        self._group_commit_s = group_commit_ms / 1000
        self._task: asyncio.Task[None] | None = None
        self._closed = False

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="interaction-journal")

    async def _run(self) -> None:
        while True:
            op = await self._queue.get()
            if op is None:
                return
            batch = [op]
            if not op.durable:
                # gather more grouped ops for one commit window
                deadline = time.monotonic() + self._group_commit_s
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or len(batch) >= 64:
                        break
                    try:
                        extra = await asyncio.wait_for(self._queue.get(), remaining)
                    except TimeoutError:
                        break
                    if extra is None:
                        await self._apply(batch)
                        return
                    batch.append(extra)
                    if extra.durable:
                        break  # durable op closes the window immediately
            await self._apply(batch)

    async def _apply(self, batch: list[_WriteOp]) -> None:
        from therapy.observability.model import WorkloadClass
        from therapy.observability.telemetry import run_in_thread

        batch_started = time.monotonic()
        batch_failed = False
        batch_bytes = sum(self._operation_bytes(op) for op in batch)
        index = 0
        while index < len(batch):
            op = batch[index]
            # consecutive stream appends commit as ONE transaction (§5.3)
            if op.method == "append_stream_event":
                run = [op]
                while (
                    index + len(run) < len(batch)
                    and batch[index + len(run)].method == "append_stream_event"
                ):
                    run.append(batch[index + len(run)])
                items = [
                    member.stream_item
                    for member in run
                    if member.stream_item is not None
                ]
                if len(items) != len(run):
                    raise JournalError("stream append operation lacks typed payload")
                started = time.monotonic()
                try:
                    await run_in_thread(
                        WorkloadClass.BACKGROUND,
                        self._store.append_stream_events,
                        items,
                    )
                    self._observe_append(
                        (time.monotonic() - started) * 1000, "success"
                    )
                except Exception as exc:
                    batch_failed = True
                    self._observe_append(
                        (time.monotonic() - started) * 1000, "error"
                    )
                    for member in run:
                        if not member.future.done():
                            member.future.set_exception(exc)
                else:
                    for member in run:
                        if not member.future.done():
                            member.future.set_result(None)
                index += len(run)
                continue
            started = time.monotonic()
            try:
                await run_in_thread(
                    WorkloadClass.BACKGROUND,
                    getattr(self._store, op.method),
                    *op.args,
                    **op.kwargs,
                )
                self._observe_append((time.monotonic() - started) * 1000, "success")
            except Exception as exc:  # propagate to the caller, keep writing
                batch_failed = True
                self._observe_append((time.monotonic() - started) * 1000, "error")
                if not op.future.done():
                    op.future.set_exception(exc)
            else:
                if not op.future.done():
                    op.future.set_result(None)
            index += 1
        from therapy.observability.telemetry import record_metric

        outcome = "error" if batch_failed else "success"
        record_metric(
            "therapy_llm_capture_group_commits_total", 1, {"outcome": outcome}
        )
        record_metric(
            "therapy_llm_capture_group_commit_seconds",
            time.monotonic() - batch_started,
            {"outcome": outcome},
        )
        record_metric("therapy_llm_capture_group_commit_records", len(batch))
        record_metric("therapy_llm_capture_append_bytes", batch_bytes)
        record_metric("therapy_llm_capture_queue_depth", self._queue.qsize())

    @staticmethod
    def _operation_bytes(op: _WriteOp) -> int:
        """Return the local canonical payload size without exporting its content."""
        if op.method == "start_attempt" and op.args:
            record = op.args[0]
            if isinstance(record, InteractionRecord):
                return len(record.canonical().encode("utf-8"))
        if op.method == "append_stream_event" and len(op.args) >= 5:
            return len(
                json.dumps(
                    op.args[4], ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            )
        if op.method in {
            "finish_success",
            "finish_error",
            "mark_incomplete",
        } and len(op.args) >= 2:
            return len(
                json.dumps(
                    op.args[1], ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            )
        return 0

    @staticmethod
    def _observe_append(duration_ms: float, outcome: str) -> None:
        from therapy.observability.telemetry import record_metric

        record_metric(
            "therapy_llm_capture_append_seconds",
            duration_ms / 1000,
            {"outcome": outcome},
        )

    async def _submit(
        self,
        method: str,
        *args: object,
        durable: bool,
        stream_item: tuple[
            str, int, InteractionEventKind, str, dict[str, JsonValue]
        ]
        | None = None,
        **kwargs: object,
    ) -> None:
        if self._closed:
            raise JournalError("journal writer is closed")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        await self._queue.put(
            _WriteOp(method, args, kwargs, future, durable, stream_item)
        )
        from therapy.observability.telemetry import record_metric

        record_metric("therapy_llm_capture_queue_depth", self._queue.qsize())
        await future

    async def start_attempt(self, record: InteractionRecord) -> None:
        await self._submit("start_attempt", record, durable=True)

    async def append_stream_event(
        self,
        interaction_id: str,
        sequence: int,
        kind: InteractionEventKind,
        observed_at: str,
        payload: dict[str, JsonValue],
    ) -> None:
        await self._submit(
            "append_stream_event",
            interaction_id,
            sequence,
            kind,
            observed_at,
            payload,
            durable=False,
            stream_item=(interaction_id, sequence, kind, observed_at, payload),
        )

    async def finish_success(
        self, interaction_id: str, terminal: dict[str, JsonValue]
    ) -> None:
        await self._submit("finish_success", interaction_id, terminal, durable=True)

    async def finish_error(
        self, interaction_id: str, error: dict[str, JsonValue]
    ) -> None:
        await self._submit("finish_error", interaction_id, error, durable=True)

    async def mark_incomplete(self, interaction_id: str, reason: str) -> None:
        await self._submit("mark_incomplete", interaction_id, reason, durable=True)

    async def flush(self) -> None:
        await self._submit("checkpoint", durable=True)

    async def close(self, timeout: float = 5.0) -> None:
        """Bounded shutdown; product shutdown never waits indefinitely."""
        if self._closed:
            return
        self._closed = True
        if self._task is not None:
            try:
                self._queue.put_nowait(None)
            except asyncio.QueueFull:
                self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout)
            except (TimeoutError, asyncio.CancelledError):
                self._task.cancel()
        try:
            from therapy.observability.model import WorkloadClass
            from therapy.observability.telemetry import run_in_thread

            await asyncio.wait_for(
                run_in_thread(WorkloadClass.MAINTENANCE, self._store.close), timeout
            )
        except TimeoutError:
            pass  # the connection dies with the process; data is committed
