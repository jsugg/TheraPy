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
import os
import sqlite3
import stat
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from therapy.observability.interactions import (
    InteractionRecord,
    JsonValue,
    canonical_json,
    checksum,
)
from therapy.observability.model import (
    INTERACTION_STATUS_TRANSITIONS,
    PAYLOAD_ENCODING,
    TERMINAL_INTERACTION_STATUSES,
    InteractionEventKind,
    InteractionStatus,
)

JOURNAL_SCHEMA_VERSION = 1

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
        # check_same_thread=False: the single async writer serializes all
        # access but hops threads via asyncio.to_thread.
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
            elif int(row["value"]) > JOURNAL_SCHEMA_VERSION:
                raise JournalError("journal schema is newer than this build")
            # future migrations branch on the stored version here

    def _tx(self):
        return _Transaction(self._conn)

    # -- lifecycle ---------------------------------------------------------

    def start_attempt(self, record: InteractionRecord) -> None:
        """Immediate durable pre-dispatch commit (§5.3)."""
        if record.status is not InteractionStatus.STARTED:
            raise JournalConflict("start_attempt requires status=started")
        payload = record.to_json_dict()
        canonical_request = canonical_json(payload["request"])  # type: ignore[arg-type]
        provider_request = canonical_json(payload["provider_native"]["request"])  # type: ignore[index]
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
                      canonical_request_json, provider_request_json,
                      terminal_json, checksum, next_sequence,
                      created_at, updated_at
                    ) VALUES (?,?,?,NULL,NULL,?,?,?,?,?,?,NULL,?,?,NULL,?,0,?,?)
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
                        provider_request,
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
        payload_text = canonical_json(payload)
        digest = checksum(payload)
        with self._tx():
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

    # -- read/replay ----------------------------------------------------------

    def load(self, interaction_id: str) -> dict[str, JsonValue] | None:
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
        return {
            "interaction": dict(row),
            "events": [dict(event) for event in events],
        }

    def verify_checksums(self, interaction_id: str) -> bool:
        import json as _json

        data = self.load(interaction_id)
        if data is None:
            return False
        for event in data["events"]:
            payload = _json.loads(event["payload_json"])
            if checksum(payload) != event["checksum"]:
                return False
        return True

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

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self._conn.execute("COMMIT")
        else:
            self._conn.execute("ROLLBACK")


@dataclass(frozen=True, slots=True)
class _WriteOp:
    method: str
    args: tuple
    kwargs: dict
    future: asyncio.Future
    #: durable ops complete their future only after commit; grouped ops
    #: (stream deltas) may resolve on enqueue-flush.
    durable: bool


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
        for op in batch:
            try:
                await asyncio.to_thread(
                    getattr(self._store, op.method), *op.args, **op.kwargs
                )
            except Exception as exc:  # propagate to the caller, keep writing
                if not op.future.done():
                    op.future.set_exception(exc)
            else:
                if not op.future.done():
                    op.future.set_result(None)

    async def _submit(self, method: str, *args, durable: bool, **kwargs) -> None:
        if self._closed:
            raise JournalError("journal writer is closed")
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        await self._queue.put(_WriteOp(method, args, kwargs, future, durable))
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
            await self._queue.put(None)
            try:
                await asyncio.wait_for(self._task, timeout)
            except TimeoutError:
                self._task.cancel()
        await asyncio.to_thread(self._store.close)
