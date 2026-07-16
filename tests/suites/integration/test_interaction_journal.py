"""Interaction journal durability contract (plan §5.3, O1 test list).

Covers pragma configuration, monotonic/idempotent transitions, duplicate and
conflict behavior, restart recovery, busy writer, torn/corrupt payloads,
retention with ACK exception, export replay, checkpointing, and the bounded
async writer's group commit.
"""

import asyncio
import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from therapy.observability.interactions import (
    InteractionRecord,
    InteractionRequest,
    InteractionResponse,
    Message,
    ProviderNative,
)
from therapy.observability.journal import (
    AsyncJournalWriter,
    JournalConflict,
    JournalError,
    JournalStore,
)
from therapy.observability.model import (
    InteractionEventKind,
    InteractionOperation,
    InteractionStatus,
    Provider,
)


def _record(interaction_id: str = "itx-j-0001") -> InteractionRecord:
    return InteractionRecord(
        interaction_id=interaction_id,
        trace_id="c" * 32,
        span_id="d" * 16,
        operation=InteractionOperation.REPLY,
        provider=Provider.ANTHROPIC,
        requested_model="claude-opus-4-8",
        actual_model="claude-opus-4-8",
        prompt_template_version="v1",
        request=InteractionRequest(
            system_instructions="sys",
            messages=(Message(role="user", content="hi"),),
        ),
        response=InteractionResponse(),
        stream=(),
        error=None,
        provider_native=ProviderNative(request={"model": "claude-opus-4-8"}),
        language="en",
        modality="voice",
        build_version="0.1.0",
        policy_version="p1",
        config_version="c1",
        started_at="2026-07-15T00:00:00+00:00",
        completed_at=None,
        status=InteractionStatus.STARTED,
    )


@pytest.fixture
def store(tmp_path: Path) -> Iterator[JournalStore]:
    journal = JournalStore(tmp_path / "journal" / "interaction-journal.sqlite3")
    yield journal
    journal.close()


def test_pragmas_and_schema(store: JournalStore) -> None:
    conn = store._conn
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {
        "interactions",
        "interaction_events",
        "interaction_exports",
        "journal_metadata",
    } <= tables
    version = conn.execute(
        "SELECT value FROM journal_metadata WHERE key='schema_version'"
    ).fetchone()[0]
    assert version == "2"


def test_lifecycle_success_and_idempotent_duplicates(store: JournalStore) -> None:
    record = _record()
    store.start_attempt(record)
    store.start_attempt(record)  # identical duplicate: fine

    store.append_stream_event(
        record.interaction_id, 0, InteractionEventKind.STREAM_DELTA, "t0", {"delta": "he"}
    )
    store.append_stream_event(
        record.interaction_id, 1, InteractionEventKind.STREAM_DELTA, "t1", {"delta": "llo"}
    )
    # identical duplicate event: idempotent
    store.append_stream_event(
        record.interaction_id, 1, InteractionEventKind.STREAM_DELTA, "t1", {"delta": "llo"}
    )
    store.finish_success(record.interaction_id, {"completion": "hello"})
    store.finish_success(record.interaction_id, {"completion": "hello"})  # idempotent

    loaded = store.load(record.interaction_id)
    assert loaded is not None
    assert loaded["interaction"]["status"] == "succeeded"
    assert [event["sequence"] for event in loaded["events"]] == [0, 1]
    assert store.verify_checksums(record.interaction_id)


def test_conflicting_duplicates_fail_visibly(store: JournalStore) -> None:
    record = _record()
    store.start_attempt(record)
    # same id with different content is a visible conflict
    conflicting = _record().with_status(InteractionStatus.STARTED, language="es")
    with pytest.raises(JournalConflict):
        store.start_attempt(conflicting)

    store.append_stream_event(
        record.interaction_id, 0, InteractionEventKind.STREAM_DELTA, "t0", {"delta": "a"}
    )
    with pytest.raises(JournalConflict):
        store.append_stream_event(
            record.interaction_id, 0, InteractionEventKind.STREAM_DELTA, "t0", {"delta": "b"}
        )


def test_terminal_states_are_monotonic(store: JournalStore) -> None:
    record = _record()
    store.start_attempt(record)
    store.finish_error(record.interaction_id, {"provider_type": "overloaded"})
    with pytest.raises(JournalConflict):
        store.finish_success(record.interaction_id, {"completion": "late"})
    with pytest.raises(JournalConflict):
        store.mark_incomplete(record.interaction_id, "too-late")
    # events after terminal are conflicts too
    with pytest.raises(JournalConflict):
        store.append_stream_event(
            record.interaction_id, 5, InteractionEventKind.STREAM_DELTA, "t", {"d": "x"}
        )


def test_unknown_interaction_rejected(store: JournalStore) -> None:
    with pytest.raises(JournalConflict):
        store.append_stream_event(
            "itx-ghost", 0, InteractionEventKind.STREAM_DELTA, "t", {}
        )
    with pytest.raises(JournalConflict):
        store.finish_success("itx-ghost", {})


def test_restart_recovery_marks_incomplete_never_success(tmp_path: Path) -> None:
    path = tmp_path / "journal.sqlite3"
    first = JournalStore(path)
    record = _record()
    first.start_attempt(record)
    first.append_stream_event(
        record.interaction_id, 0, InteractionEventKind.STREAM_DELTA, "t0", {"delta": "par"}
    )
    # simulate process death: no terminal write, no clean close
    first._conn.close()

    second = JournalStore(path)
    recovered = second.recover()
    assert recovered == 1
    loaded = second.load(record.interaction_id)
    assert loaded["interaction"]["status"] == "incomplete"
    assert "persisted_events=1" in loaded["interaction"]["terminal_json"]
    assert "last_sequence=0" in loaded["interaction"]["terminal_json"]
    # partial stream evidence is preserved
    assert [event["sequence"] for event in loaded["events"]] == [0]
    second.close()


def test_busy_writer_is_tolerated(tmp_path: Path) -> None:
    """A long-lived reader must not break writes (WAL + busy timeout)."""
    path = tmp_path / "journal.sqlite3"
    store = JournalStore(path, busy_timeout_ms=2000)
    reader = sqlite3.connect(path)
    reader.execute("BEGIN")
    reader.execute("SELECT COUNT(*) FROM interactions").fetchone()
    try:
        store.start_attempt(_record())  # must succeed despite the open reader
        loaded = store.load("itx-j-0001")
        assert loaded is not None
    finally:
        reader.close()
        store.close()


def test_torn_payload_detected_by_checksum(store: JournalStore) -> None:
    record = _record()
    store.start_attempt(record)
    store.append_stream_event(
        record.interaction_id, 0, InteractionEventKind.STREAM_DELTA, "t0", {"delta": "x"}
    )
    store._conn.execute(
        "UPDATE interaction_events SET payload_json='{\"delta\": \"TORN' "
        "WHERE interaction_id=?",
        (record.interaction_id,),
    )
    with pytest.raises(ValueError, match="Unterminated string"):
        store.verify_checksums(record.interaction_id)  # torn JSON surfaces

    store._conn.execute(
        "UPDATE interaction_events SET payload_json='{\"delta\": \"tampered\"}' "
        "WHERE interaction_id=?",
        (record.interaction_id,),
    )
    assert store.verify_checksums(record.interaction_id) is False


def test_retention_preserves_unacknowledged(store: JournalStore) -> None:
    old = _record("itx-old")
    ancient = old.with_status(InteractionStatus.STARTED)
    store.start_attempt(ancient)
    store.finish_success("itx-old", {"completion": "done"})
    # backdate it beyond retention
    store._conn.execute(
        "UPDATE interactions SET started_at='2020-01-01T00:00:00+00:00' "
        "WHERE interaction_id='itx-old'"
    )

    # with a configured backend and no ACK: preserved regardless of age
    deleted = store.apply_retention(30, require_ack_backend="phoenix")
    assert deleted == 0
    assert store.load("itx-old") is not None

    # after idempotent ACK: eligible
    store.record_export_attempt(
        "itx-old", "phoenix", acknowledged=True, backend_record_id="p1"
    )
    store.record_export_attempt("itx-old", "phoenix", acknowledged=True)  # no-op
    deleted = store.apply_retention(30, require_ack_backend="phoenix")
    assert deleted == 1
    assert store.load("itx-old") is None


def test_export_replay_cursor(store: JournalStore) -> None:
    for index in range(3):
        record = _record(f"itx-exp-{index}")
        store.start_attempt(record)
        store.finish_success(f"itx-exp-{index}", {"completion": str(index)})

    pending = store.pending_exports("phoenix")
    assert pending == ["itx-exp-0", "itx-exp-1", "itx-exp-2"]

    store.record_export_attempt("itx-exp-0", "phoenix", acknowledged=True)
    store.record_export_attempt(
        "itx-exp-1",
        "phoenix",
        acknowledged=False,
        error_type="ConnectTimeout",
        next_attempt_at="2999-01-01T00:00:00+00:00",
    )
    pending = store.pending_exports("phoenix")
    assert pending == ["itx-exp-2"]  # acked gone; backoff future gone

    health = store.health()
    assert health.total_records == 3
    assert health.unexported_records == 2  # itx-exp-1 (pending) + itx-exp-2


def test_disk_full_fails_visibly_and_preserves_existing_data(
    tmp_path: Path,
) -> None:
    """Simulated full disk (bounded page count): writes fail loudly, prior
    records stay intact and readable."""
    store = JournalStore(tmp_path / "journal.sqlite3")
    store.start_attempt(_record("itx-before-full"))
    store.finish_success("itx-before-full", {"completion": "kept"})

    store._conn.execute("PRAGMA max_page_count=8")  # no room to grow

    def _fill() -> None:
        for index in range(64):
            store.start_attempt(_record(f"itx-full-{index}"))
            store.finish_success(f"itx-full-{index}", {"completion": "x" * 2048})

    with pytest.raises(sqlite3.OperationalError, match="full"):
        _fill()

    store._conn.execute("PRAGMA max_page_count=1073741823")
    assert store.integrity_check() is True
    loaded = store.load("itx-before-full")
    assert loaded is not None
    assert loaded["interaction"]["status"] == "succeeded"
    store.close()


def test_checkpoint_and_integrity(store: JournalStore) -> None:
    for index in range(5):
        record = _record(f"itx-cp-{index}")
        store.start_attempt(record)
        store.finish_success(f"itx-cp-{index}", {"completion": "x"})
    store.checkpoint()
    assert store.integrity_check() is True
    assert store.health().last_checkpoint_at is not None


def test_async_writer_group_commit_and_shutdown(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = JournalStore(tmp_path / "async-journal.sqlite3")
        writer = AsyncJournalWriter(store, queue_size=16, group_commit_ms=20)
        await writer.start()
        record = _record("itx-async-1")
        await writer.start_attempt(record)
        for sequence in range(10):
            await writer.append_stream_event(
                "itx-async-1",
                sequence,
                InteractionEventKind.STREAM_DELTA,
                f"t{sequence}",
                {"delta": str(sequence)},
            )
        await writer.finish_success("itx-async-1", {"completion": "0123456789"})
        await writer.flush()

        loaded = store.load("itx-async-1")
        assert loaded["interaction"]["status"] == "succeeded"
        assert len(loaded["events"]) == 10
        assert store.verify_checksums("itx-async-1")

        # conflicts propagate through the writer
        with pytest.raises(JournalConflict):
            await writer.finish_error("itx-async-1", {"boom": True})

        await writer.close()
        # writes after close are rejected, not silently dropped
        with pytest.raises(JournalError):
            await writer.start_attempt(_record("itx-async-2"))

    asyncio.run(scenario())


def test_kill_during_stream_preserves_prefix(tmp_path: Path) -> None:
    """Simulated hard kill mid-stream: the committed prefix survives and
    recovery reports exactly how far the stream got."""
    path = tmp_path / "kill-journal.sqlite3"

    async def first_process() -> None:
        store = JournalStore(path)
        writer = AsyncJournalWriter(store, queue_size=8, group_commit_ms=10)
        await writer.start()
        await writer.start_attempt(_record("itx-kill-1"))
        for sequence in range(4):
            await writer.append_stream_event(
                "itx-kill-1",
                sequence,
                InteractionEventKind.STREAM_DELTA,
                f"t{sequence}",
                {"delta": str(sequence)},
            )
        await writer.flush()
        # hard kill: no terminal write, no close; drop everything on the floor
        store._conn.close()

    asyncio.run(first_process())

    second = JournalStore(path)
    assert second.recover() == 1
    loaded = second.load("itx-kill-1")
    assert loaded["interaction"]["status"] == "incomplete"
    assert len(loaded["events"]) == 4
    second.close()


def test_exact_reconstruction_and_tamper_detection(store: JournalStore) -> None:
    """Audit F-01/F-05: the full §5.2 envelope round-trips; corrupting the
    canonical row is detected."""
    record = _record("itx-recon")
    store.start_attempt(record)
    store.append_stream_event(
        "itx-recon", 0, InteractionEventKind.STREAM_DELTA, "t0", {"delta": "he"}
    )
    store.finish_success("itx-recon", {"completion": "he"})

    envelope = store.reconstruct("itx-recon")
    assert envelope is not None
    assert envelope["session_id"] is None
    assert envelope["requested_model"] == "claude-opus-4-8"
    assert envelope["prompt_template_version"] == "v1"
    assert envelope["language"] == "en"
    assert envelope["modality"] == "voice"
    assert envelope["build_version"] == "0.1.0"
    assert envelope["status"] == "succeeded"
    assert envelope["request"]["system_instructions"] == "sys"
    assert [e["delta"] for e in envelope["stream"]] == ["he"]
    assert envelope["terminal"]["kind"] == "success"
    assert store.verify_checksums("itx-recon")

    # tampering with the canonical envelope is now visible
    store._conn.execute(
        "UPDATE interactions SET canonical_record_json="
        "replace(canonical_record_json, 'sys', 'TAMPERED') "
        "WHERE interaction_id='itx-recon'"
    )
    assert store.verify_checksums("itx-recon") is False


def test_v1_to_v2_migration_flags_legacy_rows(tmp_path: Path) -> None:
    """Audit: older schemas migrate forward; legacy rows are visible, never
    silently reconstructed."""
    path = tmp_path / "legacy.sqlite3"
    first = JournalStore(path)
    first.start_attempt(_record("itx-legacy"))
    first.finish_success("itx-legacy", {"completion": "x"})
    # simulate a v1 database: drop the v2 column content and version stamp
    first._conn.execute(
        "UPDATE interactions SET canonical_record_json='' "
        "WHERE interaction_id='itx-legacy'"
    )
    first._conn.execute(
        "UPDATE journal_metadata SET value='1' WHERE key='schema_version'"
    )
    first._conn.close()

    second = JournalStore(path)
    version = second._conn.execute(
        "SELECT value FROM journal_metadata WHERE key='schema_version'"
    ).fetchone()[0]
    assert version == "2"
    with pytest.raises(JournalError, match="legacy v1 row"):
        second.reconstruct("itx-legacy")
    second.close()


def test_group_commit_is_one_transaction(tmp_path: Path) -> None:
    """Audit F-14: a batch with a conflicting member rolls back entirely."""
    store = JournalStore(tmp_path / "batch.sqlite3")
    store.start_attempt(_record("itx-batch"))
    store.append_stream_event(
        "itx-batch", 0, InteractionEventKind.STREAM_DELTA, "t", {"delta": "a"}
    )
    items = [
        ("itx-batch", 1, InteractionEventKind.STREAM_DELTA, "t", {"delta": "b"}),
        # sequence 0 with DIFFERENT content: a visible conflict
        ("itx-batch", 0, InteractionEventKind.STREAM_DELTA, "t", {"delta": "X"}),
    ]
    with pytest.raises(JournalConflict):
        store.append_stream_events(items)
    # the whole batch rolled back: sequence 1 was not persisted
    events = store.load("itx-batch")["events"]
    assert [e["sequence"] for e in events] == [0]
    store.close()
