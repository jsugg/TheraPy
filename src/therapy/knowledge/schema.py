"""Ordered SQLite migrations for Phase 4 self-knowledge stores (SPEC Appendix A).

Migrations are centralized here so every process observes one durable component
version. Existing partial-graph databases are backed up before conversion, then
rebuilt transactionally into the privacy-safe evidence-normalized schema.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import shutil
import sqlite3
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import uuid4

SCHEMA_COMPONENT = "knowledge"
SCHEMA_VERSION = 3
TOMBSTONE_KEY_FILE = "tombstone.key"

NODE_TYPES: tuple[str, ...] = (
    "identity_fact",
    "value",
    "goal",
    "pattern",
    "preference",
    "thread",
    "person",
    "strength",
    "strategy",
    "thought_record",
    "boundary",
)

EDGE_TYPES: tuple[str, ...] = (
    "involves",
    "triggers",
    "soothes",
    "works_for",
    "failed_for",
    "supports",
    "conflicts_with",
    "instance_of",
    "about",
)

LEGACY_NODE_TYPE_MAP: dict[str, str] = {
    "identity": "identity_fact",
    "routine": "pattern",
    "trait": "pattern",
    "relationship": "person",
    "trigger": "pattern",
    "note": "pattern",
}

LEGACY_EDGE_TYPE_MAP: dict[str, str] = {
    "relates_to": "about",
    "causes": "triggers",
    "part_of": "about",
    "precedes": "about",
}

CLAIM_STATUSES: tuple[str, ...] = (
    "observation",
    "proposed",
    "confirmed",
    "rejected",
    "superseded",
    "needs_revalidation",
)


@dataclass(frozen=True, slots=True)
class Migration:
    """One ordered component migration."""

    version: int
    apply: Callable[[sqlite3.Connection], None]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def normalize_statement(value: str) -> str:
    """Return stable case/space-normalized claim text."""
    return " ".join(value.casefold().split())


def load_or_create_tombstone_key(data_dir: Path) -> bytes:
    """Load the local tombstone HMAC key, creating it with owner-only mode."""
    path = data_dir / TOMBSTONE_KEY_FILE
    if path.exists():
        key = path.read_bytes()
        if len(key) != 32:
            raise RuntimeError(f"Invalid tombstone key length in {path}") from None
        return key
    key = secrets.token_bytes(32)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        key = path.read_bytes()
        if len(key) != 32:
            raise RuntimeError(f"Invalid tombstone key length in {path}") from None
        return key
    with os.fdopen(descriptor, "wb") as output:
        output.write(key)
    return key


def tombstone_digest(key: bytes, signature: str) -> str:
    """Return a keyed, normalization-stable deletion identity."""
    return hmac.new(
        key,
        normalize_statement(signature).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def edge_base_tombstone_signature(
    type_: str,
    src_type: str,
    src_statement: str,
    dst_type: str,
    dst_statement: str,
) -> str:
    """Return endpoint-stable edge identity independent of edge wording."""
    return json.dumps(
        (
            "edge",
            canonical_edge_type(type_),
            (canonical_node_type(src_type), normalize_statement(src_statement)),
            (canonical_node_type(dst_type), normalize_statement(dst_statement)),
        ),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def canonical_node_type(value: str) -> str:
    """Map a legacy node registry value to Appendix A."""
    mapped = LEGACY_NODE_TYPE_MAP.get(value, value)
    if mapped not in NODE_TYPES:
        raise ValueError(f"Unknown node type: {value!r}")
    return mapped


def canonical_edge_type(value: str) -> str:
    """Map a legacy edge registry value to Appendix A."""
    mapped = LEGACY_EDGE_TYPE_MAP.get(value, value)
    if mapped not in EDGE_TYPES:
        raise ValueError(f"Unknown edge type: {value!r}")
    return mapped


def _table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone()
    return row is not None


def _columns(connection: sqlite3.Connection, table: str) -> set[str]:
    if not _table_exists(connection, table):
        return set()
    return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}


def _create_graph_tables(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE nodes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            statement TEXT NOT NULL,
            statement_key TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'observation',
            source TEXT NOT NULL DEFAULT 'conversation',
            never_initiate INTEGER NOT NULL DEFAULT 0,
            user_edited INTEGER NOT NULL DEFAULT 0,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            n_occurrences INTEGER NOT NULL DEFAULT 0,
            n_sessions INTEGER NOT NULL DEFAULT 0,
            n_auditable_occurrences INTEGER NOT NULL DEFAULT 0,
            n_auditable_sessions INTEGER NOT NULL DEFAULT 0,
            revalidation_due_at TEXT,
            superseded_by INTEGER REFERENCES nodes(id) ON DELETE SET NULL,
            UNIQUE(type, statement_key),
            CHECK(length(trim(statement)) > 0)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE edges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            src INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
            dst INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
            type TEXT NOT NULL,
            statement TEXT NOT NULL,
            statement_key TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'observation',
            source TEXT NOT NULL DEFAULT 'conversation',
            user_edited INTEGER NOT NULL DEFAULT 0,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            n_occurrences INTEGER NOT NULL DEFAULT 0,
            n_sessions INTEGER NOT NULL DEFAULT 0,
            n_auditable_occurrences INTEGER NOT NULL DEFAULT 0,
            n_auditable_sessions INTEGER NOT NULL DEFAULT 0,
            revalidation_due_at TEXT,
            superseded_by INTEGER REFERENCES edges(id) ON DELETE SET NULL,
            UNIQUE(src, dst, type, statement_key),
            CHECK(length(trim(statement)) > 0)
        )
        """
    )


def _create_support_tables(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS claim_evidence (
            id TEXT PRIMARY KEY,
            claim_kind TEXT NOT NULL CHECK(claim_kind IN ('node','edge')),
            claim_id INTEGER NOT NULL,
            source_type TEXT NOT NULL,
            source_session_id TEXT,
            source_turn_id INTEGER,
            observed_at TEXT NOT NULL,
            language TEXT,
            quote_text TEXT,
            source_state TEXT NOT NULL DEFAULT 'active'
                CHECK(source_state IN ('active','deleted','unavailable')),
            source_deleted_at TEXT,
            source_marker TEXT,
            extractor_version TEXT NOT NULL,
            validation_event_id INTEGER,
            evidence_key TEXT NOT NULL,
            UNIQUE(claim_kind, claim_id, evidence_key)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS lifecycle_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            claim_kind TEXT NOT NULL CHECK(claim_kind IN ('node','edge')),
            claim_id INTEGER NOT NULL,
            from_status TEXT,
            to_status TEXT NOT NULL,
            event_type TEXT NOT NULL,
            session_id TEXT,
            delivery_id TEXT,
            details_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS node_aliases (
            node_id INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
            alias TEXT NOT NULL,
            alias_key TEXT NOT NULL,
            language TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY(node_id, alias_key)
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_node_alias_key ON node_aliases(alias_key)"
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS observation_inbox (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            text TEXT NOT NULL,
            language TEXT,
            created_at TEXT NOT NULL,
            processed_at TEXT,
            distillation_run_id TEXT
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS distillation_runs (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            extractor_version TEXT NOT NULL,
            state TEXT NOT NULL CHECK(state IN ('running','succeeded','failed')),
            attempt_count INTEGER NOT NULL DEFAULT 1,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            error TEXT,
            result_json TEXT,
            UNIQUE(session_id, extractor_version)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS boundaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            kind TEXT NOT NULL CHECK(kind IN ('never_store','never_initiate')),
            value TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(kind, value)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS tombstones (
            digest TEXT PRIMARY KEY,
            claim_kind TEXT NOT NULL CHECK(claim_kind IN ('node','edge')),
            created_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS privacy_purge_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            boundary_digest TEXT NOT NULL,
            graph_claims_removed INTEGER NOT NULL,
            inbox_rows_removed INTEGER NOT NULL,
            summaries_removed INTEGER NOT NULL,
            research_documents_removed INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS pending_insights (
            id TEXT PRIMARY KEY,
            claim_kind TEXT NOT NULL CHECK(claim_kind IN ('node','edge')),
            claim_id INTEGER NOT NULL,
            proposal_event_id INTEGER NOT NULL,
            statement_snapshot TEXT NOT NULL,
            evidence_snapshot_json TEXT NOT NULL,
            proposed_at TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'queued'
                CHECK(state IN ('queued','delivered','snoozed','confirmed',
                                'rejected','dismissed')),
            delivery_token TEXT,
            delivery_session_id TEXT,
            delivered_at TEXT,
            snoozed_until TEXT,
            resolved_at TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(claim_kind, claim_id, proposal_event_id)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS insight_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            insight_id TEXT NOT NULL REFERENCES pending_insights(id)
                ON DELETE CASCADE,
            event_type TEXT NOT NULL,
            session_id TEXT,
            details_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_pending_insight_state "
        "ON pending_insights(state, proposed_at)"
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS semantic_embeddings (
            entity_kind TEXT NOT NULL CHECK(entity_kind IN ('node','episode')),
            entity_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            model_revision TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            vector BLOB NOT NULL,
            indexed_at TEXT NOT NULL,
            PRIMARY KEY(entity_kind, entity_id, model_name, model_revision)
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_evidence_claim "
        "ON claim_evidence(claim_kind, claim_id, source_state)"
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_inbox_session "
        "ON observation_inbox(session_id, processed_at, id)"
    )


def _decode_json_list(value: object) -> list[object]:
    if not isinstance(value, str):
        return []
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return []
    return cast(list[object], decoded) if isinstance(decoded, list) else []


def _json_object(value: object) -> dict[str, object] | None:
    """Narrow one decoded JSON object after validating all key types."""
    if not isinstance(value, dict):
        return None
    raw = cast(dict[object, object], value)
    if not all(isinstance(key, str) for key in raw):
        return None
    return cast(dict[str, object], raw)


def _never_store_values(connection: sqlite3.Connection) -> tuple[str, ...]:
    if not _table_exists(connection, "boundaries"):
        return ()
    return tuple(
        str(row["value"]).casefold()
        for row in connection.execute(
            "SELECT value FROM boundaries WHERE kind = 'never_store'"
        )
        if str(row["value"]).strip()
    )


def _legacy_row_contains(
    row: sqlite3.Row, boundaries: tuple[str, ...], *fields: str
) -> bool:
    content = [str(row[field]) for field in fields if field in row.keys()]
    if "quotes" in row.keys():
        for quote in _decode_json_list(row["quotes"]):
            quote_object = _json_object(quote)
            if quote_object is not None and isinstance(
                quote_object.get("text"), str
            ):
                content.append(cast(str, quote_object["text"]))
            elif isinstance(quote, str):
                content.append(quote)
    combined = "\n".join(content).casefold()
    return any(value in combined for value in boundaries)


def _database_dir(connection: sqlite3.Connection) -> Path:
    row = connection.execute("PRAGMA database_list").fetchone()
    if row is None or not str(row[2]):
        raise RuntimeError("SQLite migration requires a file-backed main database")
    return Path(str(row[2])).resolve().parent


def _migrate_legacy_tombstones(connection: sqlite3.Connection) -> None:
    if not _table_exists(connection, "legacy_tombstones"):
        return
    key = load_or_create_tombstone_key(_database_dir(connection))
    legacy_nodes = {
        int(row["id"]): row
        for row in connection.execute("SELECT id, type, statement FROM legacy_nodes")
    }
    for row in connection.execute(
        "SELECT signature, created_at FROM legacy_tombstones ORDER BY created_at"
    ):
        signature = str(row["signature"])
        kind, separator, remainder = signature.partition(":")
        if not separator or kind not in {"node", "edge"}:
            raise ValueError(f"Invalid legacy tombstone signature: {signature!r}")
        type_, separator, identity = remainder.partition(":")
        if not separator:
            raise ValueError(f"Invalid legacy tombstone signature: {signature!r}")
        if kind == "node":
            converted = (
                f"node:{canonical_node_type(type_)}:{normalize_statement(identity)}"
            )
        else:
            src_raw, arrow, dst_raw = identity.partition("->")
            if not arrow:
                raise ValueError(f"Invalid legacy tombstone signature: {signature!r}")
            try:
                src_row = legacy_nodes[int(src_raw)]
                dst_row = legacy_nodes[int(dst_raw)]
            except (KeyError, ValueError) as error:
                raise ValueError(
                    f"Legacy edge tombstone has unknown endpoints: {signature!r}"
                ) from error
            converted = edge_base_tombstone_signature(
                type_,
                str(src_row["type"]),
                str(src_row["statement"]),
                str(dst_row["type"]),
                str(dst_row["statement"]),
            )
        connection.execute(
            """
            INSERT OR IGNORE INTO tombstones (digest, claim_kind, created_at)
            VALUES (?, ?, ?)
            """,
            (tombstone_digest(key, converted), kind, str(row["created_at"])),
        )


def _legacy_evidence(
    connection: sqlite3.Connection,
    *,
    claim_kind: str,
    claim_id: int,
    row: sqlite3.Row,
) -> None:
    sessions = [str(item) for item in _decode_json_list(row["sessions"]) if item]
    quotes = [
        quote
        for item in _decode_json_list(row["quotes"])
        if (quote := _json_object(item)) is not None
    ]
    count = max(int(row["n_occurrences"]), 1)
    for index in range(count):
        quote = quotes[index] if index < len(quotes) else {}
        session_id = sessions[index % len(sessions)] if sessions else None
        quote_text = quote.get("text") if isinstance(quote.get("text"), str) else None
        language = (
            quote.get("language") if isinstance(quote.get("language"), str) else None
        )
        connection.execute(
            """
            INSERT INTO claim_evidence (
                id, claim_kind, claim_id, source_type, source_session_id,
                observed_at, language, quote_text, source_state,
                extractor_version, evidence_key
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', 'legacy-v1', ?)
            """,
            (
                uuid4().hex,
                claim_kind,
                claim_id,
                str(row["source"]),
                session_id,
                str(row["first_seen"]),
                language,
                quote_text,
                f"legacy:{claim_kind}:{row['id']}:{index}",
            ),
        )


def _copy_legacy_graph(
    connection: sqlite3.Connection, boundaries: tuple[str, ...]
) -> None:
    legacy_nodes = connection.execute(
        "SELECT * FROM legacy_nodes ORDER BY id"
    ).fetchall()
    node_id_map: dict[int, int] = {}
    for row in legacy_nodes:
        if _legacy_row_contains(row, boundaries, "statement"):
            continue
        node_type = canonical_node_type(str(row["type"]))
        statement = str(row["statement"]).strip()
        key = normalize_statement(statement)
        existing = connection.execute(
            "SELECT id FROM nodes WHERE type = ? AND statement_key = ?",
            (node_type, key),
        ).fetchone()
        if existing is not None:
            node_id_map[int(row["id"])] = int(existing["id"])
            _legacy_evidence(
                connection,
                claim_kind="node",
                claim_id=int(existing["id"]),
                row=row,
            )
            continue
        status = "proposed" if str(row["status"]) == "pattern" else str(row["status"])
        connection.execute(
            """
            INSERT INTO nodes (
                id, type, statement, statement_key, status, source,
                never_initiate, user_edited, first_seen, last_seen
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(row["id"]),
                node_type,
                statement,
                key,
                status,
                str(row["source"]),
                int(row["never_initiate"]),
                int(row["user_edited"]),
                str(row["first_seen"]),
                str(row["last_seen"]),
            ),
        )
        node_id_map[int(row["id"])] = int(row["id"])
        _legacy_evidence(
            connection, claim_kind="node", claim_id=int(row["id"]), row=row
        )

    for row in connection.execute("SELECT * FROM legacy_edges ORDER BY id"):
        src = node_id_map.get(int(row["src"]))
        dst = node_id_map.get(int(row["dst"]))
        if src is None or dst is None:
            continue
        if _legacy_row_contains(row, boundaries, "statement"):
            continue
        edge_type = canonical_edge_type(str(row["type"]))
        statement = str(row["statement"]).strip()
        if not statement:
            endpoint_rows = connection.execute(
                "SELECT id, statement FROM nodes WHERE id IN (?, ?)", (src, dst)
            ).fetchall()
            endpoints = {
                int(item["id"]): str(item["statement"]) for item in endpoint_rows
            }
            statement = f"{endpoints[src]} {edge_type} {endpoints[dst]}"
        key = normalize_statement(statement)
        existing = connection.execute(
            """
            SELECT id FROM edges
            WHERE src = ? AND dst = ? AND type = ? AND statement_key = ?
            """,
            (src, dst, edge_type, key),
        ).fetchone()
        edge_id = int(existing["id"]) if existing is not None else int(row["id"])
        if existing is None:
            status = (
                "proposed" if str(row["status"]) == "pattern" else str(row["status"])
            )
            connection.execute(
                """
                INSERT INTO edges (
                    id, src, dst, type, statement, statement_key, status,
                    source, user_edited, first_seen, last_seen
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    edge_id,
                    src,
                    dst,
                    edge_type,
                    statement,
                    key,
                    status,
                    str(row["source"]),
                    int(row["user_edited"]),
                    str(row["first_seen"]),
                    str(row["last_seen"]),
                ),
            )
        _legacy_evidence(connection, claim_kind="edge", claim_id=edge_id, row=row)


def _refresh_counts(connection: sqlite3.Connection, table: str, kind: str) -> None:
    connection.execute(
        f"""
        UPDATE {table}
        SET n_occurrences = (
                SELECT count(*) FROM claim_evidence e
                WHERE e.claim_kind = ? AND e.claim_id = {table}.id
            ),
            n_sessions = (
                SELECT count(DISTINCT coalesce(e.source_session_id, e.source_marker))
                FROM claim_evidence e
                WHERE e.claim_kind = ? AND e.claim_id = {table}.id
                  AND coalesce(e.source_session_id, e.source_marker) IS NOT NULL
            ),
            n_auditable_occurrences = (
                SELECT count(*) FROM claim_evidence e
                WHERE e.claim_kind = ? AND e.claim_id = {table}.id
                  AND e.source_state = 'active'
            ),
            n_auditable_sessions = (
                SELECT count(DISTINCT e.source_session_id)
                FROM claim_evidence e
                WHERE e.claim_kind = ? AND e.claim_id = {table}.id
                  AND e.source_state = 'active' AND e.source_session_id IS NOT NULL
            )
        """,
        (kind, kind, kind, kind),
    )


def _phase4_graph(connection: sqlite3.Connection) -> None:
    legacy = "quotes" in _columns(connection, "nodes")
    boundaries = _never_store_values(connection)
    if legacy:
        connection.execute("ALTER TABLE nodes RENAME TO legacy_nodes")
        connection.execute("ALTER TABLE edges RENAME TO legacy_edges")
        if _table_exists(connection, "observation_inbox"):
            connection.execute(
                "ALTER TABLE observation_inbox RENAME TO legacy_observation_inbox"
            )
        if _table_exists(connection, "tombstones"):
            connection.execute("ALTER TABLE tombstones RENAME TO legacy_tombstones")

    _create_graph_tables(connection)
    _create_support_tables(connection)

    if legacy:
        _copy_legacy_graph(connection, boundaries)
        _migrate_legacy_tombstones(connection)
        if _table_exists(connection, "legacy_observation_inbox"):
            columns = _columns(connection, "legacy_observation_inbox")
            for row in connection.execute(
                "SELECT * FROM legacy_observation_inbox ORDER BY id"
            ):
                if _legacy_row_contains(row, boundaries, "text"):
                    continue
                processed_at = (
                    str(row["created_at"])
                    if "processed" in columns and int(row["processed"]) == 1
                    else None
                )
                connection.execute(
                    """
                INSERT INTO observation_inbox (
                    id, session_id, text, language, created_at, processed_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(row["id"]),
                        row["session_id"],
                        str(row["text"]),
                        str(row["language"]),
                        str(row["created_at"]),
                        processed_at,
                    ),
                )
        _refresh_counts(connection, "nodes", "node")
        _refresh_counts(connection, "edges", "edge")
        connection.execute("DROP TABLE legacy_edges")
        connection.execute("DROP TABLE legacy_nodes")
        if _table_exists(connection, "legacy_observation_inbox"):
            connection.execute("DROP TABLE legacy_observation_inbox")
        if _table_exists(connection, "legacy_tombstones"):
            connection.execute("DROP TABLE legacy_tombstones")


def _phase4_proactivity(connection: sqlite3.Connection) -> None:
    """Add restart-safe, owner-controlled proactivity stores."""
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS proactivity_settings (
            channel TEXT PRIMARY KEY
                CHECK(channel IN ('push','greeting','check_in','digest')),
            enabled INTEGER NOT NULL DEFAULT 0 CHECK(enabled IN (0,1)),
            timezone TEXT NOT NULL DEFAULT 'UTC',
            quiet_start TEXT NOT NULL DEFAULT '22:00',
            quiet_end TEXT NOT NULL DEFAULT '08:00',
            schedule_time TEXT NOT NULL DEFAULT '18:00',
            schedule_day INTEGER NOT NULL DEFAULT 6 CHECK(schedule_day BETWEEN 0 AND 6),
            frequency TEXT NOT NULL DEFAULT 'weekly'
                CHECK(frequency IN ('daily','weekly')),
            topic TEXT,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS proactivity_jobs (
            id TEXT PRIMARY KEY,
            channel TEXT NOT NULL
                CHECK(channel IN ('push','greeting','check_in','digest')),
            due_at TEXT NOT NULL,
            idempotency_key TEXT NOT NULL UNIQUE,
            topic TEXT,
            payload_json TEXT NOT NULL DEFAULT '{}',
            state TEXT NOT NULL DEFAULT 'pending'
                CHECK(state IN ('pending','processing','retry','delivered',
                                'suppressed','failed')),
            attempt_count INTEGER NOT NULL DEFAULT 0,
            next_attempt_at TEXT,
            result_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            delivered_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_proactivity_jobs_due
            ON proactivity_jobs(state, next_attempt_at, due_at);
        CREATE TABLE IF NOT EXISTS push_subscriptions (
            id TEXT PRIMARY KEY,
            endpoint TEXT NOT NULL UNIQUE,
            p256dh TEXT NOT NULL,
            auth TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1 CHECK(active IN (0,1)),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS in_app_messages (
            id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL UNIQUE REFERENCES proactivity_jobs(id)
                ON DELETE CASCADE,
            channel TEXT NOT NULL,
            message TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'queued'
                CHECK(state IN ('queued','seen')),
            created_at TEXT NOT NULL,
            seen_at TEXT
        );
        CREATE TABLE IF NOT EXISTS digests (
            id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL UNIQUE REFERENCES proactivity_jobs(id)
                ON DELETE CASCADE,
            period_start TEXT NOT NULL,
            period_end TEXT NOT NULL,
            content TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'unread'
                CHECK(state IN ('unread','read')),
            created_at TEXT NOT NULL,
            read_at TEXT
        );
        """
    )
    now = _utc_now()
    for channel in ("push", "greeting", "check_in", "digest"):
        connection.execute(
            """
            INSERT OR IGNORE INTO proactivity_settings (channel, updated_at)
            VALUES (?, ?)
            """,
            (channel, now),
        )


def _phase4_research(connection: sqlite3.Connection) -> None:
    """Replace lexical research rows with anchored source and versioned index stores."""
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS research_documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_title TEXT NOT NULL,
            source_ref TEXT NOT NULL,
            filename TEXT NOT NULL,
            media_type TEXT NOT NULL,
            format TEXT NOT NULL CHECK(format IN ('pdf','image','html','markdown','text')),
            content_hash TEXT NOT NULL,
            original_size INTEGER NOT NULL,
            artifact_path TEXT,
            extracted_markdown TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('indexed','review_required')),
            corpus_state TEXT NOT NULL DEFAULT 'active'
                CHECK(corpus_state IN ('active','deleted')),
            ocr_metadata_json TEXT NOT NULL DEFAULT '{}',
            index_model_name TEXT,
            index_model_revision TEXT,
            index_dimension INTEGER,
            chunk_policy_version TEXT,
            ingested_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS idx_research_active_hash
            ON research_documents(content_hash) WHERE corpus_state = 'active';
        CREATE TABLE IF NOT EXISTS research_blocks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL REFERENCES research_documents(id) ON DELETE CASCADE,
            anchor TEXT NOT NULL,
            page INTEGER,
            heading TEXT,
            text TEXT NOT NULL,
            extraction_method TEXT NOT NULL CHECK(extraction_method IN ('digital','ocr','owner_edit')),
            confidence REAL,
            needs_review INTEGER NOT NULL DEFAULT 0 CHECK(needs_review IN (0,1)),
            bbox_json TEXT,
            edited_at TEXT,
            UNIQUE(doc_id, anchor)
        );
        CREATE TABLE IF NOT EXISTS research_index (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL REFERENCES research_documents(id) ON DELETE CASCADE,
            block_id INTEGER NOT NULL REFERENCES research_blocks(id) ON DELETE CASCADE,
            chunk_ord INTEGER NOT NULL,
            anchor TEXT NOT NULL,
            text TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            vector BLOB NOT NULL,
            model_name TEXT NOT NULL,
            model_revision TEXT NOT NULL,
            dimension INTEGER NOT NULL,
            chunk_policy_version TEXT NOT NULL,
            indexed_at TEXT NOT NULL,
            UNIQUE(block_id, chunk_ord, model_name, model_revision, chunk_policy_version)
        );
        CREATE INDEX IF NOT EXISTS idx_research_index_model
            ON research_index(model_name, model_revision, chunk_policy_version);
        """
    )
    if not _table_exists(connection, "research_docs"):
        return
    docs = connection.execute(
        "SELECT id, source_title, source_ref, ingested_at FROM research_docs ORDER BY id"
    ).fetchall()
    for doc in docs:
        chunks = connection.execute(
            "SELECT ord, text FROM research_chunks WHERE doc_id = ? ORDER BY ord, id",
            (doc["id"],),
        ).fetchall()
        content = "\n\n".join(str(chunk["text"]) for chunk in chunks)
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        now = str(doc["ingested_at"])
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO research_documents (
                source_title, source_ref, filename, media_type, format,
                content_hash, original_size, artifact_path, extracted_markdown,
                status, ocr_metadata_json, ingested_at, updated_at
            ) VALUES (?, ?, ?, 'text/plain', 'text', ?, ?, NULL, ?,
                      'indexed', '{}', ?, ?)
            """,
            (
                doc["source_title"],
                doc["source_ref"],
                f"legacy-{doc['id']}.txt",
                digest,
                len(content.encode("utf-8")),
                content,
                now,
                now,
            ),
        )
        if cursor.lastrowid is None:
            target = connection.execute(
                "SELECT id FROM research_documents WHERE content_hash = ?", (digest,)
            ).fetchone()
            doc_id = int(target["id"])
        else:
            doc_id = int(cursor.lastrowid)
        for order, chunk in enumerate(chunks, start=1):
            connection.execute(
                """
                INSERT OR IGNORE INTO research_blocks (
                    doc_id, anchor, page, heading, text, extraction_method,
                    confidence, needs_review
                ) VALUES (?, ?, NULL, NULL, ?, 'digital', 1.0, 0)
                """,
                (doc_id, f"legacy-block-{order}", chunk["text"]),
            )
    connection.execute("DROP TABLE research_chunks")
    connection.execute("DROP TABLE research_docs")


MIGRATIONS: tuple[Migration, ...] = (
    Migration(1, _phase4_graph),
    Migration(2, _phase4_proactivity),
    Migration(3, _phase4_research),
)


def backup_if_needed(db_path: Path, connection: sqlite3.Connection) -> Path | None:
    from therapy.observability.logging import emit_event
    from therapy.observability.telemetry import broad_span, record_metric

    with broad_span("schema.backup", component="schema", operation="backup"):
        try:
            if not _table_exists(connection, "nodes"):
                record_metric(
                    "therapy_backups_total", 1, {"outcome": "skipped"}
                )
                return None
            current = connection.execute(
                "SELECT version FROM schema_migrations WHERE component = ?",
                (SCHEMA_COMPONENT,),
            ).fetchone()
            if current is not None and int(current[0]) >= SCHEMA_VERSION:
                record_metric(
                    "therapy_backups_total", 1, {"outcome": "skipped"}
                )
                return None
            timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
            destination = db_path.with_name(
                f"{db_path.name}.pre-phase4-{timestamp}.bak"
            )
            shutil.copy2(db_path, destination)
        except Exception as exc:
            record_metric("therapy_backups_total", 1, {"outcome": "error"})
            emit_event(
                "schema.backup_failed",
                severity=logging.CRITICAL,
                component="schema",
                operation="backup",
                outcome="error",
                error_type=type(exc).__name__,
            )
            raise
        record_metric("therapy_backups_total", 1, {"outcome": "created"})
        return destination


def _bounded_schema_version(version: int) -> int:
    """Clamp externally stored versions to the finite broad-telemetry range."""
    return min(max(version, 0), SCHEMA_VERSION + 1)


def migrate_database(db_path: Path) -> Path | None:
    """Apply ordered knowledge migrations and return backup path when created."""
    from therapy.observability.logging import emit_event
    from therapy.observability.telemetry import broad_span, record_metric

    connection: sqlite3.Connection | None = None
    backup_failed = False
    try:
        connection = sqlite3.connect(db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                component TEXT PRIMARY KEY,
                version INTEGER NOT NULL,
                applied_at TEXT NOT NULL
            )
            """
        )
        row = connection.execute(
            "SELECT version FROM schema_migrations WHERE component = ?",
            (SCHEMA_COMPONENT,),
        ).fetchone()
        current = int(row["version"]) if row is not None else 0
        with broad_span(
            "schema.migrate", component="schema", operation="migrate"
        ) as span:
            if span is not None:
                span.set_attribute(
                    "schema.from_version", _bounded_schema_version(current)
                )
                span.set_attribute(
                    "schema.to_version", _bounded_schema_version(SCHEMA_VERSION)
                )
            try:
                backup = backup_if_needed(db_path, connection)
            except Exception:
                backup_failed = True
                raise
            for migration in MIGRATIONS:
                if migration.version <= current:
                    continue
                started = time.monotonic()
                previous = current
                try:
                    with broad_span(
                        "schema.migrate.step",
                        component="schema",
                        operation="migrate",
                    ) as step_span:
                        if step_span is not None:
                            step_span.set_attribute(
                                "schema.from_version",
                                _bounded_schema_version(previous),
                            )
                            step_span.set_attribute(
                                "schema.to_version",
                                _bounded_schema_version(migration.version),
                            )
                        with connection:
                            migration.apply(connection)
                            connection.execute(
                                """
                                INSERT INTO schema_migrations (
                                    component, version, applied_at
                                )
                                VALUES (?, ?, ?)
                                ON CONFLICT(component) DO UPDATE SET
                                    version = excluded.version,
                                    applied_at = excluded.applied_at
                                """,
                                (SCHEMA_COMPONENT, migration.version, _utc_now()),
                            )
                except Exception:
                    elapsed = time.monotonic() - started
                    record_metric(
                        "therapy_schema_migration_seconds",
                        elapsed,
                        {"outcome": "error"},
                    )
                    record_metric(
                        "therapy_schema_migrations_total",
                        1,
                        {"outcome": "error"},
                    )
                    record_metric(
                        "therapy_schema_migrations_total",
                        1,
                        {"outcome": "rolled_back"},
                    )
                    raise
                record_metric(
                    "therapy_schema_migration_seconds",
                    time.monotonic() - started,
                    {"outcome": "success"},
                )
                record_metric(
                    "therapy_schema_migrations_total",
                    1,
                    {"outcome": "success"},
                )
                current = migration.version
            record_metric(
                "therapy_schema_version", current, {"component": "knowledge"}
            )
            return backup
    except Exception as exc:
        if not backup_failed:
            emit_event(
                "schema.migration_failed",
                severity=logging.CRITICAL,
                component="schema",
                operation="migrate",
                outcome="error",
                error_type=type(exc).__name__,
            )
        raise
    finally:
        if connection is not None:
            connection.close()
