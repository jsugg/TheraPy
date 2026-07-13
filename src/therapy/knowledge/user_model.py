"""Property-graph self-model: typed nodes + typed edges over SQLite.

Full schema: SPEC Appendix A. This module is framework-free (SPEC dependency
boundary — only `agent.py` imports Pipecat) and shares the local `therapy.db`
file with `memory.store`, so a single owner-controlled data directory holds
everything (SPEC §8).

Node and edge types live in extensible registries (`NODE_TYPES` / `EDGE_TYPES`)
— adding a type is config, not a migration. Nodes and edges carry the same
claim lifecycle: a canonical-English `statement`, verbatim language-tagged
`quotes` as evidence, occurrence/session counts, a `status`
(observation|pattern|confirmed), a `source` (conversation|user-stated|ser —
emotion slots in later with no schema change), timestamps, and a `user_edited`
flag.

Boundaries are enforced here: `never_store` is checked before *every* write
(nodes, edges, and the freeform observation inbox); `never_initiate` is exposed
to context assembly and the proactivity engine. Deletion writes a tombstone so
distillation cannot re-learn a removed claim; per-type decay down-weights stale
claims in the graph walk without deleting them.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Extensible registries — adding a type is configuration, not a schema change.
# The 11 starter node types (SPEC Appendix A); `note` is the general bucket the
# v1 flat facts migrate into.
NODE_TYPES: tuple[str, ...] = (
    "identity",
    "preference",
    "value",
    "goal",
    "thread",
    "routine",
    "trait",
    "relationship",
    "trigger",
    "strategy",
    "note",
)

# Starter edge catalog (typed relations between nodes).
EDGE_TYPES: tuple[str, ...] = (
    "relates_to",
    "causes",
    "supports",
    "conflicts_with",
    "part_of",
    "precedes",
    "involves",
)

STATUSES: tuple[str, ...] = ("observation", "pattern", "confirmed")
SOURCES: tuple[str, ...] = ("conversation", "user-stated", "ser")

# Graduation floor (SPEC §3): a claim is *eligible* to leave `observation` only
# once it recurs enough to be a pattern rather than a one-off.
GRADUATION_MIN_OCCURRENCES = 3
GRADUATION_MIN_SESSIONS = 2

_STATUS_WEIGHT = {"confirmed": 3.0, "pattern": 2.0, "observation": 1.0}

# Per-type decay half-life (days): how fast an unreinforced claim loses weight
# in the graph walk. Stable identity barely decays; fleeting threads fade fast.
_DECAY_HALF_LIFE_DAYS: dict[str, float] = {
    "identity": 3650.0,
    "value": 1825.0,
    "trait": 730.0,
    "preference": 365.0,
    "relationship": 365.0,
    "routine": 180.0,
    "strategy": 180.0,
    "trigger": 180.0,
    "goal": 120.0,
    "note": 120.0,
    "thread": 45.0,
}
_DEFAULT_HALF_LIFE_DAYS = 180.0

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _utc_now() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _tokens(text: str) -> set[str]:
    """Lowercase word-character tokens for lightweight relevance scoring."""
    return set(_TOKEN_RE.findall(text.lower()))


def _signature(kind: str, type_: str, statement: str) -> str:
    """Stable identity of a claim for tombstoning (case/space-insensitive)."""
    return f"{kind}:{type_}:{' '.join(statement.lower().split())}"


class UserModel:
    """Property-graph user model with claim lifecycle and boundaries."""

    def __init__(self, data_dir: Path | None = None) -> None:
        """Open (creating if needed) the graph tables in the shared db.

        Args:
            data_dir: Base directory for `therapy.db`. When omitted,
                `THERAPY_DATA_DIR` is used, then `./data` — the same
                resolution `memory.store.MemoryStore` uses, so both share
                one file.
        """
        self.data_dir = data_dir or Path(os.environ.get("THERAPY_DATA_DIR", "./data"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._db_path = self.data_dir / "therapy.db"
        self._init_schema()
        self.migrate_v1_facts()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Open a configured SQLite connection for one method call."""
        connection = sqlite3.connect(self._db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            yield connection
        finally:
            connection.close()

    def _init_schema(self) -> None:
        """Create the graph, inbox, boundary, and tombstone tables."""
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    statement TEXT NOT NULL,
                    quotes TEXT NOT NULL DEFAULT '[]',
                    n_occurrences INTEGER NOT NULL DEFAULT 1,
                    n_sessions INTEGER NOT NULL DEFAULT 0,
                    sessions TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL DEFAULT 'observation',
                    source TEXT NOT NULL DEFAULT 'conversation',
                    never_initiate INTEGER NOT NULL DEFAULT 0,
                    user_edited INTEGER NOT NULL DEFAULT 0,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    UNIQUE(type, statement)
                );

                CREATE TABLE IF NOT EXISTS edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    src INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                    dst INTEGER NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
                    type TEXT NOT NULL,
                    statement TEXT NOT NULL DEFAULT '',
                    quotes TEXT NOT NULL DEFAULT '[]',
                    n_occurrences INTEGER NOT NULL DEFAULT 1,
                    n_sessions INTEGER NOT NULL DEFAULT 0,
                    sessions TEXT NOT NULL DEFAULT '[]',
                    status TEXT NOT NULL DEFAULT 'observation',
                    source TEXT NOT NULL DEFAULT 'conversation',
                    user_edited INTEGER NOT NULL DEFAULT 0,
                    first_seen TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    UNIQUE(src, dst, type)
                );

                CREATE TABLE IF NOT EXISTS observation_inbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    text TEXT NOT NULL,
                    language TEXT,
                    created_at TEXT NOT NULL,
                    processed INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS boundaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL CHECK(kind IN ('never_store','never_initiate')),
                    value TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(kind, value)
                );

                CREATE TABLE IF NOT EXISTS tombstones (
                    signature TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                );
                """
            )

    # ------------------------------------------------------------------ #
    # Boundaries: never_store (write guard) + never_initiate (raise guard)
    # ------------------------------------------------------------------ #

    def add_boundary(self, kind: str, value: str) -> None:
        """Record a `never_store` pattern or `never_initiate` topic."""
        if kind not in {"never_store", "never_initiate"}:
            raise ValueError(f"Unknown boundary kind: {kind!r}")
        value = value.strip()
        if not value:
            raise ValueError("Boundary value must not be empty")
        with self._connect() as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO boundaries (kind, value, created_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(kind, value) DO NOTHING
                    """,
                    (kind, value, _utc_now()),
                )

    def remove_boundary(self, kind: str, value: str) -> bool:
        """Delete a boundary; return whether a row was removed."""
        with self._connect() as connection:
            with connection:
                cursor = connection.execute(
                    "DELETE FROM boundaries WHERE kind = ? AND value = ?",
                    (kind, value),
                )
        return cursor.rowcount > 0

    def boundaries(self, kind: str | None = None) -> list[dict[str, Any]]:
        """Return configured boundaries, optionally filtered by kind."""
        query = "SELECT kind, value, created_at FROM boundaries"
        params: tuple[str, ...] = ()
        if kind is not None:
            query += " WHERE kind = ?"
            params = (kind,)
        query += " ORDER BY kind ASC, value ASC"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def _never_store_patterns(self) -> list[str]:
        """The lowercase substrings that block a write."""
        return [str(b["value"]).lower() for b in self.boundaries("never_store")]

    def is_never_store(self, text: str) -> bool:
        """Whether `text` matches any `never_store` boundary (case-insensitive).

        This is the single gate every write path calls before touching the
        database — the inbox included — so a topic the user asked never to
        store cannot leak into memory (SPEC §8).
        """
        lowered = text.lower()
        return any(pattern in lowered for pattern in self._never_store_patterns())

    def never_initiate_topics(self) -> list[str]:
        """Topics the assistant must never raise on its own (context + proactivity)."""
        return [str(b["value"]) for b in self.boundaries("never_initiate")]

    # ------------------------------------------------------------------ #
    # Tombstones: deletion must prevent re-learning
    # ------------------------------------------------------------------ #

    def _is_tombstoned(self, connection: sqlite3.Connection, signature: str) -> bool:
        row = connection.execute(
            "SELECT 1 FROM tombstones WHERE signature = ?", (signature,)
        ).fetchone()
        return row is not None

    def _tombstone(self, connection: sqlite3.Connection, signature: str) -> None:
        connection.execute(
            "INSERT OR IGNORE INTO tombstones (signature, created_at) VALUES (?, ?)",
            (signature, _utc_now()),
        )

    # ------------------------------------------------------------------ #
    # Observation inbox (W2): freeform, zero schema pressure
    # ------------------------------------------------------------------ #

    def add_observation(
        self, text: str, *, session_id: str | None = None, language: str | None = None
    ) -> int | None:
        """Append a freeform observation to the inbox, or None if `never_store`.

        The inbox is written during conversation with no schema pressure;
        `distill` promotes it into nodes/edges between sessions.
        """
        text = text.strip()
        if not text or self.is_never_store(text):
            return None
        with self._connect() as connection:
            with connection:
                cursor = connection.execute(
                    """
                    INSERT INTO observation_inbox
                        (session_id, text, language, created_at, processed)
                    VALUES (?, ?, ?, ?, 0)
                    """,
                    (session_id, text, language, _utc_now()),
                )
        return cursor.lastrowid

    def pending_observations(self) -> list[dict[str, Any]]:
        """Unprocessed inbox rows, oldest first (input to distillation)."""
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, session_id, text, language, created_at
                FROM observation_inbox
                WHERE processed = 0
                ORDER BY id ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def mark_observations_processed(self, ids: list[int]) -> None:
        """Flag inbox rows consumed so distillation does not reprocess them."""
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as connection:
            with connection:
                connection.execute(
                    f"UPDATE observation_inbox SET processed = 1 "
                    f"WHERE id IN ({placeholders})",
                    tuple(ids),
                )

    # ------------------------------------------------------------------ #
    # Nodes
    # ------------------------------------------------------------------ #

    def upsert_node(
        self,
        type_: str,
        statement: str,
        *,
        source: str = "conversation",
        quotes: list[dict[str, str]] | None = None,
        session_id: str | None = None,
        never_initiate: bool = False,
    ) -> int | None:
        """Insert a node or reinforce an existing one; None if blocked.

        A `user-stated` claim is authoritative and starts `confirmed`; anything
        else starts as an `observation` and must earn its way up (`propose`
        then explicit `confirm`). Re-stating a claim bumps its occurrence count,
        records the session, and appends any new quotes — it never duplicates.

        Blocked (returns None) when the statement matches a `never_store`
        boundary or has been tombstoned by an earlier deletion.
        """
        if type_ not in NODE_TYPES:
            raise ValueError(f"Unknown node type: {type_!r}")
        if source not in SOURCES:
            raise ValueError(f"Unknown source: {source!r}")
        statement = statement.strip()
        if not statement or self.is_never_store(statement):
            return None
        signature = _signature("node", type_, statement)
        now = _utc_now()
        with self._connect() as connection:
            with connection:
                if self._is_tombstoned(connection, signature):
                    return None
                existing = connection.execute(
                    "SELECT * FROM nodes WHERE type = ? AND statement = ?",
                    (type_, statement),
                ).fetchone()
                if existing is None:
                    status = "confirmed" if source == "user-stated" else "observation"
                    sessions = [session_id] if session_id else []
                    cursor = connection.execute(
                        """
                        INSERT INTO nodes (
                            type, statement, quotes, n_occurrences, n_sessions,
                            sessions, status, source, never_initiate, user_edited,
                            first_seen, last_seen
                        ) VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?, 0, ?, ?)
                        """,
                        (
                            type_,
                            statement,
                            json.dumps(quotes or [], ensure_ascii=False),
                            len(sessions),
                            json.dumps(sessions),
                            status,
                            source,
                            int(never_initiate),
                            now,
                            now,
                        ),
                    )
                    return cursor.lastrowid
                self._reinforce_row(
                    connection, "nodes", existing, quotes, session_id, now
                )
                if never_initiate and not existing["never_initiate"]:
                    connection.execute(
                        "UPDATE nodes SET never_initiate = 1 WHERE id = ?",
                        (existing["id"],),
                    )
                return int(existing["id"])

    def _reinforce_row(
        self,
        connection: sqlite3.Connection,
        table: str,
        row: sqlite3.Row,
        quotes: list[dict[str, str]] | None,
        session_id: str | None,
        now: str,
    ) -> None:
        """Bump occurrence/session counts and merge quotes on a re-statement."""
        sessions = list(json.loads(row["sessions"]))
        if session_id and session_id not in sessions:
            sessions.append(session_id)
        merged = list(json.loads(row["quotes"]))
        for quote in quotes or []:
            if quote not in merged:
                merged.append(quote)
        connection.execute(
            f"""
            UPDATE {table}
            SET n_occurrences = n_occurrences + 1,
                n_sessions = ?,
                sessions = ?,
                quotes = ?,
                last_seen = ?
            WHERE id = ?
            """,
            (
                len(sessions),
                json.dumps(sessions),
                json.dumps(merged, ensure_ascii=False),
                now,
                row["id"],
            ),
        )

    def get_node(self, node_id: int) -> dict[str, Any] | None:
        """Return one node as a dict, or None."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM nodes WHERE id = ?", (node_id,)
            ).fetchone()
        return _node_dict(row) if row is not None else None

    def nodes(
        self, *, type_: str | None = None, status: str | None = None
    ) -> list[dict[str, Any]]:
        """Return nodes, optionally filtered by type and/or status."""
        query = "SELECT * FROM nodes"
        clauses: list[str] = []
        params: list[object] = []
        if type_ is not None:
            clauses.append("type = ?")
            params.append(type_)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY id ASC"
        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [_node_dict(row) for row in rows]

    def edit_node(
        self,
        node_id: int,
        *,
        statement: str | None = None,
        never_initiate: bool | None = None,
    ) -> bool:
        """Apply a user edit to a node (sets `user_edited`); False if unknown."""
        sets: list[str] = ["user_edited = 1", "last_seen = ?"]
        params: list[object] = [_utc_now()]
        if statement is not None:
            statement = statement.strip()
            if not statement:
                raise ValueError("statement must not be empty")
            sets.append("statement = ?")
            params.append(statement)
        if never_initiate is not None:
            sets.append("never_initiate = ?")
            params.append(int(never_initiate))
        params.append(node_id)
        with self._connect() as connection:
            with connection:
                cursor = connection.execute(
                    f"UPDATE nodes SET {', '.join(sets)} WHERE id = ?", tuple(params)
                )
        return cursor.rowcount > 0

    def delete_node(self, node_id: int) -> bool:
        """Delete a node (and its edges) and tombstone it against re-learning."""
        with self._connect() as connection:
            with connection:
                row = connection.execute(
                    "SELECT type, statement FROM nodes WHERE id = ?", (node_id,)
                ).fetchone()
                if row is None:
                    return False
                self._tombstone(
                    connection,
                    _signature("node", str(row["type"]), str(row["statement"])),
                )
                connection.execute("DELETE FROM nodes WHERE id = ?", (node_id,))
        return True

    # ------------------------------------------------------------------ #
    # Edges
    # ------------------------------------------------------------------ #

    def upsert_edge(
        self,
        src: int,
        dst: int,
        type_: str,
        *,
        statement: str = "",
        source: str = "conversation",
        quotes: list[dict[str, str]] | None = None,
        session_id: str | None = None,
    ) -> int | None:
        """Insert or reinforce a typed edge between two nodes; None if blocked."""
        if type_ not in EDGE_TYPES:
            raise ValueError(f"Unknown edge type: {type_!r}")
        if source not in SOURCES:
            raise ValueError(f"Unknown source: {source!r}")
        if statement and self.is_never_store(statement):
            return None
        signature = _signature("edge", type_, f"{src}->{dst}")
        now = _utc_now()
        with self._connect() as connection:
            with connection:
                if self._is_tombstoned(connection, signature):
                    return None
                existing = connection.execute(
                    "SELECT * FROM edges WHERE src = ? AND dst = ? AND type = ?",
                    (src, dst, type_),
                ).fetchone()
                if existing is None:
                    status = "confirmed" if source == "user-stated" else "observation"
                    sessions = [session_id] if session_id else []
                    cursor = connection.execute(
                        """
                        INSERT INTO edges (
                            src, dst, type, statement, quotes, n_occurrences,
                            n_sessions, sessions, status, source, user_edited,
                            first_seen, last_seen
                        ) VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, ?, 0, ?, ?)
                        """,
                        (
                            src,
                            dst,
                            type_,
                            statement,
                            json.dumps(quotes or [], ensure_ascii=False),
                            len(sessions),
                            json.dumps(sessions),
                            status,
                            source,
                            now,
                            now,
                        ),
                    )
                    return cursor.lastrowid
                self._reinforce_row(
                    connection, "edges", existing, quotes, session_id, now
                )
                return int(existing["id"])

    def edges(self, *, status: str | None = None) -> list[dict[str, Any]]:
        """Return edges, optionally filtered by status."""
        query = "SELECT * FROM edges"
        params: tuple[str, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY id ASC"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [_edge_dict(row) for row in rows]

    def delete_edge(self, edge_id: int) -> bool:
        """Delete one edge and tombstone it against re-learning."""
        with self._connect() as connection:
            with connection:
                row = connection.execute(
                    "SELECT src, dst, type FROM edges WHERE id = ?", (edge_id,)
                ).fetchone()
                if row is None:
                    return False
                self._tombstone(
                    connection,
                    _signature("edge", str(row["type"]), f"{row['src']}->{row['dst']}"),
                )
                connection.execute("DELETE FROM edges WHERE id = ?", (edge_id,))
        return True

    # ------------------------------------------------------------------ #
    # Graduation engine (W2): observation -> pattern -> confirmed
    # ------------------------------------------------------------------ #

    def is_eligible(self, node: dict[str, Any]) -> bool:
        """Whether a claim clears the mechanical floor to become a pattern."""
        return (
            int(node["n_occurrences"]) >= GRADUATION_MIN_OCCURRENCES
            and int(node["n_sessions"]) >= GRADUATION_MIN_SESSIONS
        )

    def propose(self, node_id: int) -> bool:
        """Graduate an eligible `observation` to `pattern` (LLM-judged step).

        The mechanical floor gates eligibility; in the live loop an LLM judges
        whether the recurrence is a genuine pattern before calling this. A
        pattern is *proposed*, not confirmed — it still needs the user.
        """
        node = self.get_node(node_id)
        if node is None or node["status"] != "observation":
            return False
        if not self.is_eligible(node):
            return False
        return self._set_node_status(node_id, "pattern")

    def confirm_node(self, node_id: int) -> bool:
        """Flip a claim to `confirmed` — the explicit user-validation gate.

        This is the *only* path to `confirmed`; the graduation floor alone can
        never mint it (SPEC §3, R1 risk mitigation).
        """
        return self._set_node_status(node_id, "confirmed")

    def reject_node(self, node_id: int) -> bool:
        """Demote a proposed pattern back to `observation` (user said no)."""
        node = self.get_node(node_id)
        if node is None:
            return False
        return self._set_node_status(node_id, "observation")

    def _set_node_status(self, node_id: int, status: str) -> bool:
        if status not in STATUSES:
            raise ValueError(f"Unknown status: {status!r}")
        with self._connect() as connection:
            with connection:
                cursor = connection.execute(
                    "UPDATE nodes SET status = ?, last_seen = ? WHERE id = ?",
                    (status, _utc_now(), node_id),
                )
        return cursor.rowcount > 0

    def confirm_edge(self, edge_id: int) -> bool:
        """Flip an edge to `confirmed` via explicit user validation."""
        with self._connect() as connection:
            with connection:
                cursor = connection.execute(
                    "UPDATE edges SET status = ?, last_seen = ? WHERE id = ?",
                    ("confirmed", _utc_now(), edge_id),
                )
        return cursor.rowcount > 0

    def pending_insights(self) -> list[dict[str, Any]]:
        """Proposed patterns awaiting the user's confirm/reject (W4 inbox)."""
        return self.nodes(status="pattern")

    # ------------------------------------------------------------------ #
    # Decay + graph walk (W3): relevance-scored context assembly
    # ------------------------------------------------------------------ #

    def _decay_weight(self, node: dict[str, Any], now: datetime) -> float:
        """Recency weight in (0, 1]; per-type half-life sets the fade rate."""
        half_life = _DECAY_HALF_LIFE_DAYS.get(str(node["type"]), _DEFAULT_HALF_LIFE_DAYS)
        try:
            last = datetime.fromisoformat(str(node["last_seen"]))
        except ValueError:
            return 1.0
        age_days = max(0.0, (now - last).total_seconds() / 86400.0)
        return 0.5 ** (age_days / half_life)

    def graph_walk(
        self, topic: str, *, k: int = 5, now: datetime | None = None
    ) -> dict[str, Any]:
        """Walk from `topic` to the top-K relevant nodes with confirmed edges.

        Relevance = token overlap with the topic, weighted by status priority
        (confirmed > pattern > observation) and per-type recency decay. Nodes
        flagged `never_initiate` are omitted entirely — they may be held, never
        surfaced. Confirmed edges among the selected nodes are then pulled in so
        the walk carries relationships, not just isolated facts.

        Returns a dict with `nodes` (scored, ranked), `edges` (confirmed, among
        the selected set), and the `never_initiate` topic list.
        """
        now = now or datetime.now(UTC)
        topic_tokens = _tokens(topic)
        scored: list[tuple[float, dict[str, Any]]] = []
        for node in self.nodes():
            if node["never_initiate"]:
                continue
            overlap = len(topic_tokens & _tokens(str(node["statement"])))
            if overlap == 0:
                continue
            weight = _STATUS_WEIGHT.get(str(node["status"]), 1.0)
            score = overlap * weight * self._decay_weight(node, now)
            scored.append((score, node))
        scored.sort(key=lambda item: (-item[0], int(item[1]["id"])))
        selected = [node for _, node in scored[:k]]
        selected_ids = {int(node["id"]) for node in selected}
        confirmed_edges = [
            edge
            for edge in self.edges(status="confirmed")
            if int(edge["src"]) in selected_ids and int(edge["dst"]) in selected_ids
        ]
        return {
            "nodes": selected,
            "edges": confirmed_edges,
            "never_initiate": self.never_initiate_topics(),
        }

    def assemble_context(
        self, topic: str = "", *, k: int = 5, now: datetime | None = None
    ) -> dict[str, Any]:
        """Build the graph-aware continuity payload for a conversation (W3).

        Always present: identity, standing preferences, and the
        `never_initiate` list. Topic-relevant material comes from the graph
        walk. This replaces the Phase-2 "flat facts + summaries" injection;
        `render_context` turns it into the system-message text.
        """
        identity = [n for n in self.nodes(type_="identity") if not n["never_initiate"]]
        preferences = [
            n
            for n in self.nodes(type_="preference")
            if not n["never_initiate"] and n["status"] in ("pattern", "confirmed")
        ]
        if topic:
            walk = self.graph_walk(topic, k=k, now=now)
        else:
            walk = {
                "nodes": [],
                "edges": [],
                "never_initiate": self.never_initiate_topics(),
            }
        return {
            "identity": identity,
            "preferences": preferences,
            "never_initiate": walk["never_initiate"],
            "walk_nodes": walk["nodes"],
            "walk_edges": walk["edges"],
        }

    # ------------------------------------------------------------------ #
    # Migration + export
    # ------------------------------------------------------------------ #

    def migrate_v1_facts(self) -> int:
        """Import v1 `facts` rows as `observation` nodes (idempotent, no loss).

        The flat `facts` table (Phase 2) is retired in favor of the graph. Each
        fact becomes exactly one node, preserving its occurrence count and
        timestamps; re-running is a no-op (statements already migrated are left
        untouched). `never_store` still applies, so a boundary added since can
        keep a fact out — that is enforcement, not loss.
        """
        with self._connect() as connection:
            has_facts = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='facts'"
            ).fetchone()
            if has_facts is None:
                return 0
            facts = connection.execute(
                """
                SELECT statement, kind, first_seen, last_seen, n_occurrences
                FROM facts ORDER BY id ASC
                """
            ).fetchall()
            patterns = self._never_store_patterns()
            migrated = 0
            with connection:
                for fact in facts:
                    statement = str(fact["statement"]).strip()
                    if not statement:
                        continue
                    if any(p in statement.lower() for p in patterns):
                        continue
                    type_ = _fact_kind_to_type(str(fact["kind"]))
                    signature = _signature("node", type_, statement)
                    if self._is_tombstoned(connection, signature):
                        continue
                    cursor = connection.execute(
                        """
                        INSERT OR IGNORE INTO nodes (
                            type, statement, quotes, n_occurrences, n_sessions,
                            sessions, status, source, never_initiate, user_edited,
                            first_seen, last_seen
                        ) VALUES (?, ?, '[]', ?, 0, '[]', 'observation',
                                  'conversation', 0, 0, ?, ?)
                        """,
                        (
                            type_,
                            statement,
                            int(fact["n_occurrences"]),
                            str(fact["first_seen"]),
                            str(fact["last_seen"]),
                        ),
                    )
                    if cursor.rowcount:
                        migrated += 1
        return migrated

    def export_all(self) -> dict[str, Any]:
        """JSON-serializable snapshot of the whole model for the data-export CLI."""
        return {
            "nodes": self.nodes(),
            "edges": self.edges(),
            "boundaries": self.boundaries(),
            "inbox": self.pending_observations(),
        }

    def delete_all(self) -> None:
        """Wipe the graph, inbox, boundaries, and tombstones (owner erase)."""
        with self._connect() as connection:
            with connection:
                connection.execute("DELETE FROM edges")
                connection.execute("DELETE FROM nodes")
                connection.execute("DELETE FROM observation_inbox")
                connection.execute("DELETE FROM boundaries")
                connection.execute("DELETE FROM tombstones")


def _fact_kind_to_type(kind: str) -> str:
    """Map a v1 `facts.kind` onto a graph node type (defaults to `note`)."""
    return kind if kind in NODE_TYPES else "note"


def _node_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Shape a node row for callers, decoding JSON columns and the flags."""
    node = dict(row)
    node["quotes"] = json.loads(node["quotes"])
    node["sessions"] = json.loads(node["sessions"])
    node["never_initiate"] = bool(node["never_initiate"])
    node["user_edited"] = bool(node["user_edited"])
    return node


def _edge_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Shape an edge row for callers, decoding JSON columns."""
    edge = dict(row)
    edge["quotes"] = json.loads(edge["quotes"])
    edge["sessions"] = json.loads(edge["sessions"])
    edge["user_edited"] = bool(edge["user_edited"])
    return edge


def render_context(assembled: dict[str, Any]) -> str | None:
    """Render assembled graph context as the LLM continuity system message.

    Older history reaches the model only as this distilled, graph-derived
    context — never verbatim transcripts (SPEC §8). Returns None when the model
    is empty so a first conversation carries no scaffolding.
    """
    identity = list(assembled.get("identity", []))  # type: ignore[arg-type]
    preferences = list(assembled.get("preferences", []))  # type: ignore[arg-type]
    never_initiate = list(assembled.get("never_initiate", []))  # type: ignore[arg-type]
    walk_nodes = list(assembled.get("walk_nodes", []))  # type: ignore[arg-type]
    walk_edges = list(assembled.get("walk_edges", []))  # type: ignore[arg-type]
    if not any([identity, preferences, never_initiate, walk_nodes]):
        return None

    parts = ["# What you know about the user"]
    if identity:
        parts.append("Identity:\n" + "\n".join(f"- {n['statement']}" for n in identity))
    if preferences:
        parts.append(
            "Preferences:\n" + "\n".join(f"- {n['statement']}" for n in preferences)
        )
    if walk_nodes:
        by_id = {int(n["id"]): n for n in walk_nodes}
        lines = [f"- ({n['status']}) {n['statement']}" for n in walk_nodes]
        for edge in walk_edges:
            src = by_id.get(int(edge["src"]))
            dst = by_id.get(int(edge["dst"]))
            if src and dst:
                label = edge["statement"] or str(edge["type"]).replace("_", " ")
                lines.append(f"- {src['statement']} — {label} → {dst['statement']}")
        parts.append("Relevant to what you're discussing now:\n" + "\n".join(lines))
    if never_initiate:
        parts.append(
            "Never raise these topics on your own (the user asked you not to); "
            "you may respond if they bring them up:\n"
            + "\n".join(f"- {topic}" for topic in never_initiate)
        )
    parts.append(
        "Use this naturally — refer back when relevant without reciting it. "
        "If the user contradicts something you remember, believe the user."
    )
    return "\n\n".join(parts)
