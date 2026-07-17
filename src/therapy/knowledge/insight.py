"""Durable longitudinal insight queue and conversational validation (Phase 4 D).

Only evidence-gated graph proposals enter this queue. One adjacent insight may
be delivered at a time; explicit es/en/pt responses resolve only the claim most
recently raised in that session, with full delivery/resolution history.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, TypedDict, cast
from uuid import uuid4

from therapy.knowledge.user_model import (
    GRADUATION_MIN_SESSIONS,
    ClaimKind,
    UserModel,
    _tokens,
)
from therapy.memory.summarizer import complete, render_transcript
from therapy.observability.model import InteractionOperation

type Recapper = Callable[[str], Awaitable[str]]
type InsightState = Literal[
    "queued", "delivered", "snoozed", "confirmed", "rejected", "dismissed"
]
type InsightMetricState = Literal[
    "proposed", "delivered", "snoozed", "dismissed", "confirmed", "rejected"
]


def _record_transition(state: InsightMetricState) -> None:
    """Record one content-free insight state transition."""
    from therapy.observability import telemetry

    telemetry.record_metric(
        "therapy_insight_transitions_total", 1, {"state": state}
    )


class InsightRecord(TypedDict):
    """Decoded owner-visible insight queue record."""

    id: str
    claim_kind: ClaimKind
    claim_id: int
    proposal_event_id: int
    statement_snapshot: str
    evidence_snapshot: dict[str, object]
    proposed_at: str
    state: InsightState
    delivery_token: str | None
    delivery_session_id: str | None
    delivered_at: str | None
    snoozed_until: str | None
    resolved_at: str | None
    created_at: str
    updated_at: str


class DeliveredInsight(TypedDict):
    """Reflection payload appended immediately before current user turn."""

    insight_id: str
    delivery_token: str
    claim_kind: ClaimKind
    claim_id: int
    reflection: str


RECAP_PROMPT = """Write a brief end-of-session reflection for the user: two or
three sentences naming what came up and any thread left open. Address the user
as "you". Phrase observations as data, never as diagnosis. No advice unless the
user asked. Plain prose, no lists."""

_CONFIRM = {
    "yes",
    "yes that fits",
    "that fits",
    "that is accurate",
    "thats accurate",
    "correct",
    "si",
    "eso encaja",
    "es correcto",
    "sim",
    "isso faz sentido",
    "isso esta certo",
}
_REJECT = {
    "no",
    "not really",
    "that is not accurate",
    "thats not accurate",
    "incorrect",
    "no encaja",
    "eso no es correcto",
    "nao",
    "não",
    "isso nao faz sentido",
    "isso não faz sentido",
}
_SNOOZE = {
    "later",
    "not now",
    "ask me later",
    "despues",
    "después",
    "ahora no",
    "mais tarde",
    "agora nao",
    "agora não",
}
_PUNCTUATION_RE = re.compile(r"[^\w\sáéíóúüñãõâêôç]", re.UNICODE)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _normalize_reply(value: str) -> str:
    return " ".join(_PUNCTUATION_RE.sub("", value.casefold()).split())


def cross_session_patterns(model: UserModel) -> list[dict[str, object]]:
    """Return recurring non-private claims ranked by auditable evidence."""
    recurring = [
        cast(dict[str, object], node)
        for node in model.nodes()
        if not node["never_initiate"]
        and node["status"] not in {"rejected", "superseded"}
        and node["n_auditable_sessions"] >= GRADUATION_MIN_SESSIONS
    ]
    recurring.sort(
        key=lambda node: (
            int(node["n_auditable_sessions"]),
            int(node["n_auditable_occurrences"]),
        ),
        reverse=True,
    )
    return recurring


class InsightService:
    """Persistent proposal delivery and exact conversational resolution."""

    def __init__(self, model: UserModel, data_dir: Path | None = None) -> None:
        self.model = model
        self.data_dir = data_dir or model.data_dir
        self._db_path = self.data_dir / "therapy.db"

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=30000")
            yield connection
        finally:
            connection.close()

    def _record_queue_depth(self) -> None:
        """Record the number of records currently eligible for delivery."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT count(*) AS count FROM pending_insights WHERE state = 'queued'"
            ).fetchone()
        from therapy.observability import telemetry

        telemetry.record_metric("therapy_insight_queue_depth", int(row["count"]), {})

    def sync_proposals(self) -> int:
        """Materialize graph proposal events into durable queue records."""
        created = 0
        for event in self.model.lifecycle_events():
            if event["event_type"] != "longitudinal_judgment":
                continue
            kind = cast(ClaimKind, event["claim_kind"])
            claim_id = int(event["claim_id"])
            claim = (
                self.model.get_node(claim_id)
                if kind == "node"
                else self.model.get_edge(claim_id)
            )
            if claim is None or claim["status"] != "proposed":
                continue
            evidence = self.model.evidence(kind, claim_id)
            snapshot = {
                "evidence_ids": [item["id"] for item in evidence],
                "auditable_occurrences": claim["n_auditable_occurrences"],
                "auditable_sessions": claim["n_auditable_sessions"],
                "source_states": [item["source_state"] for item in evidence],
            }
            now = _utc_now()
            with self._connect() as connection:
                with connection:
                    cursor = connection.execute(
                        """
                        INSERT OR IGNORE INTO pending_insights (
                            id, claim_kind, claim_id, proposal_event_id,
                            statement_snapshot, evidence_snapshot_json,
                            proposed_at, state, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)
                        """,
                        (
                            uuid4().hex,
                            kind,
                            claim_id,
                            int(event["id"]),
                            claim["statement"],
                            json.dumps(snapshot, separators=(",", ":")),
                            event["created_at"],
                            now,
                            now,
                        ),
                    )
                    inserted = cursor.rowcount > 0
                    created += int(inserted)
            if inserted:
                _record_transition("proposed")
                self._record_queue_depth()
        return created

    def list(self, *, state: InsightState | None = None) -> list[InsightRecord]:
        """Return queue records, oldest proposal first."""
        query = "SELECT * FROM pending_insights"
        params: tuple[str, ...] = ()
        if state is not None:
            query += " WHERE state = ?"
            params = (state,)
        query += " ORDER BY proposed_at, id"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._shape(row) for row in rows]

    @staticmethod
    def _shape(row: sqlite3.Row) -> InsightRecord:
        item = dict(row)
        item["evidence_snapshot"] = json.loads(str(item.pop("evidence_snapshot_json")))
        return cast(InsightRecord, item)

    def _eligible_queue(self) -> list[InsightRecord]:
        now = _utc_now()
        with self._connect() as connection:
            with connection:
                cursor = connection.execute(
                    """
                    UPDATE pending_insights SET state = 'queued', updated_at = ?
                    WHERE state = 'snoozed' AND snoozed_until <= ?
                    """,
                    (now, now),
                )
        if cursor.rowcount:
            self._record_queue_depth()
        return self.list(state="queued")

    def next_for_topic(
        self,
        topic: str,
        session_id: str,
        *,
        adjacent_claim_ids: set[int] | None = None,
    ) -> DeliveredInsight | None:
        """Deliver one adjacent queued proposal, never a second unresolved one."""
        if not topic.strip() or not session_id:
            return None
        self.sync_proposals()
        with self._connect() as connection:
            outstanding = connection.execute(
                """
                SELECT 1 FROM pending_insights
                WHERE state = 'delivered' AND delivery_session_id = ?
                """,
                (session_id,),
            ).fetchone()
        if outstanding is not None:
            return None
        topic_tokens = _tokens(topic)
        selected = next(
            (
                insight
                for insight in self._eligible_queue()
                if topic_tokens & _tokens(insight["statement_snapshot"])
                or (
                    adjacent_claim_ids is not None
                    and insight["claim_id"] in adjacent_claim_ids
                )
            ),
            None,
        )
        if selected is None:
            return None
        token = uuid4().hex
        now = _utc_now()
        with self._connect() as connection:
            with connection:
                cursor = connection.execute(
                    """
                    UPDATE pending_insights SET state = 'delivered',
                        delivery_token = ?, delivery_session_id = ?,
                        delivered_at = ?, updated_at = ?
                    WHERE id = ? AND state = 'queued'
                    """,
                    (token, session_id, now, now, selected["id"]),
                )
                if cursor.rowcount != 1:
                    return None
                self._history(connection, selected["id"], "delivered", session_id)
        _record_transition("delivered")
        self._record_queue_depth()
        count = selected["evidence_snapshot"]["auditable_sessions"]
        reflection = (
            f"I have noticed this across {count} conversations: "
            f"{selected['statement_snapshot']} Does that feel accurate to you?"
        )
        return {
            "insight_id": selected["id"],
            "delivery_token": token,
            "claim_kind": selected["claim_kind"],
            "claim_id": selected["claim_id"],
            "reflection": reflection,
        }

    def resolve_conversational(
        self, session_id: str, user_text: str
    ) -> InsightRecord | None:
        """Resolve only latest unresolved insight just raised in this session."""
        normalized = _normalize_reply(user_text)
        if normalized in _CONFIRM:
            target: InsightState = "confirmed"
        elif normalized in _REJECT:
            target = "rejected"
        elif normalized in _SNOOZE:
            target = "snoozed"
        else:
            return None
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM pending_insights
                WHERE state = 'delivered' AND delivery_session_id = ?
                ORDER BY delivered_at DESC, id DESC LIMIT 1
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        insight = self._shape(row)
        if target == "confirmed":
            resolved = (
                self.model.confirm_node(
                    insight["claim_id"],
                    session_id=session_id,
                    delivery_id=insight["delivery_token"],
                )
                if insight["claim_kind"] == "node"
                else self.model.confirm_edge(
                    insight["claim_id"],
                    session_id=session_id,
                    delivery_id=insight["delivery_token"],
                )
            )
        elif target == "rejected":
            resolved = (
                self.model.reject_node(
                    insight["claim_id"],
                    session_id=session_id,
                    delivery_id=insight["delivery_token"],
                )
                if insight["claim_kind"] == "node"
                else self.model.reject_edge(
                    insight["claim_id"],
                    session_id=session_id,
                    delivery_id=insight["delivery_token"],
                )
            )
        else:
            resolved = True
        if not resolved:
            return None
        now = _utc_now()
        snoozed_until = (
            (datetime.now(UTC) + timedelta(days=7)).isoformat()
            if target == "snoozed"
            else None
        )
        with self._connect() as connection:
            with connection:
                connection.execute(
                    """
                    UPDATE pending_insights SET state = ?, snoozed_until = ?,
                        resolved_at = ?, updated_at = ? WHERE id = ?
                    """,
                    (
                        target,
                        snoozed_until,
                        None if target == "snoozed" else now,
                        now,
                        insight["id"],
                    ),
                )
                self._history(connection, insight["id"], target, session_id)
                updated = connection.execute(
                    "SELECT * FROM pending_insights WHERE id = ?", (insight["id"],)
                ).fetchone()
        _record_transition(cast(InsightMetricState, target))
        self._record_queue_depth()
        return self._shape(updated)

    def snooze(self, insight_id: str, *, days: int = 7) -> bool:
        """Snooze queued/delivered proposal for bounded owner-selected period."""
        if not 1 <= days <= 365:
            raise ValueError("days must be between 1 and 365")
        until = (datetime.now(UTC) + timedelta(days=days)).isoformat()
        with self._connect() as connection:
            with connection:
                cursor = connection.execute(
                    """
                    UPDATE pending_insights SET state = 'snoozed',
                        snoozed_until = ?, updated_at = ?
                    WHERE id = ? AND state IN ('queued','delivered')
                    """,
                    (until, _utc_now(), insight_id),
                )
                if cursor.rowcount:
                    self._history(connection, insight_id, "snoozed", None)
        changed = cursor.rowcount > 0
        if changed:
            _record_transition("snoozed")
            self._record_queue_depth()
        return changed

    def resolve(
        self,
        insight_id: str,
        target: Literal["confirmed", "rejected"],
        *,
        session_id: str | None = None,
    ) -> InsightRecord | None:
        """Resolve one owner-selected queue record and its exact graph claim."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM pending_insights WHERE id = ?", (insight_id,)
            ).fetchone()
        if row is None or row["state"] not in {"queued", "delivered", "snoozed"}:
            return None
        insight = self._shape(row)
        delivery_id = insight["delivery_token"]
        if insight["claim_kind"] == "node":
            changed = (
                self.model.confirm_node(
                    insight["claim_id"],
                    session_id=session_id,
                    delivery_id=delivery_id,
                )
                if target == "confirmed"
                else self.model.reject_node(
                    insight["claim_id"],
                    session_id=session_id,
                    delivery_id=delivery_id,
                )
            )
        else:
            changed = (
                self.model.confirm_edge(
                    insight["claim_id"],
                    session_id=session_id,
                    delivery_id=delivery_id,
                )
                if target == "confirmed"
                else self.model.reject_edge(
                    insight["claim_id"],
                    session_id=session_id,
                    delivery_id=delivery_id,
                )
            )
        if not changed:
            return None
        now = _utc_now()
        with self._connect() as connection:
            with connection:
                connection.execute(
                    """
                    UPDATE pending_insights SET state = ?, resolved_at = ?,
                        snoozed_until = NULL, updated_at = ? WHERE id = ?
                    """,
                    (target, now, now, insight_id),
                )
                self._history(connection, insight_id, target, session_id)
                updated = connection.execute(
                    "SELECT * FROM pending_insights WHERE id = ?", (insight_id,)
                ).fetchone()
        _record_transition(target)
        self._record_queue_depth()
        return self._shape(updated)

    def resolve_claim(
        self,
        claim_kind: ClaimKind,
        claim_id: int,
        target: Literal["confirmed", "rejected"],
    ) -> bool:
        """Resolve the active queue record for a claim, or the bare proposal."""
        self.sync_proposals()
        active = next(
            (
                insight
                for insight in reversed(self.list())
                if insight["claim_kind"] == claim_kind
                and insight["claim_id"] == claim_id
                and insight["state"] in {"queued", "delivered", "snoozed"}
            ),
            None,
        )
        if active is not None:
            return self.resolve(active["id"], target) is not None
        if claim_kind == "node":
            callback = (
                self.model.confirm_node
                if target == "confirmed"
                else self.model.reject_node
            )
        else:
            callback = (
                self.model.confirm_edge
                if target == "confirmed"
                else self.model.reject_edge
            )
        changed = callback(claim_id)
        if changed:
            _record_transition(target)
            self._record_queue_depth()
        return changed

    def dismiss_claim(self, claim_kind: ClaimKind, claim_id: int) -> int:
        """Dismiss active queue snapshots superseded by an authoritative edit."""
        self.sync_proposals()
        return sum(
            self.dismiss(insight["id"])
            for insight in self.list()
            if insight["claim_kind"] == claim_kind
            and insight["claim_id"] == claim_id
            and insight["state"] in {"queued", "delivered", "snoozed"}
        )

    def dismiss(self, insight_id: str) -> bool:
        """Dismiss delivery without changing graph claim lifecycle."""
        now = _utc_now()
        with self._connect() as connection:
            with connection:
                cursor = connection.execute(
                    """
                    UPDATE pending_insights SET state = 'dismissed',
                        resolved_at = ?, updated_at = ?
                    WHERE id = ? AND state IN ('queued','delivered','snoozed')
                    """,
                    (now, now, insight_id),
                )
                if cursor.rowcount:
                    self._history(connection, insight_id, "dismissed", None)
        changed = cursor.rowcount > 0
        if changed:
            _record_transition("dismissed")
            self._record_queue_depth()
        return changed

    def history(self, insight_id: str) -> list[dict[str, object]]:
        """Return delivery/snooze/resolution audit history."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM insight_history WHERE insight_id = ? ORDER BY id",
                (insight_id,),
            ).fetchall()
        result: list[dict[str, object]] = []
        for row in rows:
            item = dict(row)
            item["details"] = json.loads(str(item.pop("details_json")))
            result.append(item)
        return result

    @staticmethod
    def _history(
        connection: sqlite3.Connection,
        insight_id: str,
        event_type: str,
        session_id: str | None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO insight_history (
                insight_id, event_type, session_id, details_json, created_at
            ) VALUES (?, ?, ?, '{}', ?)
            """,
            (insight_id, event_type, session_id, _utc_now()),
        )


def pending_insights(model: UserModel) -> list[InsightRecord]:
    """Return durable pending queue after syncing proposal events."""
    service = InsightService(model)
    service.sync_proposals()
    return service.list(state="queued")


def should_raise_now(insight: InsightRecord, topic: str) -> bool:
    """Return lexical adjacency result; semantic retrieval augments this in context."""
    return bool(_tokens(insight["statement_snapshot"]) & _tokens(topic))


def next_insight_for_topic(
    model: UserModel, topic: str, *, session_id: str = "preview"
) -> DeliveredInsight | None:
    """Deliver one adjacent insight for a concrete session."""
    return InsightService(model).next_for_topic(topic, session_id)


async def session_recap(
    turns: list[dict[str, Any]], *, recapper: Recapper | None = None
) -> str:
    """Generate owner-facing recap independently from internal summary."""
    if not turns:
        return ""
    transcript = render_transcript(turns)
    return (
        await recapper(transcript)
        if recapper is not None
        else await complete(
            RECAP_PROMPT,
            transcript,
            max_tokens=160,
            operation=InteractionOperation.RECAP,
        )
    )


__all__ = [
    "DeliveredInsight",
    "InsightRecord",
    "InsightService",
    "cross_session_patterns",
    "next_insight_for_topic",
    "pending_insights",
    "session_recap",
    "should_raise_now",
]
