"""Property-graph schema, lifecycle, privacy, and provenance integration tests."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from therapy.knowledge.insight import InsightService
from therapy.knowledge.user_model import (
    EDGE_TYPES,
    NODE_TYPES,
    GraphEdge,
    GraphNode,
    UserModel,
    render_context,
)
from therapy.memory import MemoryStore
from therapy.observability.interactions import require_json_object

type MetricCall = tuple[str, float, dict[str, str]]


@pytest.fixture
def metric_calls(monkeypatch: pytest.MonkeyPatch) -> list[MetricCall]:
    from therapy.observability import telemetry

    calls: list[MetricCall] = []

    def capture(
        name: str, value: float, attrs: dict[str, str] | None = None
    ) -> None:
        calls.append((name, value, attrs or {}))

    monkeypatch.setattr(telemetry, "record_metric", capture)
    return calls


def _reinforce_node(model: UserModel, statement: str = "Skips lunch when busy.") -> int:
    node_id: int | None = None
    for session_id in ("s1", "s1", "s2"):
        node_id = model.upsert_node("pattern", statement, session_id=session_id)
    assert node_id is not None
    return node_id


def _reinforce_edge(model: UserModel, src: int, dst: int) -> int:
    edge_id: int | None = None
    for session_id in ("s1", "s1", "s2"):
        edge_id = model.upsert_edge(
            src,
            dst,
            "triggers",
            statement="Late coffee triggers poor sleep.",
            session_id=session_id,
        )
    assert edge_id is not None
    return edge_id


def _node(model: UserModel, node_id: int) -> GraphNode:
    """Return one required graph node for assertions."""
    node = model.get_node(node_id)
    assert node is not None
    return node


def _edge(model: UserModel, edge_id: int) -> GraphEdge:
    """Return one required graph edge for assertions."""
    edge = model.get_edge(edge_id)
    assert edge is not None
    return edge


def test_appendix_a_registries_and_legacy_compatibility(tmp_path: Path) -> None:
    assert NODE_TYPES == (
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
    assert EDGE_TYPES == (
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
    model = UserModel(tmp_path)
    node_id = model.upsert_node("routine", "Walks before breakfast.")
    assert node_id is not None
    assert _node(model, node_id)["type"] == "pattern"


def test_source_authority_and_direct_user_statement_path(tmp_path: Path) -> None:
    model = UserModel(tmp_path)
    with pytest.raises(ValueError, match="add_user_statement"):
        model.upsert_node("identity_fact", "Is a nurse.", source="user-stated")

    stated = model.add_user_statement("identity_fact", "Is a nurse.")
    observed = model.upsert_node("pattern", "Walks at dawn.")

    assert stated is not None
    assert observed is not None
    assert _node(model, stated)["status"] == "confirmed"
    assert _node(model, stated)["source"] == "user-stated"
    assert _node(model, observed)["status"] == "observation"


def test_node_and_edge_share_evidence_gated_lifecycle(tmp_path: Path) -> None:
    model = UserModel(tmp_path)
    pattern = _reinforce_node(model)
    trigger = _reinforce_node(model, "Drinks coffee late.")
    edge = _reinforce_edge(model, trigger, pattern)

    assert model.is_eligible(_node(model, pattern)) is True
    assert model.is_eligible(_edge(model, edge)) is True
    assert model.confirm_node(pattern) is False
    assert model.confirm_edge(edge) is False
    assert model.propose(pattern) is True
    assert model.propose_edge(edge) is True
    assert _node(model, pattern)["status"] == "proposed"
    assert _edge(model, edge)["status"] == "proposed"
    assert model.confirm_node(pattern, session_id="validation") is True
    assert model.confirm_edge(edge, session_id="validation") is True
    assert _node(model, pattern)["status"] == "confirmed"
    assert _edge(model, edge)["status"] == "confirmed"
    assert model.lifecycle_events("edge", edge)[-1]["event_type"] == "user_confirmed"


def test_rejection_is_durable_until_new_evidence(tmp_path: Path) -> None:
    model = UserModel(tmp_path)
    node_id = _reinforce_node(model)
    assert model.propose(node_id) is True
    assert model.reject_node(node_id) is True
    assert _node(model, node_id)["status"] == "rejected"
    assert model.propose(node_id) is False

    model.upsert_node("pattern", "Skips lunch when busy.", session_id="s3")
    assert model.propose(node_id) is True


def test_never_store_guards_quotes_edits_and_purges_existing_data(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path)
    session_id = store.create_session()
    store.end_session(session_id, "Discussion about divorce.")
    model = UserModel(tmp_path)
    sensitive = model.upsert_node(
        "pattern",
        "Feels overwhelmed by legal paperwork.",
        quotes=[{"text": "the divorce paperwork", "language": "en"}],
        session_id=session_id,
    )
    safe = model.upsert_node("strength", "Uses checklists.")
    alias_only = model.upsert_node("thread", "Preparing legal documents.")
    assert sensitive is not None
    assert safe is not None
    assert alias_only is not None
    edge = model.upsert_edge(
        sensitive,
        safe,
        "supports",
        statement="Paperwork routines support checklist use.",
    )
    assert edge is not None
    with sqlite3.connect(tmp_path / "therapy.db") as connection:
        connection.execute(
            """
            INSERT INTO node_aliases (node_id, alias, alias_key, language, created_at)
            VALUES (?, 'divorce documents', 'divorce documents', 'en', ?)
            """,
            (alias_only, datetime.now(UTC).isoformat()),
        )

    purge = model.add_boundary("never_store", "divorce")

    assert purge["graph_claims_removed"] == 3
    assert purge["summaries_removed"] == 1
    assert model.get_node(sensitive) is None
    assert model.get_node(alias_only) is None
    assert model.get_edge(edge) is None
    assert model.edit_node(safe, statement="Divorce checklist") is False
    assert model.upsert_node("pattern", "Divorce remains stressful.") is None
    assert model.add_observation("Divorce remains stressful.") is None
    assert MemoryStore(tmp_path).sessions()[0]["summary"] is None
    with sqlite3.connect(tmp_path / "therapy.db") as connection:
        for table in ("claim_evidence", "lifecycle_events", "pending_insights"):
            assert connection.execute(
                f"SELECT 1 FROM {table} WHERE claim_kind = 'edge' AND claim_id = ?",
                (edge,),
            ).fetchone() is None


def test_keyed_tombstones_hide_content_and_survive_endpoint_id_changes(
    tmp_path: Path,
) -> None:
    model = UserModel(tmp_path)
    src = model.upsert_node("pattern", "Late coffee.")
    dst = model.upsert_node("thread", "Poor sleep.")
    assert src is not None
    assert dst is not None
    edge = model.upsert_edge(
        src, dst, "triggers", statement="Late coffee triggers poor sleep."
    )
    assert edge is not None
    assert model.delete_edge(edge)

    with sqlite3.connect(tmp_path / "therapy.db") as connection:
        connection.execute("UPDATE nodes SET id = 101 WHERE id = ?", (src,))
        connection.execute("UPDATE nodes SET id = 102 WHERE id = ?", (dst,))
        tombstone = connection.execute("SELECT digest FROM tombstones").fetchone()[0]
    assert "coffee" not in tombstone
    assert len(tombstone) == 64
    assert (
        model.upsert_edge(
            101, 102, "triggers", statement="Coffee late at night harms sleep."
        )
        is None
    )


def test_edge_tombstone_identity_is_structurally_collision_safe(tmp_path: Path) -> None:
    model = UserModel(tmp_path)
    first_src = model.add_user_statement("pattern", "alpha:thread:beta")
    first_dst = model.add_user_statement("goal", "gamma")
    second_src = model.add_user_statement("pattern", "alpha")
    second_dst = model.add_user_statement("thread", "beta:goal:gamma")
    assert first_src is not None
    assert first_dst is not None
    assert second_src is not None
    assert second_dst is not None
    first_edge = model.upsert_edge(
        first_src, first_dst, "about", statement="same relationship wording"
    )
    assert first_edge is not None
    assert model.delete_edge(first_edge)

    second_edge = model.upsert_edge(
        second_src, second_dst, "about", statement="same relationship wording"
    )

    assert second_edge is not None


def test_owner_node_delete_removes_incident_claim_metadata(tmp_path: Path) -> None:
    model = UserModel(tmp_path)
    source = _reinforce_node(model, "Late coffee.")
    target = _reinforce_node(model, "Poor sleep.")
    edge = _reinforce_edge(model, source, target)
    assert model.propose(source)
    assert model.propose_edge(edge)
    assert InsightService(model).sync_proposals() == 2
    with sqlite3.connect(tmp_path / "therapy.db") as connection:
        connection.execute(
            """
            INSERT INTO semantic_embeddings (
                entity_kind, entity_id, model_name, model_revision,
                dimension, content_hash, vector, indexed_at
            ) VALUES ('node', ?, 'test', 'v1', 1, 'hash', ?, 'now')
            """,
            (str(source), b"\x00\x00\x00\x00"),
        )

    assert model.delete_node(source)

    with sqlite3.connect(tmp_path / "therapy.db") as connection:
        for table in ("claim_evidence", "lifecycle_events", "pending_insights"):
            for claim_kind, claim_id in (("node", source), ("edge", edge)):
                assert connection.execute(
                    f"SELECT 1 FROM {table} WHERE claim_kind = ? AND claim_id = ?",
                    (claim_kind, claim_id),
                ).fetchone() is None
        assert connection.execute(
            "SELECT 1 FROM semantic_embeddings "
            "WHERE entity_kind = 'node' AND entity_id = ?",
            (str(source),),
        ).fetchone() is None
        assert connection.execute(
            "SELECT 1 FROM node_aliases WHERE node_id = ?", (source,)
        ).fetchone() is None


def test_decay_flags_stale_nodes_and_edges_for_revalidation(tmp_path: Path) -> None:
    model = UserModel(tmp_path)
    thread = model.upsert_node("thread", "Preparing for an interview.")
    pattern = model.upsert_node("pattern", "Over-prepares for interviews.")
    assert thread is not None
    assert pattern is not None
    edge = model.upsert_edge(
        thread, pattern, "triggers", statement="Interview prep triggers over-preparing."
    )
    assert edge is not None

    changed = model.revalidate_stale(datetime.now(UTC) + timedelta(days=100))

    assert changed == 3
    assert _node(model, thread)["status"] == "needs_revalidation"
    assert _node(model, pattern)["status"] == "needs_revalidation"
    assert _edge(model, edge)["status"] == "needs_revalidation"


def test_session_deletion_sanitizes_quotes_or_removes_derived_claims(
    tmp_path: Path,
) -> None:
    model = UserModel(tmp_path)
    kept = model.upsert_node(
        "pattern",
        "Skips lunch.",
        quotes=[{"text": "I skipped lunch", "language": "en"}],
        session_id="keep",
    )
    removed = model.upsert_node(
        "thread",
        "Temporary incident.",
        quotes=[{"text": "temporary incident", "language": "en"}],
        session_id="remove",
    )
    assert kept is not None
    assert removed is not None

    result = model.delete_session_evidence("keep")
    evidence = model.evidence("node", kept)
    assert result["evidence_affected"] == 1
    assert evidence[0]["quote_text"] is None
    assert evidence[0]["source_state"] == "deleted"
    assert evidence[0]["source_marker"] is not None
    assert _node(model, kept)["status"] == "needs_revalidation"

    removed_result = model.delete_session_evidence("remove", remove_derived=True)
    assert removed_result["claims_removed"] == 1
    assert model.get_node(removed) is None


def test_context_allows_never_initiate_only_when_user_raises_topic(
    tmp_path: Path,
) -> None:
    model = UserModel(tmp_path)
    model.add_boundary("never_initiate", "my father")
    private = model.add_user_statement(
        "thread", "Conflict with my father.", never_initiate=True
    )
    goal = model.add_user_statement("goal", "Finish the portfolio.")
    assert private is not None
    assert goal is not None

    unsolicited = model.graph_walk("family conflict", allow_never_initiate=False)
    user_raised = model.graph_walk("I want to discuss my father")

    assert private not in {node["id"] for node in unsolicited["nodes"]}
    assert private in {node["id"] for node in user_raised["nodes"]}
    rendered = render_context(model.assemble_context("portfolio"))
    assert rendered is not None
    assert "Finish the portfolio" in rendered
    assert "Never initiate" in rendered


def test_partial_graph_migration_is_backed_up_and_preserves_evidence(
    tmp_path: Path,
) -> None:
    db = tmp_path / "therapy.db"
    with sqlite3.connect(db) as connection:
        connection.executescript(
            """
            CREATE TABLE nodes (
                id INTEGER PRIMARY KEY, type TEXT, statement TEXT, quotes TEXT,
                n_occurrences INTEGER, n_sessions INTEGER, sessions TEXT,
                status TEXT, source TEXT, never_initiate INTEGER,
                user_edited INTEGER, first_seen TEXT, last_seen TEXT
            );
            CREATE TABLE edges (
                id INTEGER PRIMARY KEY, src INTEGER, dst INTEGER, type TEXT,
                statement TEXT, quotes TEXT, n_occurrences INTEGER,
                n_sessions INTEGER, sessions TEXT, status TEXT, source TEXT,
                user_edited INTEGER, first_seen TEXT, last_seen TEXT
            );
            """
        )
        values = (1, 1, '["s1"]', "observation", "conversation", 0, 0, "2026-01-01", "2026-01-02")
        connection.execute(
            "INSERT INTO nodes VALUES (1,'trigger','Late coffee.','[]',?,?,?,?,?,?,?,?,?)",
            values,
        )
        connection.execute(
            "INSERT INTO nodes VALUES (2,'thread','Poor sleep.','[]',?,?,?,?,?,?,?,?,?)",
            values,
        )
        connection.execute(
            "INSERT INTO edges VALUES (1,1,2,'causes','','[]',1,1,'[\"s1\"]',"
            "'pattern','conversation',0,'2026-01-01','2026-01-02')"
        )

    model = UserModel(tmp_path)

    assert model.schema_version() == 3
    assert model.schema_backup_path is not None
    assert model.schema_backup_path.is_file()
    assert {node["type"] for node in model.nodes()} == {"pattern", "thread"}
    migrated_edge = model.edges()[0]
    assert migrated_edge["type"] == "triggers"
    assert migrated_edge["statement"]
    assert migrated_edge["status"] == "proposed"
    assert len(model.evidence("edge", migrated_edge["id"])) == 1


def test_legacy_migration_enforces_boundaries_and_converts_tombstones(
    tmp_path: Path,
) -> None:
    db = tmp_path / "therapy.db"
    with sqlite3.connect(db) as connection:
        connection.executescript(
            """
            CREATE TABLE nodes (
                id INTEGER PRIMARY KEY, type TEXT, statement TEXT, quotes TEXT,
                n_occurrences INTEGER, n_sessions INTEGER, sessions TEXT,
                status TEXT, source TEXT, never_initiate INTEGER,
                user_edited INTEGER, first_seen TEXT, last_seen TEXT
            );
            CREATE TABLE edges (
                id INTEGER PRIMARY KEY, src INTEGER, dst INTEGER, type TEXT,
                statement TEXT, quotes TEXT, n_occurrences INTEGER,
                n_sessions INTEGER, sessions TEXT, status TEXT, source TEXT,
                user_edited INTEGER, first_seen TEXT, last_seen TEXT
            );
            CREATE TABLE observation_inbox (
                id INTEGER PRIMARY KEY, session_id TEXT, text TEXT,
                language TEXT, created_at TEXT, processed INTEGER
            );
            CREATE TABLE boundaries (
                id INTEGER PRIMARY KEY, kind TEXT, value TEXT, created_at TEXT,
                UNIQUE(kind, value)
            );
            CREATE TABLE tombstones (signature TEXT PRIMARY KEY, created_at TEXT);
            """
        )
        node_values = (1, 1, '["s1"]', "observation", "conversation", 0, 0, "a", "b")
        connection.execute(
            "INSERT INTO nodes VALUES (1,'trigger','Late coffee.','[]',?,?,?,?,?,?,?,?,?)",
            node_values,
        )
        connection.execute(
            "INSERT INTO nodes VALUES (2,'thread','Poor sleep.','[]',?,?,?,?,?,?,?,?,?)",
            node_values,
        )
        connection.execute(
            "INSERT INTO nodes VALUES (3,'pattern','Legal paperwork.',"
            "'[{\"text\":\"divorce papers\",\"language\":\"en\"}]',"
            "?,?,?,?,?,?,?,?,?)",
            node_values,
        )
        connection.execute(
            "INSERT INTO edges VALUES (1,1,3,'supports','Routines support paperwork.',"
            "'[]',1,1,'[\"s1\"]','observation','conversation',0,'a','b')"
        )
        connection.execute(
            "INSERT INTO observation_inbox VALUES "
            "(1,'s1','divorce follow-up','en','a',0),"
            "(2,'s1','safe follow-up','en','a',0)"
        )
        connection.execute(
            "INSERT INTO boundaries VALUES (1,'never_store','divorce','a')"
        )
        connection.execute(
            "INSERT INTO tombstones VALUES ('node:trigger:Old secret.','a')"
        )
        connection.execute(
            "INSERT INTO tombstones VALUES ('edge:causes:1->2','a')"
        )

    model = UserModel(tmp_path)

    assert {node["statement"] for node in model.nodes()} == {
        "Late coffee.",
        "Poor sleep.",
    }
    assert model.edges() == []
    assert [row["text"] for row in model.pending_observations("s1")] == [
        "safe follow-up"
    ]
    assert model.upsert_node("pattern", "Old secret.") is None
    assert (
        model.upsert_edge(
            1, 2, "triggers", statement="Coffee timing interferes with rest."
        )
        is None
    )
    with sqlite3.connect(db) as connection:
        tombstones = [row[0] for row in connection.execute("SELECT digest FROM tombstones")]
    assert len(tombstones) == 2
    assert all(len(digest) == 64 for digest in tombstones)
    assert "coffee" not in "".join(tombstones)


def test_flat_fact_table_migrates_then_is_retired(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    del store
    with sqlite3.connect(tmp_path / "therapy.db") as connection:
        connection.execute(
            """
            CREATE TABLE facts (
                id INTEGER PRIMARY KEY, statement TEXT, kind TEXT,
                first_seen TEXT, last_seen TEXT, n_occurrences INTEGER
            )
            """
        )
        connection.execute(
            "INSERT INTO facts VALUES (1,'Knows Ana.','relationship','a','b',2)"
        )
    model = UserModel(tmp_path)
    assert model.nodes()[0]["type"] == "person"
    assert model.nodes()[0]["n_occurrences"] == 2
    with sqlite3.connect(tmp_path / "therapy.db") as connection:
        assert connection.execute(
            "SELECT 1 FROM sqlite_master WHERE name = 'facts'"
        ).fetchone() is None
    with pytest.raises(RuntimeError, match="retired"):
        MemoryStore(tmp_path).upsert_fact("no")


def test_export_contains_full_provenance_and_no_plaintext_tombstone(
    tmp_path: Path,
) -> None:
    model = UserModel(tmp_path)
    node_id = model.upsert_node("strength", "Writes clear plans.")
    assert node_id is not None
    model.delete_node(node_id)

    snapshot = model.export_all()
    encoded = json.dumps(snapshot)

    assert snapshot["schema_version"] == 3
    assert "claim_evidence" in snapshot
    assert "distillation_runs" in snapshot
    assert "Writes clear plans" not in encoded
    tombstones = snapshot["tombstones"]
    assert isinstance(tombstones, list)
    tombstone_value = cast(list[object], tombstones)[0]
    tombstone = require_json_object(tombstone_value, "snapshot.tombstones[0]")
    digest = tombstone["digest"]
    assert isinstance(digest, str)
    assert len(digest) == 64


def test_graph_mutation_metrics_cover_success_rejection_and_cardinality(
    tmp_path: Path, metric_calls: list[MetricCall]
) -> None:
    model = UserModel(tmp_path)
    metric_calls.clear()

    source = model.add_user_statement("thread", "A private scheduling thread.")
    target = model.add_user_statement("pattern", "Uses a written checklist.")
    assert source is not None
    assert target is not None
    edge = model.add_user_edge(
        source,
        target,
        "supports",
        "The scheduling thread supports checklist use.",
    )
    assert edge is not None
    assert model.edit_node(target, never_initiate=True)
    assert model.edit_edge(edge, statement="Planning supports checklist use.")
    assert model.revalidate_stale(datetime.now(UTC) + timedelta(days=100)) == 3
    _, distilled_nodes, distilled_edges = model.apply_distillation(
        session_id="metric-session",
        extractor_version="metric-v1",
        candidates=[
            {
                "kind": "node",
                "type": "strength",
                "statement": "Keeps plans concise.",
            }
        ],
        inbox_ids=[],
    )
    assert len(distilled_nodes) == 1
    assert distilled_edges == []
    assert model.delete_edge(edge)
    assert model.delete_node(source)
    assert model.delete_node(999_999) is False

    mutations = [
        attrs
        for name, _, attrs in metric_calls
        if name == "therapy_graph_mutations_total"
    ]
    assert {
        "add_node",
        "add_edge",
        "edit",
        "delete",
        "revalidate",
        "distill_apply",
    } <= {attrs["operation"] for attrs in mutations}
    assert all(set(attrs) == {"operation", "outcome"} for attrs in mutations)
    assert all(
        attrs["operation"]
        in {
            "add_node",
            "add_edge",
            "edit",
            "delete",
            "purge",
            "revalidate",
            "distill_apply",
        }
        and attrs["outcome"] in {"success", "error", "timeout", "rejected"}
        for attrs in mutations
    )
    assert ("therapy_graph_mutations_total", 1, {"operation": "delete", "outcome": "rejected"}) in metric_calls
    encoded = json.dumps(metric_calls)
    assert "private scheduling" not in encoded.casefold()
    assert "metric-session" not in encoded


def test_never_store_metrics_count_only_bounded_storage_classes(
    tmp_path: Path, metric_calls: list[MetricCall]
) -> None:
    model = UserModel(tmp_path)
    sensitive = model.add_user_statement("thread", "A private recurring matter.")
    safe = model.add_user_statement("strength", "Uses clear plans.")
    assert sensitive is not None
    assert safe is not None
    edge = model.add_user_edge(
        sensitive, safe, "about", "The private matter affects planning."
    )
    assert edge is not None
    assert model.add_observation("A private follow-up.") is not None
    with sqlite3.connect(tmp_path / "therapy.db") as connection:
        connection.execute(
            """
            INSERT INTO research_documents (
                source_title, source_ref, filename, media_type, format,
                content_hash, original_size, artifact_path, extracted_markdown,
                status, ingested_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?)
            """,
            (
                "Private notes",
                "owner-upload",
                "notes.txt",
                "text/plain",
                "text",
                "test-hash",
                10,
                "A private research note.",
                "indexed",
                datetime.now(UTC).isoformat(),
                datetime.now(UTC).isoformat(),
            ),
        )
    metric_calls.clear()

    purge = model.add_boundary("never_store", "private")
    assert purge["graph_claims_removed"] == 2
    assert purge["inbox_rows_removed"] == 1
    assert purge["research_documents_removed"] == 1
    assert model.add_observation("Another private follow-up.") is None
    assert model.add_user_statement("thread", "Another private matter.") is None
    assert model.edit_node(safe, statement="A private plan.") is False

    suppressions = [
        (value, attrs)
        for name, value, attrs in metric_calls
        if name == "therapy_never_store_suppressions_total"
    ]
    assert all(set(attrs) == {"table_class"} for _, attrs in suppressions)
    assert all(
        attrs["table_class"] in {"node", "edge", "file", "turn"}
        for _, attrs in suppressions
    )
    totals = {
        table_class: sum(
            value
            for value, attrs in suppressions
            if attrs["table_class"] == table_class
        )
        for table_class in {"node", "edge", "file", "turn"}
    }
    assert totals["node"] == 3
    assert totals["edge"] == 1
    assert totals["file"] == 1
    assert totals["turn"] == 2
    assert (
        "therapy_graph_mutations_total",
        1,
        {"operation": "purge", "outcome": "success"},
    ) in metric_calls
    assert "private" not in json.dumps(metric_calls).casefold()


def test_graph_mutation_unexpected_exception_records_error_without_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metric_calls: list[MetricCall],
) -> None:
    model = UserModel(tmp_path)
    metric_calls.clear()

    def fail(*_args: object, **_kwargs: object) -> int | None:
        raise RuntimeError("private statement and node 42")

    monkeypatch.setattr(model, "_upsert_node", fail)
    with pytest.raises(RuntimeError, match="private statement"):
        model.add_user_statement("thread", "Content must stay restricted.")

    assert metric_calls == [
        (
            "therapy_graph_mutations_total",
            1,
            {"operation": "add_node", "outcome": "error"},
        )
    ]
    assert "private statement" not in json.dumps(metric_calls).casefold()


def test_graph_read_paths_do_not_emit_mutation_metrics(
    tmp_path: Path, metric_calls: list[MetricCall]
) -> None:
    model = UserModel(tmp_path)
    assert model.add_user_statement("strength", "Uses clear plans.") is not None
    metric_calls.clear()

    model.nodes()
    model.edges()
    model.graph_walk("clear plans")

    assert not any(
        name == "therapy_graph_mutations_total" for name, _, _ in metric_calls
    )


def test_inferred_claims_upgrade_in_place_when_owner_confirms_them(
    tmp_path: Path,
) -> None:
    model = UserModel(tmp_path)
    source = model.add_user_statement("thread", "Planning the coming week.")
    target = model.upsert_node("strategy", "Uses a written checklist.")
    assert source is not None
    assert target is not None
    edge = model.upsert_edge(
        source,
        target,
        "supports",
        statement="Weekly planning supports checklist use.",
    )
    assert edge is not None

    confirmed_target = model.add_user_statement("strategy", "Uses a written checklist.")
    confirmed_edge = model.add_user_edge(
        source,
        target,
        "supports",
        "Weekly planning supports checklist use.",
    )

    assert confirmed_target == target
    assert confirmed_edge == edge
    assert (_node(model, target)["status"], _node(model, target)["source"]) == (
        "confirmed",
        "user-stated",
    )
    assert (_edge(model, edge)["status"], _edge(model, edge)["source"]) == (
        "confirmed",
        "user-stated",
    )
    assert [
        (event["from_status"], event["to_status"], event["event_type"])
        for event in model.lifecycle_events()
        if (event["claim_kind"], event["claim_id"])
        in {("node", target), ("edge", edge)}
    ] == [
        (None, "observation", "observed"),
        (None, "observation", "observed"),
        ("observation", "confirmed", "direct_user_statement"),
        ("observation", "confirmed", "direct_user_statement"),
    ]


@pytest.mark.parametrize(
    ("node_type", "days"),
    [("thread", 14), ("pattern", 56), ("strategy", 90), ("strength", None)],
)
def test_node_revalidation_schedule_is_type_specific(
    tmp_path: Path, node_type: str, days: int | None
) -> None:
    model = UserModel(tmp_path)
    node_id = model.upsert_node(node_type, f"Schedule for {node_type}.")
    assert node_id is not None
    node = _node(model, node_id)

    if days is None:
        assert node["revalidation_due_at"] is None
        return
    assert node["revalidation_due_at"] is not None
    assert datetime.fromisoformat(node["revalidation_due_at"]) - datetime.fromisoformat(
        node["last_seen"]
    ) == timedelta(days=days)


def test_distillation_filters_private_candidates_and_replays_success(
    tmp_path: Path, metric_calls: list[MetricCall]
) -> None:
    model = UserModel(tmp_path)
    model.add_boundary("never_store", "restricted")
    tombstoned = model.add_user_statement("strategy", "Retire this draft.")
    assert tombstoned is not None
    assert model.delete_node(tombstoned)
    metric_calls.clear()

    result = model.apply_distillation(
        session_id="distill-filter-session",
        extractor_version="filter-v1",
        candidates=[
            {
                "kind": "node",
                "type": "thread",
                "statement": "Plans the coming week.",
                "aliases": [
                    {"text": ""},
                    {"text": "restricted planning alias"},
                    {"text": "Weekly planning", "language": "en"},
                ],
            },
            {
                "kind": "node",
                "type": "strategy",
                "statement": "Uses concise checklists.",
            },
            {
                "kind": "node",
                "type": "thread",
                "statement": "A restricted private topic.",
            },
            {
                "kind": "node",
                "type": "strategy",
                "statement": "Retire this draft.",
            },
            {
                "kind": "edge",
                "type": "supports",
                "src": "Plans the coming week.",
                "dst": "Uses concise checklists.",
                "statement": "Weekly planning supports concise checklists.",
            },
            {
                "kind": "edge",
                "type": "conflicts_with",
                "src": "Plans the coming week.",
                "dst": "Uses concise checklists.",
                "statement": "A restricted relationship.",
            },
        ],
        inbox_ids=[],
    )
    replay = model.apply_distillation(
        session_id="distill-filter-session",
        extractor_version="filter-v1",
        candidates=[],
        inbox_ids=[],
    )

    assert replay == result
    assert len(result[1]) == 2
    assert len(result[2]) == 1
    assert {node["statement"] for node in model.nodes()} == {
        "Plans the coming week.",
        "Uses concise checklists.",
    }
    assert [edge["statement"] for edge in model.edges()] == [
        "Weekly planning supports concise checklists."
    ]
    with sqlite3.connect(tmp_path / "therapy.db") as connection:
        aliases = {
            str(row[0])
            for row in connection.execute("SELECT alias FROM node_aliases").fetchall()
        }
    assert "Weekly planning" in aliases
    assert "restricted planning alias" not in aliases
    assert "Retire this draft." not in aliases
    assert all(
        set(attrs) <= {"operation", "outcome", "table_class"}
        for _, _, attrs in metric_calls
    )
    assert "restricted" not in json.dumps(metric_calls).casefold()
    assert "distill-filter-session" not in json.dumps(metric_calls)


def test_distillation_rejects_invalid_scope_and_self_edges_atomically(
    tmp_path: Path, metric_calls: list[MetricCall]
) -> None:
    model = UserModel(tmp_path)
    inbox_id = model.add_observation("Owner-scoped observation.", session_id="owner")
    endpoint = model.add_user_statement("thread", "One endpoint.")
    assert inbox_id is not None
    assert endpoint is not None
    metric_calls.clear()

    with pytest.raises(ValueError, match="belong to the session"):
        model.apply_distillation(
            session_id="foreign",
            extractor_version="scope-v1",
            candidates=[
                {
                    "kind": "node",
                    "type": "strength",
                    "statement": "This mutation must roll back.",
                }
            ],
            inbox_ids=[inbox_id],
        )
    with pytest.raises(ValueError, match="endpoints must be different"):
        model.apply_distillation(
            session_id="owner",
            extractor_version="self-edge-v1",
            candidates=[
                {
                    "kind": "edge",
                    "type": "supports",
                    "src": "One endpoint.",
                    "dst": "One endpoint.",
                    "statement": "A self relationship.",
                }
            ],
            inbox_ids=[],
        )

    assert model.get_node(endpoint) is not None
    assert all(
        node["statement"] != "This mutation must roll back." for node in model.nodes()
    )
    assert [row["id"] for row in model.pending_observations("owner")] == [inbox_id]
    assert model.edges() == []
    failed = [
        run
        for run in cast(list[dict[str, object]], model.export_all()["distillation_runs"])
        if run["state"] == "failed"
    ]
    assert len(failed) == 2
    assert all(str(run["error"]).startswith("ValueError:") for run in failed)
    assert [
        attrs["outcome"]
        for name, _, attrs in metric_calls
        if name == "therapy_graph_mutations_total"
        and attrs["operation"] == "distill_apply"
    ] == ["rejected", "rejected"]


def test_distillation_run_retry_and_failure_state_are_durable(tmp_path: Path) -> None:
    model = UserModel(tmp_path)
    run_id = model.start_distillation_run("retry-session", "retry-v1")
    assert model.start_distillation_run("retry-session", "retry-v1") == run_id

    model.fail_distillation_run(run_id, RuntimeError("x" * 1_200))
    failed = model.distillation_run(run_id)
    assert failed is not None
    assert failed["state"] == "failed"
    assert len(str(failed["error"])) == 1_000

    assert model.start_distillation_run("retry-session", "retry-v1") == run_id
    retried = model.distillation_run(run_id)
    assert retried is not None
    assert retried["state"] == "running"
    assert retried["attempt_count"] == 2
    assert retried["error"] is None

    applied_run, node_ids, edge_ids = model.apply_distillation(
        session_id="retry-session",
        extractor_version="retry-v1",
        candidates=[],
        inbox_ids=[],
    )
    assert (applied_run, node_ids, edge_ids) == (run_id, [], [])
    model.fail_distillation_run(run_id, RuntimeError("must not replace success"))
    succeeded = model.distillation_run(run_id)
    assert succeeded is not None
    assert succeeded["state"] == "succeeded"
    assert succeeded["result"] == {"node_ids": [], "edge_ids": []}
    assert model.distillation_run("missing-run") is None
    with pytest.raises(ValueError, match="required"):
        model.start_distillation_run("", "retry-v1")


def test_mutation_failures_emit_bounded_error_and_timeout_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metric_calls: list[MetricCall],
) -> None:
    model = UserModel(tmp_path)
    source = model.add_user_statement("thread", "Source node.")
    target = model.add_user_statement("strategy", "Target node.")
    assert source is not None
    assert target is not None
    metric_calls.clear()

    with pytest.raises(ValueError, match="session_id"):
        model.apply_distillation(
            session_id="", extractor_version="v1", candidates=[], inbox_ids=[]
        )

    def fail_distillation(**_kwargs: object) -> tuple[str, list[int], list[int]]:
        raise TimeoutError("private distillation payload")

    monkeypatch.setattr(model, "_apply_distillation", fail_distillation)
    with pytest.raises(TimeoutError, match="private distillation"):
        model.apply_distillation(
            session_id="timeout-session",
            extractor_version="v1",
            candidates=[],
            inbox_ids=[],
        )

    def fail_revalidation(_now: datetime | None = None) -> int:
        raise TimeoutError("private revalidation payload")

    monkeypatch.setattr(model, "_revalidate_stale", fail_revalidation)
    with pytest.raises(TimeoutError, match="private revalidation"):
        model.revalidate_stale()

    def fail_node(*_args: object, **_kwargs: object) -> int | None:
        raise TimeoutError("private node payload")

    monkeypatch.setattr(model, "_upsert_node", fail_node)
    with pytest.raises(TimeoutError, match="private node"):
        model.upsert_node("thread", "Do not emit this node.")

    busy = sqlite3.OperationalError("private edge payload")
    busy.sqlite_errorcode = sqlite3.SQLITE_BUSY

    def fail_edge(*_args: object, **_kwargs: object) -> int | None:
        raise busy

    monkeypatch.setattr(model, "_upsert_edge", fail_edge)
    with pytest.raises(sqlite3.OperationalError, match="private edge"):
        model.upsert_edge(
            source, target, "supports", statement="Do not emit this edge."
        )

    mutations = [
        (attrs["operation"], attrs["outcome"])
        for name, _, attrs in metric_calls
        if name == "therapy_graph_mutations_total"
    ]
    assert mutations == [
        ("distill_apply", "rejected"),
        ("distill_apply", "timeout"),
        ("revalidate", "timeout"),
        ("add_node", "timeout"),
        ("add_edge", "timeout"),
    ]
    assert all(
        set(attrs) == {"operation", "outcome"}
        for name, _, attrs in metric_calls
        if name == "therapy_graph_mutations_total"
    )
    assert "private" not in json.dumps(metric_calls).casefold()


def test_processed_observation_audit_rows_are_marked_and_pruned(
    tmp_path: Path,
) -> None:
    model = UserModel(tmp_path)
    first = model.add_observation("First bounded observation.", session_id="s1")
    second = model.add_observation("Second bounded observation.", session_id="s1")
    assert first is not None
    assert second is not None

    model.mark_observations_processed([], run_id="ignored")
    model.mark_observations_processed([first], run_id="run-1")
    pending = model.pending_observations("s1")
    assert [row["id"] for row in pending] == [second]
    audited = model.pending_observations("s1", include_processed=True)
    assert audited[0]["distillation_run_id"] == "run-1"

    with sqlite3.connect(tmp_path / "therapy.db") as connection:
        connection.execute(
            "UPDATE observation_inbox SET processed_at = ? WHERE id = ?",
            ("2000-01-01T00:00:00+00:00", first),
        )
        connection.commit()
    assert model.prune_processed_observations(older_than_days=1) == 1
    assert [row["id"] for row in model.pending_observations(include_processed=True)] == [
        second
    ]
    with pytest.raises(ValueError, match=">= 1"):
        model.prune_processed_observations(older_than_days=0)
