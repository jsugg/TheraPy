"""Property-graph schema, lifecycle, privacy, and provenance integration tests."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from therapy.knowledge.insight import InsightService
from therapy.knowledge.user_model import (
    EDGE_TYPES,
    NODE_TYPES,
    UserModel,
    render_context,
)
from therapy.memory import MemoryStore


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
    assert model.get_node(node_id)["type"] == "pattern"


def test_source_authority_and_direct_user_statement_path(tmp_path: Path) -> None:
    model = UserModel(tmp_path)
    with pytest.raises(ValueError, match="add_user_statement"):
        model.upsert_node("identity_fact", "Is a nurse.", source="user-stated")

    stated = model.add_user_statement("identity_fact", "Is a nurse.")
    observed = model.upsert_node("pattern", "Walks at dawn.")

    assert stated is not None
    assert observed is not None
    assert model.get_node(stated)["status"] == "confirmed"
    assert model.get_node(stated)["source"] == "user-stated"
    assert model.get_node(observed)["status"] == "observation"


def test_node_and_edge_share_evidence_gated_lifecycle(tmp_path: Path) -> None:
    model = UserModel(tmp_path)
    pattern = _reinforce_node(model)
    trigger = _reinforce_node(model, "Drinks coffee late.")
    edge = _reinforce_edge(model, trigger, pattern)

    assert model.is_eligible(model.get_node(pattern)) is True
    assert model.is_eligible(model.get_edge(edge)) is True
    assert model.confirm_node(pattern) is False
    assert model.confirm_edge(edge) is False
    assert model.propose(pattern) is True
    assert model.propose_edge(edge) is True
    assert model.get_node(pattern)["status"] == "proposed"
    assert model.get_edge(edge)["status"] == "proposed"
    assert model.confirm_node(pattern, session_id="validation") is True
    assert model.confirm_edge(edge, session_id="validation") is True
    assert model.get_node(pattern)["status"] == "confirmed"
    assert model.get_edge(edge)["status"] == "confirmed"
    assert model.lifecycle_events("edge", edge)[-1]["event_type"] == "user_confirmed"


def test_rejection_is_durable_until_new_evidence(tmp_path: Path) -> None:
    model = UserModel(tmp_path)
    node_id = _reinforce_node(model)
    assert model.propose(node_id) is True
    assert model.reject_node(node_id) is True
    assert model.get_node(node_id)["status"] == "rejected"
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
    assert model.get_node(thread)["status"] == "needs_revalidation"
    assert model.get_node(pattern)["status"] == "needs_revalidation"
    assert model.get_edge(edge)["status"] == "needs_revalidation"


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
    assert model.get_node(kept)["status"] == "needs_revalidation"

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
    assert len(snapshot["tombstones"][0]["digest"]) == 64
