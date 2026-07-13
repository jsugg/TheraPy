"""Tests for the property-graph user model (W1) and graph walk (W3)."""

from __future__ import annotations

from pathlib import Path

from therapy.knowledge.user_model import UserModel, render_context
from therapy.memory import MemoryStore


def test_user_stated_starts_confirmed_conversation_starts_observation(
    tmp_path: Path,
) -> None:
    model = UserModel(tmp_path)
    stated = model.upsert_node("identity", "Is a nurse.", source="user-stated")
    observed = model.upsert_node("routine", "Walks at dawn.")
    assert model.get_node(stated)["status"] == "confirmed"
    assert model.get_node(observed)["status"] == "observation"


def test_graduation_floor_then_explicit_confirmation(tmp_path: Path) -> None:
    model = UserModel(tmp_path)
    nid = model.upsert_node("routine", "Skips lunch when busy.", session_id="s1")
    # A second occurrence in the SAME session does not clear the floor.
    model.upsert_node("routine", "Skips lunch when busy.", session_id="s1")
    assert model.propose(nid) is False
    assert model.get_node(nid)["status"] == "observation"
    # A third occurrence in a SECOND session clears >=3 across >=2 sessions.
    model.upsert_node("routine", "Skips lunch when busy.", session_id="s2")
    node = model.get_node(nid)
    assert node["n_occurrences"] == 3
    assert node["n_sessions"] == 2
    assert model.is_eligible(node) is True
    assert model.propose(nid) is True
    assert model.get_node(nid)["status"] == "pattern"
    # The floor alone never mints `confirmed`; only explicit validation does.
    assert model.confirm_node(nid) is True
    assert model.get_node(nid)["status"] == "confirmed"


def test_confirmed_requires_pattern_first_via_the_engine(tmp_path: Path) -> None:
    model = UserModel(tmp_path)
    nid = model.upsert_node("trait", "Tends to over-prepare.", session_id="s1")
    # Not eligible yet: propose is refused, so it cannot become a pattern.
    assert model.propose(nid) is False
    assert model.get_node(nid)["status"] == "observation"


def test_reinforce_merges_quotes_and_counts_sessions(tmp_path: Path) -> None:
    model = UserModel(tmp_path)
    nid = model.upsert_node(
        "preference",
        "Prefers written over spoken instructions.",
        quotes=[{"text": "please write it down", "lang": "en"}],
        session_id="s1",
    )
    model.upsert_node(
        "preference",
        "Prefers written over spoken instructions.",
        quotes=[{"text": "escríbelo", "lang": "es"}],
        session_id="s2",
    )
    node = model.get_node(nid)
    assert node["n_occurrences"] == 2
    assert node["n_sessions"] == 2
    assert len(node["quotes"]) == 2


def test_never_store_blocks_every_write_path(tmp_path: Path) -> None:
    model = UserModel(tmp_path)
    model.add_boundary("never_store", "divorce")
    assert model.upsert_node("note", "Went through a divorce.") is None
    assert model.add_observation("The divorce was hard.") is None
    assert model.upsert_node("note", "Adopted a dog.") is not None
    assert model.pending_observations() == []


def test_delete_tombstones_against_relearning(tmp_path: Path) -> None:
    model = UserModel(tmp_path)
    nid = model.upsert_node("note", "Owns a boat.")
    assert model.delete_node(nid) is True
    # Distillation must not be able to re-learn a deleted claim.
    assert model.upsert_node("note", "Owns a boat.") is None
    assert model.nodes() == []


def test_graph_walk_includes_confirmed_edges_and_omits_never_initiate(
    tmp_path: Path,
) -> None:
    model = UserModel(tmp_path)
    sleep = model.upsert_node("thread", "Struggles with sleep lately.")
    caffeine = model.upsert_node("trigger", "Drinks coffee late in the day.")
    model.confirm_node(sleep)
    model.confirm_node(caffeine)
    edge = model.upsert_edge(caffeine, sleep, "causes", statement="caffeine hurts sleep")
    model.confirm_edge(edge)
    secret = model.upsert_node(
        "thread", "Sleep problems around a private matter.", never_initiate=True
    )
    assert secret is not None

    walk = model.graph_walk("sleep and coffee")
    ids = {int(n["id"]) for n in walk["nodes"]}
    assert sleep in ids
    assert caffeine in ids
    assert secret not in ids  # never_initiate node omitted from the walk
    assert any(int(e["id"]) == edge for e in walk["edges"])


def test_assemble_and_render_context_carries_never_initiate_list(
    tmp_path: Path,
) -> None:
    model = UserModel(tmp_path)
    model.upsert_node("identity", "Lives alone.", source="user-stated")
    model.add_boundary("never_initiate", "my father")
    rendered = render_context(model.assemble_context("living situation"))
    assert rendered is not None
    assert "Lives alone." in rendered
    assert "my father" in rendered


def test_migration_imports_v1_facts_as_observation_nodes(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path)
    store.upsert_fact("Has a dog named Bruno.")
    store.upsert_fact("Has a dog named Bruno.")  # reinforced -> n_occurrences 2
    store.upsert_fact("Knows Ana.", kind="relationship")

    model = UserModel(tmp_path)  # migration runs on init
    nodes = {n["statement"]: n for n in model.nodes()}
    assert len(nodes) == 2  # zero loss: two distinct facts -> two nodes
    assert nodes["Has a dog named Bruno."]["status"] == "observation"
    assert nodes["Has a dog named Bruno."]["n_occurrences"] == 2
    assert nodes["Knows Ana."]["type"] == "relationship"

    # Idempotent: re-running migration does not duplicate or re-bump.
    assert model.migrate_v1_facts() == 0
    assert len(model.nodes()) == 2
