"""Phase-4 acceptance (SPEC §9): the longitudinal self-knowledge loop.

Deterministic and offline — no live server, no LLM, no network. Everything runs
against an *isolated* `THERAPY_DATA_DIR` (a fresh temp dir), so the acceptance
never touches the real `/data` (the Hardening 7-9 lesson). A stubbed distillation
extractor stands in for the cloud LLM so the run is reproducible.

Proves, on seeded multi-session data:

  1. A claim graduates observation -> pattern -> confirmed **only** after
     explicit user validation; the mechanical floor alone never mints
     `confirmed` (SPEC §3, R1).
  2. Graph-walk context assembly includes confirmed edges and omits
     `never_initiate` nodes (W3).
  3. One proactive channel respects quiet hours — suppressed inside the quiet
     window, cleared to fire outside it — and defers to `never_initiate` (W5).
  4. A research-KB query grounds a technique choice (silent grounding) and
     returns a source citation on demand (W6).
  5. Former v1 flat `facts` migrate into the property graph as `observation`
     nodes with zero loss (W1 migration).

Run (framework-free venv is enough):

    .venv/bin/python scripts/phase4_acceptance.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, cast

RESULTS: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    """Record a pass/fail line."""
    mark = "✓" if ok else "FAIL"
    suffix = f" — {detail}" if detail else ""
    RESULTS.append(f"[{label}] {'' if ok else 'FAIL: '}{mark}{suffix}")


async def _seed_and_graduate(model, distill) -> int:
    """Seed three sessions of a recurring claim and run distillation each time.

    Returns the node id of the recurring claim after graduation.
    """
    statement = "Skips lunch when the day gets busy."

    async def stub_extract(transcript: str, observations: list[str]) -> list[dict]:
        # Stands in for the cloud LLM: promotes the recurring observation into a
        # routine node (and never invents `confirmed`).
        return [{"kind": "node", "type": "routine", "statement": statement}]

    for session in ("s1", "s2", "s3"):
        model.add_observation(
            f"I keep skipping lunch, session {session}.", session_id=session
        )
        await distill.distill_session(model, [], session, extractor=stub_extract)

    matches = [n for n in model.nodes() if n["statement"] == statement]
    return int(matches[0]["id"]) if matches else -1


async def main() -> int:
    with tempfile.TemporaryDirectory(prefix="therapy-phase4-") as tmp:
        os.environ["THERAPY_DATA_DIR"] = tmp

        from datetime import datetime

        from therapy.dialogue import proactive
        from therapy.dialogue.policy import crisis_contacts
        from therapy.knowledge import distill
        from therapy.knowledge.research import ResearchKB
        from therapy.knowledge.user_model import UserModel, render_context
        from therapy.memory import MemoryStore

        assert Path(tmp).resolve() != Path("/data"), "must never touch real /data"

        # --- 5. v1 facts migrate as observation nodes (seed BEFORE UserModel) ---
        store = MemoryStore()
        store.upsert_fact("Has a dog named Bruno.")
        store.upsert_fact("Has a dog named Bruno.")  # reinforced
        store.upsert_fact("Knows Ana.", kind="relationship")
        n_facts = len(store.facts())

        model = UserModel()  # migration runs on init
        migrated = [n for n in model.nodes() if n["source"] == "conversation"]
        migrated_stmts = {n["statement"] for n in migrated}
        check(
            "migration",
            len(migrated) == n_facts
            and all(n["status"] == "observation" for n in migrated)
            and "Has a dog named Bruno." in migrated_stmts,
            f"{n_facts} v1 facts -> {len(migrated)} observation nodes (zero loss)",
        )

        # --- 1. graduation only via explicit validation ---
        node_id = await _seed_and_graduate(model, distill)
        node = model.get_node(node_id)
        floor_ok = node is not None and node["status"] == "pattern"
        # The floor alone must NOT have produced `confirmed`.
        no_premature_confirm = node is not None and node["status"] != "confirmed"
        model.confirm_node(node_id)  # explicit user validation
        confirmed_node = model.get_node(node_id)
        confirmed_ok = confirmed_node is not None and (
            confirmed_node["status"] == "confirmed"
        )
        check(
            "graduation",
            floor_ok and no_premature_confirm and confirmed_ok,
            "observation -> pattern (floor) -> confirmed (only after validation)",
        )

        # --- 2. graph walk includes confirmed edges, omits never_initiate ---
        caffeine = model.upsert_node("trigger", "Drinks coffee late in the day.")
        sleep = model.upsert_node("thread", "Sleep has been poor lately.")
        assert caffeine is not None and sleep is not None
        model.confirm_node(caffeine)
        model.confirm_node(sleep)
        edge = model.upsert_edge(
            caffeine, sleep, "causes", statement="late caffeine worsens sleep"
        )
        assert edge is not None
        model.confirm_edge(edge)
        model.add_boundary("never_initiate", "my brother")
        secret = model.upsert_node(
            "thread", "A private worry about sleep and my brother.",
            never_initiate=True,
        )
        walk = model.graph_walk("coffee and sleep")
        walk_nodes = cast("list[dict[str, Any]]", walk["nodes"])
        walk_edges = cast("list[dict[str, Any]]", walk["edges"])
        walk_ids = {int(n["id"]) for n in walk_nodes}
        edge_present = any(int(e["id"]) == edge for e in walk_edges)
        secret_omitted = secret not in walk_ids
        rendered = render_context(model.assemble_context("coffee and sleep")) or ""
        check(
            "graph-walk",
            edge_present and secret_omitted and "my brother" in rendered,
            "confirmed edge included; never_initiate node omitted; boundary in context",
        )

        # --- 3. proactive channel respects quiet hours + never_initiate ---
        config = proactive.ProactivityConfig(
            channels={
                proactive.CHECK_IN: proactive.ChannelConfig(
                    enabled=True, quiet_hours=proactive.QuietHours(start=22, end=8)
                )
            }
        )
        at_night = datetime(2026, 7, 12, 2, 0)
        at_day = datetime(2026, 7, 12, 14, 0)
        suppressed = not proactive.should_fire(proactive.CHECK_IN, at_night, config)
        fires = proactive.should_fire(proactive.CHECK_IN, at_day, config)
        boundary_blocks = not proactive.should_fire(
            proactive.CHECK_IN, at_day, config,
            topic="a check-in about my brother",
            never_initiate=model.never_initiate_topics(),
        )
        check(
            "proactive",
            suppressed and fires and boundary_blocks,
            "suppressed 02:00, fires 14:00, defers to never_initiate",
        )

        # --- 4. research KB: silent grounding + cited psychoeducation ---
        kb = ResearchKB()
        kb.ingest(
            "Body doubling for task initiation",
            "Brown 2021, ADHD Practice Review",
            "Body doubling supports task initiation in ADHD. Working beside "
            "another person makes it easier to start a task when you cannot get "
            "going.",
        )
        kb.ingest(
            "Deep pressure for sensory regulation",
            "Lee 2020, Occupational Therapy Review",
            "Deep pressure input can reduce sensory overload and support a "
            "calmer level of alertness during transitions.",
        )
        grounded = kb.ground("how do I start a task when I can't get going?")
        grounding_ok = grounded is not None and "body doubling" in grounded.lower()
        answer = kb.psychoeducation("what helps with task initiation?")
        sources = cast("list[dict[str, str]]", answer["sources"])
        cite_ok = bool(answer["answer"]) and any(
            s["ref"] == "Brown 2021, ADHD Practice Review" for s in sources
        )
        check(
            "research-kb",
            grounding_ok and cite_ok,
            "silent grounding picks body-doubling; psychoeducation cites its source",
        )

        # Crisis-resource config is exercised as a smoke check (W8).
        os.environ["THERAPY_CRISIS_CONTACTS"] = (
            '[{"label": "Línea 135", "value": "135"}]'
        )
        check("crisis-config", crisis_contacts()[0]["value"] == "135", "configurable")

    print("\n=== Phase-4 acceptance summary ===")
    for line in RESULTS:
        print(line)
    if any("FAIL" in line for line in RESULTS):
        print("\nFAIL — phase-4 acceptance not green.")
        return 1
    print("\nPASS — longitudinal loop, graph walk, proactivity, research KB verified.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
