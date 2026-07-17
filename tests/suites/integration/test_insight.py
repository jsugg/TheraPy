"""Durable insight delivery and conversational resolution tests (Phase 4 D)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from therapy.knowledge import insight
from therapy.knowledge.user_model import UserModel

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


def _proposed_node(model: UserModel, statement: str = "Skips lunch when busy.") -> int:
    node_id: int | None = None
    for session_id in ("s1", "s1", "s2"):
        node_id = model.upsert_node("pattern", statement, session_id=session_id)
    assert node_id is not None
    assert model.propose(node_id)
    return node_id


def test_explicit_model_directory_overrides_process_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("THERAPY_DATA_DIR", str(tmp_path / "unrelated-default"))
    model = UserModel(tmp_path)
    service = insight.InsightService(model)

    assert service.sync_proposals() == 0
    assert service.data_dir == tmp_path


def test_cross_session_patterns_rank_auditable_and_hide_private(
    tmp_path: Path,
) -> None:
    model = UserModel(tmp_path)
    _proposed_node(model)
    model.upsert_node("pattern", "Watched a film once.", session_id="s1")
    private = model.upsert_node(
        "thread", "A private recurring worry.", never_initiate=True, session_id="s1"
    )
    model.upsert_node("thread", "A private recurring worry.", session_id="s2")
    assert private is not None

    statements = [node["statement"] for node in insight.cross_session_patterns(model)]

    assert "Skips lunch when busy." in statements
    assert "Watched a film once." not in statements
    assert "A private recurring worry." not in statements


def test_queue_records_snapshot_delivery_and_exact_confirmation(
    tmp_path: Path, metric_calls: list[MetricCall]
) -> None:
    model = UserModel(tmp_path)
    node_id = _proposed_node(model)
    service = insight.InsightService(model)
    metric_calls.clear()

    assert service.sync_proposals() == 1
    queued = service.list(state="queued")
    assert len(queued) == 1
    assert queued[0]["evidence_snapshot"]["auditable_sessions"] == 2
    assert service.resolve_conversational("session", "yes") is None

    delivered = service.next_for_topic("lunch and meals", "session")
    assert delivered is not None
    assert "Does that feel accurate" in delivered["reflection"]
    assert service.next_for_topic("lunch", "session") is None
    assert service.resolve_conversational("session", "maybe") is None

    resolved = service.resolve_conversational("session", "Yes, that fits.")
    assert resolved is not None
    assert resolved["state"] == "confirmed"
    assert model.get_node(node_id)["status"] == "confirmed"
    assert [event["event_type"] for event in service.history(resolved["id"])] == [
        "delivered",
        "confirmed",
    ]
    transitions = [
        attrs["state"]
        for name, _, attrs in metric_calls
        if name == "therapy_insight_transitions_total"
    ]
    assert transitions == ["proposed", "delivered", "confirmed"]
    assert [
        value
        for name, value, attrs in metric_calls
        if name == "therapy_insight_queue_depth" and attrs == {}
    ] == [1, 0, 0]


def test_rejection_updates_only_claim_just_delivered(
    tmp_path: Path, metric_calls: list[MetricCall]
) -> None:
    model = UserModel(tmp_path)
    first = _proposed_node(model, "Skips lunch when busy.")
    second = _proposed_node(model, "Skips breaks when busy.")
    service = insight.InsightService(model)
    service.sync_proposals()
    metric_calls.clear()

    delivered = service.next_for_topic("lunch", "session")
    assert delivered is not None
    resolved = service.resolve_conversational("session", "Não")

    assert resolved is not None
    assert resolved["claim_id"] == first
    assert model.get_node(first)["status"] == "rejected"
    assert model.get_node(second)["status"] == "proposed"
    assert (
        "therapy_insight_transitions_total",
        1,
        {"state": "rejected"},
    ) in metric_calls
    assert (
        "therapy_insight_queue_depth",
        1,
        {},
    ) in metric_calls


def test_snooze_and_dismiss_preserve_graph_proposal(
    tmp_path: Path, metric_calls: list[MetricCall]
) -> None:
    model = UserModel(tmp_path)
    node_id = _proposed_node(model)
    service = insight.InsightService(model)
    service.sync_proposals()
    queued = service.list(state="queued")[0]

    assert service.snooze(queued["id"], days=2)
    assert service.list(state="snoozed")[0]["snoozed_until"] is not None
    assert model.get_node(node_id)["status"] == "proposed"
    assert service.dismiss(queued["id"])
    assert service.list(state="dismissed")[0]["resolved_at"] is not None
    assert model.get_node(node_id)["status"] == "proposed"
    states = [
        attrs["state"]
        for name, _, attrs in metric_calls
        if name == "therapy_insight_transitions_total"
    ]
    assert states[-2:] == ["snoozed", "dismissed"]


def test_session_recap_uses_injected_recapper() -> None:
    async def recapper(transcript: str) -> str:
        return f"recap of {len(transcript)} chars"

    turns = [
        {"role": "user", "text": "hi", "language": "en", "modality": "text"}
    ]
    result = asyncio.run(insight.session_recap(turns, recapper=recapper))
    assert result.startswith("recap of")
    assert asyncio.run(insight.session_recap([], recapper=recapper)) == ""


def test_direct_resolution_and_claim_dismissal_emit_bounded_states(
    tmp_path: Path, metric_calls: list[MetricCall]
) -> None:
    model = UserModel(tmp_path)
    confirmed = _proposed_node(model, "Checks a private schedule.")
    dismissed = _proposed_node(model, "Reviews a private checklist.")
    service = insight.InsightService(model)
    metric_calls.clear()

    assert service.resolve_claim("node", confirmed, "confirmed")
    assert service.dismiss_claim("node", dismissed) == 1

    transitions = [
        attrs
        for name, _, attrs in metric_calls
        if name == "therapy_insight_transitions_total"
    ]
    assert {attrs["state"] for attrs in transitions} == {
        "proposed",
        "confirmed",
        "dismissed",
    }
    assert all(set(attrs) == {"state"} for attrs in transitions)
    assert all(
        attrs["state"]
        in {
            "proposed",
            "delivered",
            "snoozed",
            "dismissed",
            "confirmed",
            "rejected",
        }
        for attrs in transitions
    )
    gauges = [
        (value, attrs)
        for name, value, attrs in metric_calls
        if name == "therapy_insight_queue_depth"
    ]
    assert gauges
    assert all(attrs == {} and value >= 0 for value, attrs in gauges)
    encoded = repr(metric_calls).casefold()
    assert "private schedule" not in encoded
    assert "private checklist" not in encoded


def test_insight_expected_failure_emits_no_false_transition(
    tmp_path: Path, metric_calls: list[MetricCall]
) -> None:
    service = insight.InsightService(UserModel(tmp_path))
    metric_calls.clear()

    assert service.snooze("missing-insight") is False
    assert service.resolve("missing-insight", "rejected") is None
    assert service.dismiss("missing-insight") is False

    assert not any(
        name in {"therapy_insight_transitions_total", "therapy_insight_queue_depth"}
        for name, _, _ in metric_calls
    )


def test_insight_unexpected_exception_does_not_emit_content_or_false_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    metric_calls: list[MetricCall],
) -> None:
    model = UserModel(tmp_path)
    service = insight.InsightService(model)
    metric_calls.clear()

    def fail(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        raise RuntimeError("private topic, claim 42, insight secret-id")

    monkeypatch.setattr(model, "lifecycle_events", fail)
    with pytest.raises(RuntimeError, match="private topic"):
        service.sync_proposals()

    assert metric_calls == []
