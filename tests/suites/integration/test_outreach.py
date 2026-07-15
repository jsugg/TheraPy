"""Persistent proactivity delivery, restart, safety, and channel integration."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import pytest

from therapy.dialogue.outreach import ProactivityService
from therapy.knowledge.user_model import UserModel


def _settings(
    service: ProactivityService,
    channel: Literal["push", "greeting", "check_in", "digest"],
    *,
    enabled: bool = True,
    timezone: str = "UTC",
    quiet_start: str = "22:00",
    quiet_end: str = "08:00",
    schedule_time: str = "18:00",
    schedule_day: int = 6,
    frequency: Literal["daily", "weekly"] = "weekly",
    topic: str | None = None,
) -> None:
    service.update_settings(
        channel,
        enabled=enabled,
        timezone=timezone,
        quiet_start=quiet_start,
        quiet_end=quiet_end,
        schedule_time=schedule_time,
        schedule_day=schedule_day,
        frequency=frequency,
        topic=topic,
    )


def _proposal(model: UserModel, statement: str = "Late meetings drain energy.") -> int:
    claim_id: int | None = None
    for index in range(3):
        claim_id = model.upsert_node(
            "pattern",
            statement,
            session_id=f"session-{index}",
            evidence_key=f"evidence-{index}",
        )
    assert claim_id is not None
    assert model.propose(claim_id)
    return claim_id


def test_channels_default_off_and_settings_validate_iana_zone(tmp_path: Path) -> None:
    service = ProactivityService(tmp_path, push_sender=lambda *_: None)

    assert [item["enabled"] for item in service.settings()] == [False] * 4

    with pytest.raises(ValueError, match="IANA timezone"):
        _settings(service, "check_in", timezone="Not/AZone")


def test_scheduler_survives_restart_and_period_job_is_idempotent(tmp_path: Path) -> None:
    now = datetime(2026, 7, 15, 20, tzinfo=UTC)
    first = ProactivityService(tmp_path, push_sender=lambda *_: None)
    _settings(
        first,
        "check_in",
        quiet_start="00:00",
        quiet_end="00:00",
        schedule_time="19:00",
        frequency="daily",
        topic="weekly planning",
    )

    assert first.ensure_scheduled(now) == 1
    assert first.ensure_scheduled(now) == 0

    restarted = ProactivityService(tmp_path, push_sender=lambda *_: None)
    assert restarted.ensure_scheduled(now) == 0
    assert len(restarted.jobs()) == 1
    assert restarted.tick(now=now) == 1
    assert restarted.jobs()[0]["state"] == "delivered"
    assert restarted.in_app_messages()[0]["channel"] == "check_in"


def test_delivery_rechecks_never_initiate_immediately_before_send(tmp_path: Path) -> None:
    model = UserModel(tmp_path)
    service = ProactivityService(tmp_path, model=model, push_sender=lambda *_: None)
    _settings(
        service,
        "check_in",
        quiet_start="00:00",
        quiet_end="00:00",
        topic="family conflict",
    )
    job_id = service.enqueue(
        "check_in", datetime.now(UTC), idempotency_key="guarded", topic="family conflict"
    )
    model.add_boundary("never_initiate", "family")

    job = service.deliver(job_id)

    assert job["state"] == "suppressed"
    assert job["result"] == {"reason": "never_initiate"}
    assert service.in_app_messages() == []


def test_overnight_quiet_hours_postpone_across_dst_zone(tmp_path: Path) -> None:
    service = ProactivityService(tmp_path, push_sender=lambda *_: None)
    _settings(
        service,
        "check_in",
        timezone="America/New_York",
        quiet_start="22:00",
        quiet_end="08:00",
        topic="planning",
    )
    # 01:30 local on the 2026 fall-back date is quiet regardless of fold.
    now = datetime(2026, 11, 1, 5, 30, tzinfo=UTC)
    job_id = service.enqueue("check_in", now, idempotency_key="dst", topic="planning")

    job = service.deliver(job_id, now=now)

    assert job["state"] == "retry"
    assert job["attempt_count"] == 0
    assert datetime.fromisoformat(job["next_attempt_at"]) > now


def test_push_is_encrypted_adapter_payload_without_sensitive_statement(
    tmp_path: Path,
) -> None:
    model = UserModel(tmp_path)
    statement = "A private but allowed recurring pattern."
    _proposal(model, statement)
    deliveries: list[tuple[dict[str, object], str]] = []
    service = ProactivityService(
        tmp_path,
        model=model,
        push_sender=lambda subscription, payload: deliveries.append(
            (subscription, payload)
        ),
    )
    _settings(service, "push", quiet_start="00:00", quiet_end="00:00")
    service.subscribe("https://push.example/owner", "p256dh", "auth")
    job_id = service.enqueue("push", datetime.now(UTC), idempotency_key="push-one")

    assert service.deliver(job_id)["state"] == "delivered"
    assert len(deliveries) == 1
    assert statement not in deliveries[0][1]
    assert "reflection_available" in deliveries[0][1]


def test_greeting_and_digest_surface_pending_insight_locally(tmp_path: Path) -> None:
    model = UserModel(tmp_path)
    statement = "Transitions go better after a written checklist."
    _proposal(model, statement)
    service = ProactivityService(tmp_path, model=model, push_sender=lambda *_: None)
    _settings(service, "greeting", quiet_start="00:00", quiet_end="00:00")
    _settings(service, "digest", quiet_start="00:00", quiet_end="00:00")
    now = datetime(2026, 7, 15, 20, tzinfo=UTC)

    greeting_id = service.queue_greeting(now)
    assert greeting_id is not None
    assert service.deliver(greeting_id, now=now)["state"] == "delivered"
    digest_id = service.enqueue(
        "digest",
        now,
        idempotency_key="digest-one",
        payload={"period": "2026-W29"},
    )
    assert service.deliver(digest_id, now=now)["state"] == "delivered"

    messages = service.in_app_messages()
    assert any(statement in str(item["message"]) for item in messages)
    assert statement in service.digests()[0]["content"]
    assert len(service.in_app_messages(consume=True)) == 2
    assert service.in_app_messages() == []


def test_external_delivery_retries_are_bounded_and_idempotent(tmp_path: Path) -> None:
    model = UserModel(tmp_path)
    _proposal(model)

    def fail(_subscription: dict[str, object], _payload: str) -> None:
        raise TimeoutError("push timed out")

    service = ProactivityService(tmp_path, model=model, push_sender=fail)
    _settings(service, "push", quiet_start="00:00", quiet_end="00:00")
    service.subscribe("https://push.example/retry", "p256dh", "auth")
    start = datetime(2026, 7, 15, 20, tzinfo=UTC)
    job_id = service.enqueue("push", start, idempotency_key="retry-one")

    job = service.deliver(job_id, now=start)
    for _ in range(2):
        assert job["next_attempt_at"] is not None
        retry = datetime.fromisoformat(job["next_attempt_at"]) + timedelta(seconds=1)
        job = service.deliver(job_id, now=retry)

    assert job["state"] == "failed"
    assert job["attempt_count"] == 3
    assert len(service.jobs()) == 1
