"""Persistent proactivity delivery, restart, safety, and channel integration."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import threading
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest

from therapy.dialogue.outreach import (
    ProactivityScheduler,
    ProactivityService,
    VapidKeys,
    WebPushSender,
)
from therapy.knowledge.user_model import UserModel

type MetricCall = tuple[str, float, dict[str, str]]


def _discard_push(subscription: dict[str, object], payload: str) -> None:
    del subscription, payload


def _metric_recorder(
    calls: list[MetricCall],
) -> Callable[[str, float, dict[str, str] | None], None]:
    """Build one fully typed metric recorder test double."""

    def record(
        name: str, value: float, attrs: dict[str, str] | None = None
    ) -> None:
        calls.append((name, value, attrs or {}))

    return record


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
    service = ProactivityService(tmp_path, push_sender=_discard_push)

    assert [item["enabled"] for item in service.settings()] == [False] * 4

    with pytest.raises(ValueError, match="IANA timezone"):
        _settings(service, "check_in", timezone="Not/AZone")


def test_scheduler_survives_restart_and_period_job_is_idempotent(tmp_path: Path) -> None:
    now = datetime(2026, 7, 15, 20, tzinfo=UTC)
    first = ProactivityService(tmp_path, push_sender=_discard_push)
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

    restarted = ProactivityService(tmp_path, push_sender=_discard_push)
    assert restarted.ensure_scheduled(now) == 0
    assert len(restarted.jobs()) == 1
    assert restarted.tick(now=now) == 1
    assert restarted.jobs()[0]["state"] == "delivered"
    assert restarted.in_app_messages()[0]["channel"] == "check_in"


def test_scheduler_loop_survives_repeated_tick_failures_and_stops_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from therapy.observability import telemetry
    from therapy.observability.logging import BroadJsonFormatter

    service = ProactivityService(tmp_path, push_sender=_discard_push)
    tick_calls = 0

    def fail_tick() -> int:
        nonlocal tick_calls
        tick_calls += 1
        raise RuntimeError("private-scheduler-failure-canary")

    monkeypatch.setattr(service, "tick", fail_tick)
    metric_calls: list[tuple[str, float, dict[str, str]]] = []
    monkeypatch.setattr(telemetry, "record_metric", _metric_recorder(metric_calls))

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(
        BroadJsonFormatter(service_version="0.1.0", environment="test")
    )
    logger = logging.getLogger("therapy.broad")
    previous_handlers = logger.handlers
    previous_level = logger.level
    previous_propagate = logger.propagate
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False

    async def scenario() -> None:
        scheduler = ProactivityScheduler(service)
        scheduler.interval_seconds = 0.01
        task = asyncio.create_task(scheduler.run())
        deadline = asyncio.get_running_loop().time() + 1.0
        while tick_calls < 3 and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)
        assert tick_calls >= 3
        assert not task.done()
        scheduler.stop()
        await asyncio.wait_for(task, timeout=0.5)

    try:
        asyncio.run(scenario())
    finally:
        logger.handlers = previous_handlers
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate

    events = [json.loads(line) for line in stream.getvalue().splitlines()]
    failures = [
        event for event in events if event["event.name"] == "scheduler.tick_failed"
    ]
    assert len(failures) == tick_calls
    assert all(event["outcome"] == "error" for event in failures)
    assert all(event["error.type"] == "RuntimeError" for event in failures)
    assert sum(
        1
        for name, value, attrs in metric_calls
        if name == "therapy_proactivity_ticks_total"
        and value == 1
        and attrs == {"outcome": "error"}
    ) == tick_calls
    broad_payload = stream.getvalue()
    assert "private-scheduler-failure-canary" not in broad_payload
    assert "/Users/" not in broad_payload


def test_scheduler_hanging_tick_respects_bounded_task_shutdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = ProactivityService(tmp_path, push_sender=_discard_push)
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def hang_tick() -> int:
        started.set()
        try:
            release.wait()
        finally:
            finished.set()
        return 0

    monkeypatch.setattr(service, "tick", hang_tick)

    async def scenario() -> None:
        scheduler = ProactivityScheduler(service)
        task = asyncio.create_task(scheduler.run())
        try:
            assert await asyncio.to_thread(started.wait, 1.0)
            scheduler.stop()
            before = asyncio.get_running_loop().time()
            with pytest.raises(TimeoutError):
                await asyncio.wait_for(task, timeout=0.05)
            assert asyncio.get_running_loop().time() - before < 0.5
            assert task.cancelled()
        finally:
            release.set()
            assert await asyncio.to_thread(finished.wait, 1.0)
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())


def test_delivery_rechecks_never_initiate_immediately_before_send(tmp_path: Path) -> None:
    model = UserModel(tmp_path)
    service = ProactivityService(tmp_path, model=model, push_sender=_discard_push)
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
    service = ProactivityService(tmp_path, push_sender=_discard_push)
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
    next_attempt_at = job["next_attempt_at"]
    assert isinstance(next_attempt_at, str)
    assert datetime.fromisoformat(next_attempt_at) > now


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
    service = ProactivityService(tmp_path, model=model, push_sender=_discard_push)
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
    digest_content = service.digests()[0]["content"]
    assert isinstance(digest_content, str)
    assert statement in digest_content
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


def test_proactivity_job_and_subscription_metrics_cover_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from therapy.observability import telemetry

    service = ProactivityService(tmp_path, push_sender=_discard_push)
    calls: list[tuple[str, float, dict[str, str]]] = []
    monkeypatch.setattr(telemetry, "record_metric", _metric_recorder(calls))
    _settings(service, "check_in", quiet_start="00:00", quiet_end="00:00")
    now = datetime(2026, 7, 17, 12, tzinfo=UTC)
    service.enqueue("check_in", now - timedelta(seconds=5), idempotency_key="metric")

    assert service.tick(now=now) >= 1
    _settings(service, "check_in", enabled=False)
    service.enqueue(
        "check_in", now - timedelta(seconds=1), idempotency_key="metric-suppressed"
    )
    assert service.tick(now=now) >= 1
    subscription_id = service.subscribe("https://push.example/metric", "key", "auth")
    service.subscribe("https://push.example/metric", "key-2", "auth-2")
    assert service.unsubscribe(subscription_id) is True

    jobs = [
        attrs
        for name, value, attrs in calls
        if name == "therapy_proactivity_jobs_total" and value == 1
    ]
    assert {item["stage"] for item in jobs} >= {
        "due",
        "claimed",
        "delivered",
        "suppressed",
        "finalized",
    }
    assert all(item["channel"] == "check_in" for item in jobs)
    assert any(
        name == "therapy_proactivity_oldest_due_age_seconds" and value >= 5
        for name, value, _ in calls
    )
    queue_series = {
        (attrs["state"], attrs["channel"])
        for name, _, attrs in calls
        if name == "therapy_proactivity_queue_jobs"
    }
    assert queue_series == {
        (state, channel)
        for state in {"pending", "retry", "processing"}
        for channel in {"push", "greeting", "check_in", "digest"}
    }
    assert (
        "therapy_proactivity_suppressions_total",
        1,
        {"reason": "channel_disabled", "channel": "check_in"},
    ) in calls
    subscription_events = [
        attrs["event"]
        for name, _, attrs in calls
        if name == "therapy_webpush_subscription_events_total"
    ]
    assert subscription_events == ["created", "refreshed", "invalidated"]
    active_counts = [
        value
        for name, value, _ in calls
        if name == "therapy_webpush_active_subscriptions"
    ]
    assert active_counts[-3:] == [1, 1, 0]
    assert "push.example" not in repr(calls)


def test_webpush_adapter_records_bounded_status_without_exception_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pywebpush

    from therapy.observability import telemetry

    keys = VapidKeys(tmp_path)
    monkeypatch.setattr(keys, "ensure", lambda: None)
    calls: list[tuple[str, float, dict[str, str]]] = []
    monkeypatch.setattr(telemetry, "record_metric", _metric_recorder(calls))
    sender = WebPushSender(keys)
    def successful_webpush(**_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(status_code=201)

    monkeypatch.setattr(pywebpush, "webpush", successful_webpush)
    subscription: dict[str, object] = {
        "endpoint": "private-endpoint-canary",
        "keys": {"auth": "private-auth-canary", "p256dh": "private-key-canary"},
    }
    sender(subscription, "private-payload-canary")

    class RejectedPush(RuntimeError):
        response = SimpleNamespace(status_code=410)

    def reject(**_kwargs: object) -> None:
        raise RejectedPush("private-provider-body-canary")

    monkeypatch.setattr(pywebpush, "webpush", reject)
    with pytest.raises(RejectedPush, match="private-provider-body"):
        sender(subscription, "private-payload-canary")

    statuses = [
        attrs["status_class"]
        for name, _, attrs in calls
        if name == "therapy_webpush_deliveries_total"
    ]
    assert statuses == ["2xx", "4xx"]
    ttl_values = [
        value for name, value, _ in calls if name == "therapy_webpush_ttl_seconds"
    ]
    assert ttl_values == [3_600, 3_600]
    assert "private-" not in repr(calls)
