"""Persistent, restart-safe delivery for the four opt-in proactivity channels."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import sqlite3
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Literal, TypedDict, cast
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from therapy.dialogue.proactive import CHANNELS, CHECK_IN, GREETING, PUSH
from therapy.knowledge.schema import migrate_database
from therapy.knowledge.user_model import UserModel
from therapy.observability.logging import emit_event

type Channel = Literal["push", "greeting", "check_in", "digest"]
type JobState = Literal[
    "pending", "processing", "retry", "delivered", "suppressed", "failed"
]
type PushSender = Callable[[dict[str, object], str], None]

_DEFAULT_DATA_DIR = Path.home() / ".local" / "share" / "therapy"
_MAX_ATTEMPTS = 3

logger = logging.getLogger(__name__)
_GENERIC_PUSH = json.dumps(
    {"type": "reflection_available", "title": "TheraPy", "url": "/#model"},
    separators=(",", ":"),
)


class ChannelSettings(TypedDict):
    """Persisted owner controls for one channel."""

    channel: Channel
    enabled: bool
    timezone: str
    quiet_start: str
    quiet_end: str
    schedule_time: str
    schedule_day: int
    frequency: Literal["daily", "weekly"]
    topic: str | None
    updated_at: str


class OutreachJob(TypedDict):
    """Decoded persistent job ledger record."""

    id: str
    channel: Channel
    due_at: str
    idempotency_key: str
    topic: str | None
    payload: dict[str, object]
    state: JobState
    attempt_count: int
    next_attempt_at: str | None
    result: dict[str, object] | None
    created_at: str
    updated_at: str
    delivered_at: str | None


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return (
        parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)
    )


def _parse_clock(value: str) -> time:
    try:
        parsed = time.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("time must use HH:MM") from exc
    if parsed.second or parsed.microsecond:
        raise ValueError("time must use HH:MM")
    return parsed


def _timezone(value: str) -> ZoneInfo:
    try:
        return ZoneInfo(value)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f"Unknown IANA timezone: {value!r}") from exc


def _matches(topic: str, boundaries: list[str]) -> bool:
    folded = topic.casefold()
    return any(boundary.casefold() in folded for boundary in boundaries)


def _quiet(settings: ChannelSettings, now: datetime) -> tuple[bool, datetime | None]:
    zone = _timezone(settings["timezone"])
    local = now.astimezone(zone)
    start = _parse_clock(settings["quiet_start"])
    end = _parse_clock(settings["quiet_end"])
    start_minutes = start.hour * 60 + start.minute
    end_minutes = end.hour * 60 + end.minute
    current_minutes = local.hour * 60 + local.minute
    if start_minutes == end_minutes:
        return False, None
    inside = (
        start_minutes <= current_minutes < end_minutes
        if start_minutes < end_minutes
        else current_minutes >= start_minutes or current_minutes < end_minutes
    )
    if not inside:
        return False, None
    end_date = local.date()
    if start_minutes > end_minutes and current_minutes >= start_minutes:
        end_date += timedelta(days=1)
    local_end = datetime.combine(end_date, end, tzinfo=zone)
    return True, local_end.astimezone(UTC)


class VapidKeys:
    """Local VAPID key provisioner with a mode-0600 private key."""

    def __init__(self, data_dir: Path) -> None:
        self.path = data_dir / "vapid-private.pem"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def ensure(self) -> None:
        """Create the P-256 key once, entirely on the owner host."""
        if self.path.exists():
            os.chmod(self.path, 0o600)
            return
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ec

        key = ec.generate_private_key(ec.SECP256R1())
        payload = key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)

    def public_key(self) -> str:
        """Return URL-safe uncompressed P-256 application-server key."""
        from cryptography.hazmat.primitives import serialization

        self.ensure()
        private = serialization.load_pem_private_key(
            self.path.read_bytes(), password=None
        )
        public = private.public_key().public_bytes(
            serialization.Encoding.X962,
            serialization.PublicFormat.UncompressedPoint,
        )
        return base64.urlsafe_b64encode(public).rstrip(b"=").decode("ascii")


class WebPushSender:
    """Encrypted Web Push adapter; imported lazily outside domain tests."""

    def __init__(self, keys: VapidKeys) -> None:
        self.keys = keys

    def __call__(self, subscription: dict[str, object], payload: str) -> None:
        """Send one minimal encrypted notification with a bounded timeout."""
        from pywebpush import webpush

        self.keys.ensure()
        webpush(
            subscription_info=subscription,
            data=payload,
            vapid_private_key=str(self.keys.path),
            vapid_claims={
                "sub": os.getenv("THERAPY_VAPID_SUBJECT", "mailto:therapy@localhost")
            },
            ttl=3_600,
            timeout=10,
        )


class ProactivityService:
    """Own settings, job creation, guarded delivery, retries, and owner inboxes."""

    def __init__(
        self,
        data_dir: Path | None = None,
        *,
        model: UserModel | None = None,
        push_sender: PushSender | None = None,
    ) -> None:
        self.data_dir = Path(
            data_dir or os.getenv("THERAPY_DATA_DIR", str(_DEFAULT_DATA_DIR))
        )
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.data_dir / "therapy.db"
        migrate_database(self.db_path)
        self.model = model or UserModel(self.data_dir)
        self.vapid = VapidKeys(self.data_dir)
        self.push_sender = push_sender or WebPushSender(self.vapid)
        with self._connect() as connection:
            with connection:
                connection.execute(
                    """
                    UPDATE proactivity_jobs SET state = 'retry',
                        next_attempt_at = ?, updated_at = ?
                    WHERE state = 'processing'
                    """,
                    (_iso(_utc_now()), _iso(_utc_now())),
                )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=30000")
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _settings(row: sqlite3.Row) -> ChannelSettings:
        item = dict(row)
        item["enabled"] = bool(item["enabled"])
        return cast(ChannelSettings, item)

    @staticmethod
    def _job(row: sqlite3.Row) -> OutreachJob:
        item = dict(row)
        item["payload"] = json.loads(str(item.pop("payload_json")))
        result = item.pop("result_json")
        item["result"] = json.loads(str(result)) if result else None
        return cast(OutreachJob, item)

    def settings(self) -> list[ChannelSettings]:
        """Return all four channel settings in canonical order."""
        with self._connect() as connection:
            rows = connection.execute("SELECT * FROM proactivity_settings").fetchall()
        by_channel = {row["channel"]: self._settings(row) for row in rows}
        return [by_channel[channel] for channel in CHANNELS]

    def channel_settings(self, channel: Channel) -> ChannelSettings:
        """Return one channel's settings."""
        if channel not in CHANNELS:
            raise ValueError(f"Unknown channel: {channel!r}")
        return next(item for item in self.settings() if item["channel"] == channel)

    def update_settings(
        self,
        channel: Channel,
        *,
        enabled: bool,
        timezone: str,
        quiet_start: str,
        quiet_end: str,
        schedule_time: str,
        schedule_day: int,
        frequency: Literal["daily", "weekly"],
        topic: str | None,
    ) -> ChannelSettings:
        """Validate and persist one opt-in channel contract."""
        if channel not in CHANNELS:
            raise ValueError(f"Unknown channel: {channel!r}")
        _timezone(timezone)
        _parse_clock(quiet_start)
        _parse_clock(quiet_end)
        _parse_clock(schedule_time)
        if not 0 <= schedule_day <= 6:
            raise ValueError("schedule_day must be between 0 and 6")
        normalized_topic = topic.strip() if topic else None
        if normalized_topic and len(normalized_topic) > 500:
            raise ValueError("topic exceeds 500 characters")
        now = _iso(_utc_now())
        with self._connect() as connection:
            with connection:
                connection.execute(
                    """
                    UPDATE proactivity_settings SET enabled = ?, timezone = ?,
                        quiet_start = ?, quiet_end = ?, schedule_time = ?,
                        schedule_day = ?, frequency = ?, topic = ?, updated_at = ?
                    WHERE channel = ?
                    """,
                    (
                        int(enabled),
                        timezone,
                        quiet_start,
                        quiet_end,
                        schedule_time,
                        schedule_day,
                        frequency,
                        normalized_topic,
                        now,
                        channel,
                    ),
                )
        return self.channel_settings(channel)

    def enqueue(
        self,
        channel: Channel,
        due_at: datetime,
        *,
        idempotency_key: str,
        topic: str | None = None,
        payload: Mapping[str, object] | None = None,
    ) -> str:
        """Idempotently append one bounded outreach job."""
        if channel not in CHANNELS:
            raise ValueError(f"Unknown channel: {channel!r}")
        if len(idempotency_key) > 300:
            raise ValueError("idempotency key is too long")
        job_id = uuid4().hex
        now = _iso(_utc_now())
        with self._connect() as connection:
            with connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO proactivity_jobs (
                        id, channel, due_at, idempotency_key, topic, payload_json,
                        state, attempt_count, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 'pending', 0, ?, ?)
                    """,
                    (
                        job_id,
                        channel,
                        _iso(due_at),
                        idempotency_key,
                        topic,
                        json.dumps(payload or {}, ensure_ascii=False, sort_keys=True),
                        now,
                        now,
                    ),
                )
                row = connection.execute(
                    "SELECT id FROM proactivity_jobs WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
        return str(row["id"])

    @staticmethod
    def _scheduled_due(
        settings: ChannelSettings, now: datetime
    ) -> tuple[datetime, str]:
        zone = _timezone(settings["timezone"])
        local_now = now.astimezone(zone)
        clock = _parse_clock(settings["schedule_time"])
        if settings["frequency"] == "daily":
            local_due = datetime.combine(local_now.date(), clock, tzinfo=zone)
            if local_due > local_now:
                local_due -= timedelta(days=1)
            period = local_due.date().isoformat()
        else:
            days_since = (local_now.weekday() - settings["schedule_day"]) % 7
            due_date = local_now.date() - timedelta(days=days_since)
            local_due = datetime.combine(due_date, clock, tzinfo=zone)
            if local_due > local_now:
                local_due -= timedelta(days=7)
            year, week, _ = local_due.isocalendar()
            period = f"{year}-W{week:02d}"
        return local_due.astimezone(UTC), period

    def ensure_scheduled(self, now: datetime | None = None) -> int:
        """Materialize the latest due period; uniqueness makes restarts safe."""
        current = (now or _utc_now()).astimezone(UTC)
        created = 0
        for settings in self.settings():
            if not settings["enabled"] or settings["channel"] == GREETING:
                continue
            due_at, period = self._scheduled_due(settings, current)
            key = f"schedule:{settings['channel']}:{settings['frequency']}:{period}"
            before = len(self.jobs())
            self.enqueue(
                settings["channel"],
                due_at,
                idempotency_key=key,
                topic=settings["topic"],
                payload={"period": period},
            )
            created += int(len(self.jobs()) > before)
        return created

    def queue_greeting(self, now: datetime | None = None) -> str | None:
        """Queue at most one opted-in in-app reflection per local day."""
        settings = self.channel_settings(GREETING)
        if not settings["enabled"]:
            return None
        current = (now or _utc_now()).astimezone(UTC)
        local_date = current.astimezone(_timezone(settings["timezone"])).date()
        return self.enqueue(
            GREETING,
            current,
            idempotency_key=f"open:greeting:{local_date.isoformat()}",
            topic=settings["topic"],
        )

    def jobs(self, *, state: JobState | None = None) -> list[OutreachJob]:
        """Return observable job ledger records."""
        query = "SELECT * FROM proactivity_jobs"
        params: tuple[str, ...] = ()
        if state is not None:
            query += " WHERE state = ?"
            params = (state,)
        query += " ORDER BY due_at, id"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [self._job(row) for row in rows]

    def _pending_statements(self, boundaries: list[str]) -> list[str]:
        from therapy.knowledge.insight import InsightService

        InsightService(self.model).sync_proposals()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT statement_snapshot FROM pending_insights
                WHERE state IN ('queued','delivered','snoozed')
                ORDER BY proposed_at LIMIT 10
                """
            ).fetchall()
        return [
            str(row["statement_snapshot"])
            for row in rows
            if not _matches(str(row["statement_snapshot"]), boundaries)
        ][:5]

    def _subscriptions(self) -> list[tuple[str, dict[str, object]]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, endpoint, p256dh, auth FROM push_subscriptions WHERE active = 1"
            ).fetchall()
        return [
            (
                str(row["id"]),
                {
                    "endpoint": row["endpoint"],
                    "keys": {"p256dh": row["p256dh"], "auth": row["auth"]},
                },
            )
            for row in rows
        ]

    def subscribe(self, endpoint: str, p256dh: str, auth: str) -> str:
        """Upsert one browser subscription without logging its secret values."""
        subscription_id = hashlib.sha256(endpoint.encode()).hexdigest()
        now = _iso(_utc_now())
        with self._connect() as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO push_subscriptions (
                        id, endpoint, p256dh, auth, active, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, 1, ?, ?)
                    ON CONFLICT(endpoint) DO UPDATE SET p256dh = excluded.p256dh,
                        auth = excluded.auth, active = 1, updated_at = excluded.updated_at
                    """,
                    (subscription_id, endpoint, p256dh, auth, now, now),
                )
        return subscription_id

    def unsubscribe(self, subscription_id: str) -> bool:
        """Deactivate a local browser push subscription."""
        with self._connect() as connection:
            with connection:
                cursor = connection.execute(
                    "UPDATE push_subscriptions SET active = 0, updated_at = ? WHERE id = ?",
                    (_iso(_utc_now()), subscription_id),
                )
        return cursor.rowcount > 0

    def _send_push(self) -> int:
        subscriptions = self._subscriptions()
        if not subscriptions:
            raise RuntimeError("push enabled but no active browser subscription exists")
        sent = 0
        for subscription_id, subscription in subscriptions:
            try:
                self.push_sender(subscription, _GENERIC_PUSH)
                sent += 1
            except Exception as exc:
                response = getattr(exc, "response", None)
                if response is not None and getattr(response, "status_code", None) in {
                    404,
                    410,
                }:
                    self.unsubscribe(subscription_id)
                raise
        return sent

    def _local_message(self, job: OutreachJob, message: str) -> None:
        with self._connect() as connection:
            with connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO in_app_messages (
                        id, job_id, channel, message, state, created_at
                    ) VALUES (?, ?, ?, ?, 'queued', ?)
                    """,
                    (uuid4().hex, job["id"], job["channel"], message, _iso(_utc_now())),
                )

    def _digest(self, job: OutreachJob, statements: list[str], now: datetime) -> None:
        period_end = now.date()
        period = str(job["payload"].get("period", ""))
        days = 1 if period.count("-") == 2 else 7
        period_start = period_end - timedelta(days=days)
        lines = ["Reflection digest"] + [f"• {statement}" for statement in statements]
        content = "\n".join(lines)
        with self._connect() as connection:
            with connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO digests (
                        id, job_id, period_start, period_end, content, state, created_at
                    ) VALUES (?, ?, ?, ?, ?, 'unread', ?)
                    """,
                    (
                        uuid4().hex,
                        job["id"],
                        period_start.isoformat(),
                        period_end.isoformat(),
                        content,
                        _iso(now),
                    ),
                )
        self._local_message(
            job, "A written reflection digest is ready when you want it."
        )

    def _deliver_content(
        self, job: OutreachJob, boundaries: list[str], now: datetime
    ) -> dict[str, object]:
        statements = self._pending_statements(boundaries)
        if job["channel"] == PUSH:
            if not statements and not job["topic"]:
                return {"suppressed": "nothing_due"}
            return {"push_subscriptions": self._send_push()}
        if job["channel"] == GREETING:
            if not statements:
                return {"suppressed": "nothing_due"}
            self._local_message(
                job,
                f"One observation is waiting for your take: {statements[0]}",
            )
            return {"in_app": 1}
        if job["channel"] == CHECK_IN:
            message = (
                f"A gentle check-in about {job['topic']} is here whenever useful."
                if job["topic"]
                else "A gentle check-in is here whenever it would be useful."
            )
            self._local_message(job, message)
            return {"in_app": 1}
        if not statements:
            return {"suppressed": "nothing_due"}
        self._digest(job, statements, now)
        push_settings = self.channel_settings(PUSH)
        push_sent = 0
        push_quiet, _ = _quiet(push_settings, now)
        if push_settings["enabled"] and not push_quiet and self._subscriptions():
            push_sent = self._send_push()
        return {"digest": 1, "push_subscriptions": push_sent}

    def deliver(self, job_id: str, *, now: datetime | None = None) -> OutreachJob:
        """Claim and deliver one job after re-fetching every safety guard."""
        current = (now or _utc_now()).astimezone(UTC)
        with self._connect() as connection:
            with connection:
                row = connection.execute(
                    "SELECT * FROM proactivity_jobs WHERE id = ?", (job_id,)
                ).fetchone()
                if row is None:
                    raise KeyError(job_id)
                job = self._job(row)
                ready_at = job["next_attempt_at"] or job["due_at"]
                if (
                    job["state"] not in {"pending", "retry"}
                    or _parse_datetime(ready_at) > current
                ):
                    return job
                cursor = connection.execute(
                    """
                    UPDATE proactivity_jobs SET state = 'processing',
                        attempt_count = attempt_count + 1, updated_at = ?
                    WHERE id = ? AND state IN ('pending','retry')
                    """,
                    (_iso(current), job_id),
                )
                if not cursor.rowcount:
                    return job
        settings = self.channel_settings(job["channel"])
        boundaries = self.model.never_initiate_topics()
        quiet, allowed_at = _quiet(settings, current)
        if not settings["enabled"]:
            return self._finish(
                job_id, "suppressed", {"reason": "channel_disabled"}, current
            )
        if job["topic"] and _matches(job["topic"], boundaries):
            return self._finish(
                job_id, "suppressed", {"reason": "never_initiate"}, current
            )
        if quiet and allowed_at is not None:
            return self._postpone(job_id, allowed_at, current)
        try:
            result = self._deliver_content(job, boundaries, current)
            if "suppressed" in result:
                return self._finish(
                    job_id, "suppressed", {"reason": result["suppressed"]}, current
                )
            return self._finish(job_id, "delivered", result, current)
        except Exception as exc:
            emit_event(
                "proactivity.delivery_failed",
                severity=logging.WARNING,
                component="proactivity",
                operation=str(job["channel"]),
                outcome="error",
                error_type=type(exc).__name__,
            )
            return self._retry_or_fail(job_id, str(exc), current)

    def _finish(
        self,
        job_id: str,
        state: Literal["delivered", "suppressed"],
        result: object,
        now: datetime,
    ) -> OutreachJob:
        with self._connect() as connection:
            with connection:
                connection.execute(
                    """
                    UPDATE proactivity_jobs SET state = ?, result_json = ?,
                        next_attempt_at = NULL, delivered_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        state,
                        json.dumps(result, ensure_ascii=False, sort_keys=True),
                        _iso(now) if state == "delivered" else None,
                        _iso(now),
                        job_id,
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM proactivity_jobs WHERE id = ?", (job_id,)
                ).fetchone()
        return self._job(row)

    def _postpone(
        self, job_id: str, allowed_at: datetime, now: datetime
    ) -> OutreachJob:
        with self._connect() as connection:
            with connection:
                connection.execute(
                    """
                    UPDATE proactivity_jobs SET state = 'retry', attempt_count = 0,
                        next_attempt_at = ?, result_json = ?, updated_at = ? WHERE id = ?
                    """,
                    (
                        _iso(allowed_at),
                        json.dumps({"reason": "quiet_hours"}),
                        _iso(now),
                        job_id,
                    ),
                )
                row = connection.execute(
                    "SELECT * FROM proactivity_jobs WHERE id = ?", (job_id,)
                ).fetchone()
        return self._job(row)

    def _retry_or_fail(self, job_id: str, error: str, now: datetime) -> OutreachJob:
        with self._connect() as connection:
            with connection:
                row = connection.execute(
                    "SELECT attempt_count FROM proactivity_jobs WHERE id = ?", (job_id,)
                ).fetchone()
                attempts = int(row["attempt_count"])
                if attempts >= _MAX_ATTEMPTS:
                    state: JobState = "failed"
                    retry_at = None
                else:
                    state = "retry"
                    digest = hashlib.sha256(f"{job_id}:{attempts}".encode()).digest()
                    jitter = digest[0] % 31
                    retry_at = _iso(
                        now + timedelta(seconds=(2**attempts) * 30 + jitter)
                    )
                connection.execute(
                    """
                    UPDATE proactivity_jobs SET state = ?, next_attempt_at = ?,
                        result_json = ?, updated_at = ? WHERE id = ?
                    """,
                    (
                        state,
                        retry_at,
                        json.dumps({"error": error[:500]}),
                        _iso(now),
                        job_id,
                    ),
                )
                updated = connection.execute(
                    "SELECT * FROM proactivity_jobs WHERE id = ?", (job_id,)
                ).fetchone()
        return self._job(updated)

    def tick(self, *, now: datetime | None = None, limit: int = 10) -> int:
        """Schedule and deliver a bounded due batch."""
        current = (now or _utc_now()).astimezone(UTC)
        self.ensure_scheduled(current)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id FROM proactivity_jobs
                WHERE state IN ('pending','retry')
                  AND COALESCE(next_attempt_at, due_at) <= ?
                ORDER BY COALESCE(next_attempt_at, due_at), id LIMIT ?
                """,
                (_iso(current), max(1, min(limit, 100))),
            ).fetchall()
        for row in rows:
            self.deliver(str(row["id"]), now=current)
        return len(rows)

    def in_app_messages(self, *, consume: bool = False) -> list[dict[str, object]]:
        """Return queued local greetings/check-ins, optionally marking seen."""
        with self._connect() as connection:
            with connection:
                rows = connection.execute(
                    "SELECT * FROM in_app_messages WHERE state = 'queued' ORDER BY created_at"
                ).fetchall()
                if consume and rows:
                    ids = [str(row["id"]) for row in rows]
                    marks = ",".join("?" for _ in ids)
                    connection.execute(
                        f"UPDATE in_app_messages SET state = 'seen', seen_at = ? "
                        f"WHERE id IN ({marks})",
                        (_iso(_utc_now()), *ids),
                    )
        return [dict(row) for row in rows]

    def digests(self) -> list[dict[str, object]]:
        """Return owner-local written digests, newest first."""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM digests ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def export_all(self) -> dict[str, object]:
        """Export settings, ledger, subscriptions, messages, and digests."""
        tables = (
            "proactivity_settings",
            "proactivity_jobs",
            "push_subscriptions",
            "in_app_messages",
            "digests",
        )
        with self._connect() as connection:
            return {
                table: [
                    dict(row) for row in connection.execute(f"SELECT * FROM {table}")
                ]
                for table in tables
            }

    def delete_all(self) -> None:
        """Delete all proactivity data and rotate VAPID identity."""
        with self._connect() as connection:
            with connection:
                for table in (
                    "in_app_messages",
                    "digests",
                    "push_subscriptions",
                    "proactivity_jobs",
                ):
                    connection.execute(f"DELETE FROM {table}")
                connection.execute("UPDATE proactivity_settings SET enabled = 0")
        self.vapid.path.unlink(missing_ok=True)


class ProactivityScheduler:
    """Bounded in-process scheduler whose durable state lives in SQLite."""

    def __init__(
        self, service: ProactivityService, *, interval_seconds: float = 30.0
    ) -> None:
        self.service = service
        self.interval_seconds = max(1.0, interval_seconds)
        self._stop = asyncio.Event()

    async def run(self) -> None:
        """Tick until application shutdown without blocking the event loop."""
        while not self._stop.is_set():
            try:
                processed = await asyncio.to_thread(self.service.tick)
                if processed:
                    # only non-empty batches get a trace root (obs plan
                    # O2.2/O2.3: empty ticks are metric-only, never spans)
                    from therapy.observability.context import current_trace_context
                    from therapy.observability.telemetry import link_root

                    parent = current_trace_context()
                    with link_root(
                        "proactivity.batch",
                        component="scheduler",
                        operation="tick",
                        parent_trace_id=parent.trace_id,
                        parent_span_id=parent.span_id,
                    ) as span:
                        if span is not None:
                            span.set_attribute("count", processed)
            except Exception as exc:
                emit_event(
                    "scheduler.tick_failed",
                    severity=logging.ERROR,
                    component="scheduler",
                    operation="tick",
                    outcome="error",
                    error_type=type(exc).__name__,
                    exc=exc,
                    owned_failure=True,
                )
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval_seconds)
            except TimeoutError:
                pass

    def stop(self) -> None:
        """Request prompt scheduler shutdown."""
        self._stop.set()


__all__ = [
    "ChannelSettings",
    "OutreachJob",
    "ProactivityScheduler",
    "ProactivityService",
    "VapidKeys",
]
