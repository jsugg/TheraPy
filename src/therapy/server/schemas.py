"""Validated HTTP request contracts for the local owner API."""

from datetime import datetime
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)


class APIRequest(BaseModel):
    """Forbid accidental or model-invented request fields."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class RenameSessionRequest(APIRequest):
    """Owner-selected session title."""

    title: str = Field(min_length=1, max_length=80)


class GraphNodePatch(APIRequest):
    """Editable node fields."""

    statement: str | None = Field(default=None, min_length=1, max_length=2_000)
    never_initiate: StrictBool | None = None

    @model_validator(mode="after")
    def require_change(self) -> "GraphNodePatch":
        """Require at least one explicit field."""
        if self.statement is None and self.never_initiate is None:
            raise ValueError("at least one editable field is required")
        return self


class GraphEdgePatch(APIRequest):
    """Editable edge claim fields."""

    statement: str = Field(min_length=1, max_length=2_000)


class BoundaryRequest(APIRequest):
    """Privacy boundary write contract."""

    kind: Literal["never_store", "never_initiate"]
    value: str = Field(min_length=1, max_length=500)


class InsightSnoozeRequest(APIRequest):
    """Bounded pending-insight snooze."""

    days: int = Field(default=7, ge=1, le=365)


class ResearchBlockCorrection(APIRequest):
    """Owner correction for one stable OCR block anchor."""

    text: str = Field(min_length=1, max_length=20_000)


class DeleteAllRequest(APIRequest):
    """Explicit irreversible owner-data deletion confirmation."""

    confirmation: Literal["DELETE EVERYTHING"]


class AcceptanceAgentTurn(APIRequest):
    """Deterministic test-only agent turn."""

    session_id: str | None = Field(default=None, min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=5_000)
    language: Literal["es", "en", "pt"] = "en"
    finalize: StrictBool = False


class AcceptanceOutreachRun(APIRequest):
    """Deterministic test-only invocation of the persistent delivery boundary."""

    channel: Literal["push", "greeting", "check_in", "digest"]
    due_at: datetime
    now: datetime
    idempotency_key: str = Field(
        min_length=1, max_length=300, pattern=r"^[A-Za-z0-9:_-]+$"
    )
    topic: str | None = Field(default=None, max_length=500)

    @field_validator("due_at", "now")
    @classmethod
    def require_aware_datetime(cls, value: datetime) -> datetime:
        """Reject local-time ambiguity at the scheduler test boundary."""
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("datetime must include a UTC offset")
        return value


class PushSubscriptionRequest(APIRequest):
    """Browser push subscription; endpoint and keys stay owner-local."""

    endpoint: str = Field(min_length=1, max_length=4_096)
    p256dh: str = Field(min_length=1, max_length=512)
    auth: str = Field(min_length=1, max_length=512)

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, value: str) -> str:
        """Accept only encrypted web-push endpoints."""
        if not value.startswith("https://"):
            raise ValueError("push endpoint must use https")
        return value


class ProactivityChannelPatch(APIRequest):
    """Owner-controlled channel settings."""

    enabled: StrictBool
    timezone: str = Field(min_length=1, max_length=100)
    quiet_start: str = Field(pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")
    quiet_end: str = Field(pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")
    schedule_time: str = Field(pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")
    schedule_day: int = Field(default=6, ge=0, le=6)
    frequency: Literal["daily", "weekly"] = "weekly"
    topic: str | None = Field(default=None, max_length=500)


class ClientTelemetryEvent(APIRequest):
    """One strict first-party browser event (obs plan O4.1).

    Names/enums are finite; numbers are finite and range-bound; free text,
    URLs, IDs, stacks, SDP, and media/device data are unrepresentable.
    Strict mode: lax coercions (numeric strings, booleans as numbers,
    integral floats as ints) are rejected at this untrusted boundary
    (O4 audit F-03).
    """

    model_config = ConfigDict(strict=True)

    name: Literal[
        "media_permission",
        "signaling_state",
        "ice_state",
        "data_channel_state",
        "peer_state",
        "transcript_echo_timeout",
        "playback_failure",
        "disconnect",
        "webrtc_sample",
        "sw_lifecycle",
        "shell_fetch",
        "cache_fallback",
        "cache_recovery",
        "push_lifecycle",
    ]
    outcome: Literal[
        "success",
        "error",
        "timeout",
        "fallback",
        "recovered",
        "denied",
        "granted",
        "received",
        "shown",
        "clicked",
        "connected",
        "disconnected",
        "failed",
        "installed",
        "activated",
        "refreshed",
        "deactivated",
    ] = "success"
    duration_ms: float | None = Field(
        default=None, ge=0, le=600_000, allow_inf_nan=False
    )
    rtt_ms: float | None = Field(default=None, ge=0, le=60_000, allow_inf_nan=False)
    jitter_ms: float | None = Field(
        default=None, ge=0, le=10_000, allow_inf_nan=False
    )
    packet_loss_ratio: float | None = Field(
        default=None, ge=0.0, le=1.0, allow_inf_nan=False
    )
    bitrate_kbps: float | None = Field(
        default=None, ge=0, le=1_000_000, allow_inf_nan=False
    )
    bytes_delta: int | None = Field(default=None, ge=0, le=10_000_000_000)
    concealed_samples: int | None = Field(default=None, ge=0, le=1_000_000_000)
    candidate_type: Literal["relay", "host", "srflx"] | None = None
    dropped_events: int = Field(default=0, ge=0, le=10_000)


class ClientTelemetryBatch(APIRequest):
    """Bounded batch: schema v1, at most 20 events (obs plan O4.1)."""

    model_config = ConfigDict(strict=True)

    schema_version: Literal[1]
    events: list[ClientTelemetryEvent] = Field(min_length=1, max_length=20)

    @field_validator("schema_version", mode="before")
    @classmethod
    def reject_boolean_version(cls, value: object) -> object:
        """`Literal[1]` admits `True` (bool subclasses int) even in strict mode."""
        if isinstance(value, bool):
            raise ValueError("schema_version must be the integer 1")
        return value
