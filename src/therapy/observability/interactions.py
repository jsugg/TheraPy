"""Canonical interaction record: frozen, typed, exactly serialized (§5.2).

The `json-v1` wire shape is produced by `to_json_dict()` with explicit
boundary validation — `dict[str, Any]` is never the public interface.
Provider-native envelopes stay provider-shaped (they must round-trip the
exact wire evidence) but are validated to be JSON-serializable and are
checksummed over canonical UTF-8 bytes.

Raw audio never enters this record. Native timing units/values are kept
exact; any normalization happens in separate derived fields.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Self

from therapy.observability.model import (
    PAYLOAD_ENCODING,
    InteractionOperation,
    InteractionStatus,
    Provider,
)

SCHEMA_VERSION = 1

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def canonical_json(payload: dict[str, JsonValue]) -> str:
    """Deterministic canonical serialization used for storage + checksums."""
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def checksum(payload: dict[str, JsonValue]) -> str:
    """SHA-256 over canonical UTF-8 bytes; corruption/dedup detection only,
    never a substitute for the content itself (§5.3)."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _require_json_value(value: object, where: str) -> JsonValue:
    """Boundary validation: reject anything not exactly JSON-shaped."""
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        raise TypeError(f"{where}: non-finite float is not valid JSON")
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list | tuple):
        return [_require_json_value(item, where) for item in value]
    if isinstance(value, dict):
        out: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{where}: non-string key {key!r}")
            out[key] = _require_json_value(item, f"{where}.{key}")
        return out
    raise TypeError(f"{where}: {type(value).__name__} is not JSON-serializable")


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class TranscriptTurn:
    role: str
    language: str
    modality: str
    text: str


@dataclass(frozen=True, slots=True)
class RetrievedDocument:
    source_type: str
    source_id: str
    anchor: str
    score: float
    rank: int
    text: str


@dataclass(frozen=True, slots=True)
class ToolUse:
    definition: dict[str, JsonValue]
    arguments: dict[str, JsonValue] | None
    authorization_outcome: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class ToolResult:
    tool_call_id: str
    content: JsonValue
    is_error: bool


@dataclass(frozen=True, slots=True)
class Truncation:
    applied: bool
    dropped_messages: int


@dataclass(frozen=True, slots=True)
class InteractionRequest:
    system_instructions: str
    messages: tuple[Message, ...]
    transcript: tuple[TranscriptTurn, ...] = ()
    memory_notes: tuple[str, ...] = ()
    retrieved_documents: tuple[RetrievedDocument, ...] = ()
    tools: tuple[ToolUse, ...] = ()
    parameters: dict[str, JsonValue] = field(default_factory=dict)
    response_schema: dict[str, JsonValue] | None = None
    context_order: tuple[str, ...] = ("system", "memory", "retrieval", "messages")
    truncation: Truncation = field(default_factory=lambda: Truncation(False, 0))


@dataclass(frozen=True, slots=True)
class InteractionResponse:
    messages: tuple[Message, ...] = ()
    completion: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_results: tuple[ToolResult, ...] = ()
    finish_reason: str | None = None
    usage: dict[str, JsonValue] | None = None


@dataclass(frozen=True, slots=True)
class StreamEvent:
    sequence: int
    observed_at: str
    delta: str | None
    tool_delta: str | None = None


@dataclass(frozen=True, slots=True)
class InteractionError:
    """Exact provider failure evidence. Only explicitly parsed fields ever
    cross the provider boundary — raw headers remain forbidden (§5.2)."""

    http_status: int | None
    provider_type: str | None
    provider_code: str | None
    provider_error_body: str | None
    retry_attempt: int
    provider_request_id: str | None


@dataclass(frozen=True, slots=True)
class ProviderNative:
    """Exact provider wire evidence; shape is provider-owned by design."""

    request: dict[str, JsonValue]
    ordered_events: tuple[dict[str, JsonValue], ...] = ()
    terminal_response: dict[str, JsonValue] | None = None
    terminal_error: dict[str, JsonValue] | None = None
    extra: dict[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class InteractionRecord:
    """One provider attempt (§5.2 `json-v1`)."""

    interaction_id: str
    trace_id: str
    span_id: str
    operation: InteractionOperation
    provider: Provider
    requested_model: str
    actual_model: str
    prompt_template_version: str
    request: InteractionRequest
    response: InteractionResponse
    stream: tuple[StreamEvent, ...]
    error: InteractionError | None
    provider_native: ProviderNative
    language: str
    modality: str
    build_version: str
    policy_version: str
    config_version: str
    started_at: str
    completed_at: str | None
    status: InteractionStatus
    session_id: str | None = None
    turn_id: int | None = None

    def __post_init__(self) -> None:
        hex_digits = set("0123456789abcdef")
        if len(self.trace_id) != 32 or not set(self.trace_id) <= hex_digits:
            raise ValueError("trace_id must be 32 lowercase W3C hex digits")
        if len(self.span_id) != 16 or not set(self.span_id) <= hex_digits:
            raise ValueError("span_id must be 16 lowercase W3C hex digits")
        if not self.interaction_id:
            raise ValueError("interaction_id required")
        sequences = [event.sequence for event in self.stream]
        if sequences != sorted(sequences) or len(set(sequences)) != len(sequences):
            raise ValueError("stream sequences must be strictly increasing")

    def to_json_dict(self) -> dict[str, JsonValue]:
        """The validated `json-v1` shape (§5.2)."""
        payload = asdict(self)
        payload["operation"] = self.operation.value
        payload["provider"] = self.provider.value
        payload["status"] = self.status.value
        # flatten the provider-native `extra` seam into its envelope
        native = payload["provider_native"]
        extra = native.pop("extra", {}) or {}
        native.update(extra)
        return _require_json_value(payload, "interaction")  # type: ignore[return-value]

    def canonical(self) -> str:
        return canonical_json(self.to_json_dict())

    def checksum(self) -> str:
        return checksum(self.to_json_dict())

    @property
    def schema_version(self) -> int:
        return SCHEMA_VERSION

    @property
    def payload_encoding(self) -> str:
        return PAYLOAD_ENCODING

    def with_status(self, status: InteractionStatus, **changes: object) -> Self:
        from dataclasses import replace

        return replace(self, status=status, **changes)  # type: ignore[arg-type]
