"""Plane classification, denylist, and canary scanning (plan §5.5, O0.2).

Framework-free: the OTel `PlaneRoutingSpanProcessor` in `telemetry.py` wraps
`classify_scope()`; tests and scripts use the scanner directly.

Default-deny: an unknown instrumentation scope routes NOWHERE. It is counted
by the caller (rate-limited diagnostic), never fanned out to both planes.
"""

from __future__ import annotations

from dataclasses import dataclass

from therapy.observability.model import TelemetryPlane

#: Audited scopes whose spans may carry exact content (restricted plane only).
#: `pipecat`/`pipecat.turn` come from the pinned 1.5.0 snapshot
#: (tests/fixtures/observability/pipecat/snapshot-1.5.0.json).
RESTRICTED_SCOPES: frozenset[str] = frozenset(
    {
        "pipecat",
        "pipecat.turn",
        "therapy.interactions",
    }
)

#: Allowlisted content-free scopes (broad plane).
BROAD_SCOPES: frozenset[str] = frozenset(
    {
        "therapy.broad",
        "opentelemetry.instrumentation.fastapi",
        "opentelemetry.instrumentation.asgi",
        "opentelemetry.instrumentation.httpx",
        "opentelemetry.instrumentation.system_metrics",
    }
)


def classify_scope(scope_name: str | None) -> TelemetryPlane | None:
    """Route a span's instrumentation scope; None means drop (default-deny)."""
    if not scope_name:
        return None
    if scope_name in RESTRICTED_SCOPES or scope_name.startswith("pipecat."):
        return TelemetryPlane.RESTRICTED
    if scope_name in BROAD_SCOPES:
        return TelemetryPlane.BROAD
    return None


#: Span attribute keys that may NEVER appear on a broad span (plan O2.1 and
#: §1 forbidden list). Exact keys plus prefix rules below.
BROAD_DENYLIST_KEYS: frozenset[str] = frozenset(
    {
        "url.query",
        "url.full",
        "http.target",
        "http.url",
        "client.address",
        "client.port",
        "net.peer.ip",
        "net.sock.peer.addr",
        "user_agent.original",
        "enduser.id",
        "session.id",
        "db.statement",
        "db.query.text",
        "exception.message",
    }
)

BROAD_DENYLIST_PREFIXES: tuple[str, ...] = (
    "http.request.header",
    "http.response.header",
)

#: Content-bearing attribute keys observed in the pinned Pipecat snapshot;
#: they exist only on restricted spans and must never survive a broad scrub.
CONTENT_ATTRIBUTE_KEYS: frozenset[str] = frozenset(
    {
        "transcript",
        "text",
        "text_output",
        "input",
        "output",
        "messages",
        "context_messages",
        "context_system_instruction",
        "instructions",
        "gen_ai.system_instructions",
        "tools",
        "tools.definitions",
        "arguments",
        "function_calls",
        "gen_ai.prompt",
        "gen_ai.completion",
        "input.value",
        "output.value",
    }
)


def is_forbidden_broad_key(key: str) -> bool:
    if key in BROAD_DENYLIST_KEYS or key in CONTENT_ATTRIBUTE_KEYS:
        return True
    return any(key.startswith(prefix) for prefix in BROAD_DENYLIST_PREFIXES)


@dataclass(frozen=True, slots=True)
class ScrubResult:
    attributes: dict[str, object]
    dropped_keys: tuple[str, ...]


def scrub_broad_attributes(attributes: dict[str, object]) -> ScrubResult:
    """Drop every denylisted/content key from a broad-bound attribute map.

    The dropped keys are returned so the caller can emit a rate-limited
    diagnostic (a drop on a broad span is a routing bug to surface, not
    silently normal).
    """
    clean: dict[str, object] = {}
    dropped: list[str] = []
    for key, value in attributes.items():
        if is_forbidden_broad_key(key):
            dropped.append(key)
        else:
            clean[key] = value
    return ScrubResult(attributes=clean, dropped_keys=tuple(dropped))


def find_canaries(text: str, canaries: dict[str, str]) -> list[str]:
    """Names of canaries present in `text` (runtime twin of the O0 scanner)."""
    return [name for name, value in canaries.items() if value in text]
