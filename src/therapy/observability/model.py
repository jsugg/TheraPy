"""Framework-free observability contracts (plan §3 dependency rule, §5.1).

Everything in this module is stdlib-only so product/domain code may import it
freely. Vendor SDK imports live in `telemetry.py` / `exporters.py` only; the
architecture tests enforce that split.

Unknown values arriving from external boundaries are never turned into labels:
`normalize_enum()` collapses them to the bounded ``unknown`` member.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum
from typing import Literal

#: Version of the bounded-enum vocabulary below. Bump when a member is added
#: so exported records/labels can be interpreted against the right set.
ENUM_SCHEMA_VERSION = 1

#: Serialized canonical-record payload encoding (journal `payload_encoding`).
PAYLOAD_ENCODING = "json-v1"


class TelemetryPlane(StrEnum):
    """The two isolated planes. There is no third destination."""

    RESTRICTED = "restricted"
    BROAD = "broad"


class FieldClassification(StrEnum):
    """Destination policy for one schema field (O0.2 destination matrix)."""

    RESTRICTED = "restricted"  # exact content; journal/restricted backend only
    BROAD = "broad"  # bounded enum/count/duration/size/status/ID
    FORBIDDEN = "forbidden"  # must not appear in either plane


class CaptureMode(StrEnum):
    """`THERAPY_CAPTURE_MODE`; evaluation fails closed on any evidence gap."""

    DISABLED = "disabled"
    RUNTIME = "runtime"
    EVALUATION = "evaluation"


class InteractionOperation(StrEnum):
    """Versioned operation vocabulary for LLM attempts (plan §5.1)."""

    REPLY = "reply"
    SUMMARY = "summary"
    DISTILL = "distill"
    JUDGE = "judge"
    RECAP = "recap"
    TITLE = "title"
    RESEARCH_GROUNDING = "research_grounding"
    TOOL = "tool"
    EVALUATION = "evaluation"


class InteractionStatus(StrEnum):
    """Monotonic journal lifecycle states (plan §5.3)."""

    STARTED = "started"
    STREAMING = "streaming"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


#: Legal monotonic transitions; identical repeats are idempotent successes,
#: anything else is a visible conflict (plan §5.3).
INTERACTION_STATUS_TRANSITIONS: dict[InteractionStatus, frozenset[InteractionStatus]] = {
    InteractionStatus.STARTED: frozenset(
        {
            InteractionStatus.STREAMING,
            InteractionStatus.SUCCEEDED,
            InteractionStatus.FAILED,
            InteractionStatus.INCOMPLETE,
        }
    ),
    InteractionStatus.STREAMING: frozenset(
        {
            InteractionStatus.SUCCEEDED,
            InteractionStatus.FAILED,
            InteractionStatus.INCOMPLETE,
        }
    ),
    InteractionStatus.SUCCEEDED: frozenset(),
    InteractionStatus.FAILED: frozenset(),
    InteractionStatus.INCOMPLETE: frozenset(),
}

TERMINAL_INTERACTION_STATUSES = frozenset(
    {
        InteractionStatus.SUCCEEDED,
        InteractionStatus.FAILED,
        InteractionStatus.INCOMPLETE,
    }
)


class InteractionEventKind(StrEnum):
    """Ordered per-attempt journal event kinds (plan §5.2 stream/native)."""

    STREAM_DELTA = "stream_delta"
    TOOL_DELTA = "tool_delta"
    PROVIDER_EVENT = "provider_event"
    RETRY = "retry"
    TERMINAL_RESPONSE = "terminal_response"
    TERMINAL_ERROR = "terminal_error"


class Provider(StrEnum):
    """LLM providers with owned wire adapters."""

    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    UNKNOWN = "unknown"


class Outcome(StrEnum):
    """Bounded broad-plane outcome for any instrumented operation."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    INCOMPLETE = "incomplete"
    UNKNOWN = "unknown"


class Component(StrEnum):
    """Bounded broad-plane component dimension."""

    SERVER = "server"
    VOICE = "voice"
    STT = "stt"
    TTS = "tts"
    LLM = "llm"
    MEMORY = "memory"
    KNOWLEDGE = "knowledge"
    RESEARCH = "research"
    PROACTIVITY = "proactivity"
    PUSH = "push"
    DATA = "data"
    SCHEDULER = "scheduler"
    JOURNAL = "journal"
    TELEMETRY = "telemetry"
    TURN = "turn"
    UNKNOWN = "unknown"


class WorkloadClass(StrEnum):
    """Finite executor/thread workload classes (plan §8 observer health)."""

    REALTIME = "realtime"
    BACKGROUND = "background"
    BATCH = "batch"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"


class LanguageGroup(StrEnum):
    """Bounded language dimension; raw codes never become labels."""

    EN = "en"
    ES = "es"
    PT = "pt"
    CODE_SWITCH = "code_switch"
    NONE = "none"  # silence / no speech
    OTHER = "other"
    UNKNOWN = "unknown"


class Modality(StrEnum):
    """Turn input modality."""

    VOICE = "voice"
    TEXT = "text"
    UNKNOWN = "unknown"


class Destination(StrEnum):
    """Finite outbound-network destinations (plan O2.1 item 3)."""

    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    OLLAMA = "ollama"
    WEBPUSH = "webpush"
    EMBEDDING_DOWNLOAD = "embedding_download"
    LLM_TELEMETRY = "llm_telemetry"
    OTLP = "otlp"
    UNKNOWN = "unknown"


def normalize_enum[E: Enum](value: object, enum: type[E], default: E) -> E:
    """Collapse arbitrary external input to a bounded member, never a label.

    Accepts a member, its value, or its (case-insensitive) name; everything
    else becomes `default` (the enum's bounded ``unknown``-style member).
    """
    if isinstance(value, enum):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        for member in enum:
            if member.value == lowered or member.name.lower() == lowered:
                return member
    return default


# --------------------------------------------------------------------------
# HTTP route policy manifest (plan O0.1 item 2)
# --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RoutePolicy:
    """One FastAPI operation and its broad-plane policy.

    `broad_traced` — whether the route emits a broad HTTP server span
    (liveness/static shell traffic is excluded per plan O2.1 item 2).
    `mutation_audit` — destructive/research/data operations that emit a
    minimal content-free audit event (plan O3.1).
    `test_only` — acceptance routes that must carry `test_mode=true` and
    alert outside explicit test deployments.
    """

    method: str
    path: str
    name: str
    broad_traced: bool = True
    mutation_audit: bool = False
    test_only: bool = False


HTTP_ROUTE_MANIFEST: tuple[RoutePolicy, ...] = (
    RoutePolicy("GET", "/health", "health", broad_traced=False),
    RoutePolicy("GET", "/ready", "ready", broad_traced=False),
    RoutePolicy("POST", "/api/offer", "offer"),
    RoutePolicy("POST", "/api/voice/disconnect", "disconnect_voice"),
    RoutePolicy("GET", "/api/ice-config", "ice_config"),
    RoutePolicy("GET", "/api/resumable", "resumable"),
    RoutePolicy("GET", "/api/sessions", "sessions"),
    RoutePolicy("GET", "/api/sessions/{session_id}", "session_detail"),
    RoutePolicy(
        "GET", "/api/sessions/{session_id}/turns/{turn_id}/audio", "turn_audio"
    ),
    RoutePolicy("PATCH", "/api/sessions/{session_id}", "rename_session"),
    RoutePolicy(
        "DELETE", "/api/sessions/{session_id}", "delete_session", mutation_audit=True
    ),
    RoutePolicy("GET", "/api/graph", "graph"),
    RoutePolicy("GET", "/api/graph/pending", "pending_insights"),
    RoutePolicy("GET", "/api/graph/nodes/{node_id}", "node_detail"),
    RoutePolicy("GET", "/api/graph/edges/{edge_id}", "edge_detail"),
    RoutePolicy("PATCH", "/api/graph/nodes/{node_id}", "edit_node"),
    RoutePolicy("POST", "/api/graph/nodes/{node_id}/confirm", "confirm_node"),
    RoutePolicy("POST", "/api/graph/nodes/{node_id}/reject", "reject_node"),
    RoutePolicy(
        "DELETE", "/api/graph/nodes/{node_id}", "delete_node", mutation_audit=True
    ),
    RoutePolicy("PATCH", "/api/graph/edges/{edge_id}", "edit_edge"),
    RoutePolicy("POST", "/api/graph/edges/{edge_id}/confirm", "confirm_edge"),
    RoutePolicy("POST", "/api/graph/edges/{edge_id}/reject", "reject_edge"),
    RoutePolicy(
        "DELETE", "/api/graph/edges/{edge_id}", "delete_edge", mutation_audit=True
    ),
    RoutePolicy("GET", "/api/graph/boundaries", "boundaries"),
    RoutePolicy("POST", "/api/graph/boundaries", "add_boundary"),
    RoutePolicy(
        "DELETE", "/api/graph/boundaries", "remove_boundary", mutation_audit=True
    ),
    RoutePolicy("POST", "/api/insights/{insight_id}/confirm", "confirm_insight"),
    RoutePolicy("POST", "/api/insights/{insight_id}/reject", "reject_insight"),
    RoutePolicy("POST", "/api/insights/{insight_id}/snooze", "snooze_insight"),
    RoutePolicy("POST", "/api/insights/{insight_id}/dismiss", "dismiss_insight"),
    RoutePolicy("GET", "/api/insights/{insight_id}/history", "insight_history"),
    RoutePolicy("GET", "/api/research", "research_documents"),
    RoutePolicy("POST", "/api/research/ingest", "research_ingest", mutation_audit=True),
    RoutePolicy("GET", "/api/research/query", "research_query"),
    RoutePolicy("GET", "/api/research/{document_id}", "research_document"),
    RoutePolicy(
        "PATCH",
        "/api/research/{document_id}/blocks/{anchor}",
        "correct_research_block",
        mutation_audit=True,
    ),
    RoutePolicy(
        "POST",
        "/api/research/{document_id}/reindex",
        "reindex_research",
        mutation_audit=True,
    ),
    RoutePolicy(
        "DELETE", "/api/research/{document_id}", "delete_research", mutation_audit=True
    ),
    RoutePolicy("GET", "/api/proactivity", "proactivity_settings"),
    RoutePolicy("PUT", "/api/proactivity/{channel}", "update_proactivity"),
    RoutePolicy("GET", "/api/proactivity/jobs", "proactivity_jobs"),
    RoutePolicy("GET", "/api/push/public-key", "push_public_key"),
    RoutePolicy("POST", "/api/push/subscriptions", "add_push_subscription"),
    RoutePolicy(
        "DELETE",
        "/api/push/subscriptions/{subscription_id}",
        "remove_push_subscription",
    ),
    RoutePolicy("GET", "/api/proactivity/in-app", "in_app_outreach"),
    RoutePolicy("GET", "/api/proactivity/digests", "proactivity_digests"),
    RoutePolicy("GET", "/api/data/export", "export_owner_data", mutation_audit=True),
    RoutePolicy("POST", "/api/data/restore", "restore_owner_data", mutation_audit=True),
    RoutePolicy("DELETE", "/api/data", "delete_owner_data", mutation_audit=True),
    RoutePolicy(
        "POST", "/api/testing/agent/turn", "acceptance_agent_turn", test_only=True
    ),
    RoutePolicy(
        "POST",
        "/api/testing/proactivity/run",
        "acceptance_proactivity_run",
        test_only=True,
    ),
    RoutePolicy(
        "POST", "/api/telemetry/client", "client_telemetry", broad_traced=False
    ),
    RoutePolicy("GET", "/api/crisis-resources", "crisis_resources_config"),
    RoutePolicy("GET", "/", "index", broad_traced=False),
)


# --------------------------------------------------------------------------
# LLM boundary manifest (plan O0.1 item 3)
# --------------------------------------------------------------------------


class FailPolicy(StrEnum):
    """What an LLM boundary does when pre-dispatch capture fails (§5.3)."""

    #: Runtime mode: continue after the documented safety/availability
    #: exception, returning a visible capture gap.
    FAIL_OPEN_WITH_GAP = "fail_open_with_gap"
    #: Evaluation mode / evaluation-only boundaries: stop immediately.
    FAIL_CLOSED = "fail_closed"


class ProviderPath(StrEnum):
    """Which wire seam serves a boundary."""

    #: Non-realtime single-shot completion via `memory.summarizer.complete`.
    COMPLETION_CLIENT = "completion_client"
    #: Realtime streaming service built by `pipecat.pipeline.make_llm_service`.
    PIPECAT_LLM_SERVICE = "pipecat_llm_service"


@dataclass(frozen=True, slots=True)
class LLMBoundary:
    """One audited LLM call site: where it lives and what it must capture."""

    name: str
    operation: InteractionOperation
    module: str  # dotted module containing the call site
    provider_path: ProviderPath
    #: Native evidence the adapter must retain (bounded event kinds).
    expected_evidence: tuple[InteractionEventKind, ...]
    fail_policy: FailPolicy


_NON_REALTIME_EVIDENCE = (
    InteractionEventKind.PROVIDER_EVENT,
    InteractionEventKind.RETRY,
    InteractionEventKind.TERMINAL_RESPONSE,
    InteractionEventKind.TERMINAL_ERROR,
)

_REALTIME_EVIDENCE = (
    InteractionEventKind.STREAM_DELTA,
    InteractionEventKind.TOOL_DELTA,
    InteractionEventKind.PROVIDER_EVENT,
    InteractionEventKind.RETRY,
    InteractionEventKind.TERMINAL_RESPONSE,
    InteractionEventKind.TERMINAL_ERROR,
)

LLM_BOUNDARY_MANIFEST: tuple[LLMBoundary, ...] = (
    LLMBoundary(
        name="session_summary",
        operation=InteractionOperation.SUMMARY,
        module="therapy.memory.summarizer",
        provider_path=ProviderPath.COMPLETION_CLIENT,
        expected_evidence=_NON_REALTIME_EVIDENCE,
        fail_policy=FailPolicy.FAIL_OPEN_WITH_GAP,
    ),
    LLMBoundary(
        name="session_title",
        operation=InteractionOperation.TITLE,
        module="therapy.memory.summarizer",
        provider_path=ProviderPath.COMPLETION_CLIENT,
        expected_evidence=_NON_REALTIME_EVIDENCE,
        fail_policy=FailPolicy.FAIL_OPEN_WITH_GAP,
    ),
    LLMBoundary(
        name="legacy_fact_distillation",
        operation=InteractionOperation.DISTILL,
        module="therapy.memory.distill",
        provider_path=ProviderPath.COMPLETION_CLIENT,
        expected_evidence=_NON_REALTIME_EVIDENCE,
        fail_policy=FailPolicy.FAIL_OPEN_WITH_GAP,
    ),
    LLMBoundary(
        name="knowledge_extraction",
        operation=InteractionOperation.DISTILL,
        module="therapy.knowledge.distill",
        provider_path=ProviderPath.COMPLETION_CLIENT,
        expected_evidence=_NON_REALTIME_EVIDENCE,
        fail_policy=FailPolicy.FAIL_OPEN_WITH_GAP,
    ),
    LLMBoundary(
        name="knowledge_judgment",
        operation=InteractionOperation.JUDGE,
        module="therapy.knowledge.distill",
        provider_path=ProviderPath.COMPLETION_CLIENT,
        expected_evidence=_NON_REALTIME_EVIDENCE,
        fail_policy=FailPolicy.FAIL_OPEN_WITH_GAP,
    ),
    LLMBoundary(
        name="insight_recap",
        operation=InteractionOperation.RECAP,
        module="therapy.knowledge.insight",
        provider_path=ProviderPath.COMPLETION_CLIENT,
        expected_evidence=_NON_REALTIME_EVIDENCE,
        fail_policy=FailPolicy.FAIL_OPEN_WITH_GAP,
    ),
    LLMBoundary(
        name="realtime_reply",
        operation=InteractionOperation.REPLY,
        module="therapy.integrations.pipecat.pipeline",
        provider_path=ProviderPath.PIPECAT_LLM_SERVICE,
        expected_evidence=_REALTIME_EVIDENCE,
        fail_policy=FailPolicy.FAIL_OPEN_WITH_GAP,
    ),
)


def route_manifest_json() -> list[dict[str, object]]:
    """Machine-readable route manifest (scripts/report tooling)."""
    return [
        {
            "method": route.method,
            "path": route.path,
            "name": route.name,
            "broad_traced": route.broad_traced,
            "mutation_audit": route.mutation_audit,
            "test_only": route.test_only,
        }
        for route in HTTP_ROUTE_MANIFEST
    ]


def llm_boundary_manifest_json() -> list[dict[str, object]]:
    """Machine-readable LLM-boundary manifest (scripts/report tooling)."""
    return [
        {
            "name": boundary.name,
            "operation": boundary.operation.value,
            "module": boundary.module,
            "provider_path": boundary.provider_path.value,
            "expected_evidence": [kind.value for kind in boundary.expected_evidence],
            "fail_policy": boundary.fail_policy.value,
        }
        for boundary in LLM_BOUNDARY_MANIFEST
    ]


# --------------------------------------------------------------------------
# Retrieval and tool boundary manifest (plan O3 gate)
# --------------------------------------------------------------------------

type RetrievalToolBoundaryKind = Literal["retrieval", "tool"]
type BoundaryEvidence = Literal["restricted_capture", "product_store"]


@dataclass(frozen=True, slots=True)
class RetrievalToolBoundary:
    """One audited retrieval or owner-operated tool boundary."""

    name: str
    kind: RetrievalToolBoundaryKind
    module: str
    entrypoint: str
    evidence: BoundaryEvidence
    notes: str


RETRIEVAL_TOOL_BOUNDARY_MANIFEST: tuple[RetrievalToolBoundary, ...] = (
    RetrievalToolBoundary(
        name="research_semantic_query",
        kind="retrieval",
        module="therapy.knowledge.research",
        entrypoint="ResearchKB.query",
        evidence="product_store",
        notes="Ranked passages and scores derive from the owned research store.",
    ),
    RetrievalToolBoundary(
        name="research_grounding_context",
        kind="retrieval",
        module="therapy.knowledge.research",
        entrypoint="ResearchKB.grounding_context",
        evidence="restricted_capture",
        notes="Rendered grounding enters the exact captured turn context.",
    ),
    RetrievalToolBoundary(
        name="turn_context_assembly",
        kind="retrieval",
        module="therapy.knowledge.context",
        entrypoint="ContextAssembler.assemble",
        evidence="restricted_capture",
        notes="Graph, episode, insight, and research selections enter captured context.",
    ),
    RetrievalToolBoundary(
        name="episodic_memory_selection",
        kind="retrieval",
        module="therapy.knowledge.context",
        entrypoint="ContextAssembler._episodes",
        evidence="restricted_capture",
        notes="Selected summaries and relevance scores enter captured turn context.",
    ),
    RetrievalToolBoundary(
        name="owner_destructive_routes",
        kind="tool",
        module="therapy.server.app",
        entrypoint="_audit",
        evidence="product_store",
        notes=(
            "Owner-authorized destructive routes commit through owned stores and "
            "emit a bounded terminal audit outcome; no model runtime invokes them."
        ),
    ),
)


def retrieval_tool_boundary_manifest_json() -> list[dict[str, str]]:
    """Machine-readable retrieval/tool boundary manifest."""
    return [
        {
            "name": boundary.name,
            "kind": boundary.kind,
            "module": boundary.module,
            "entrypoint": boundary.entrypoint,
            "evidence": boundary.evidence,
            "notes": boundary.notes,
        }
        for boundary in RETRIEVAL_TOOL_BOUNDARY_MANIFEST
    ]


#: Frozen bounded bucket vocabularies (plan O3.1/O3.3): raw counts and byte
#: sizes never become labels — only these enumerated buckets do.
COUNT_BUCKETS: tuple[str, ...] = ("0", "1-9", "10-99", "100-999", "1000+")
BYTE_BUCKETS: tuple[str, ...] = ("0", "<4k", "<64k", "<1m", "<16m", "16m+")


def count_bucket(count: int) -> str:
    """Map a result/row count to its frozen bucket label."""
    if count <= 0:
        return "0"
    if count < 10:
        return "1-9"
    if count < 100:
        return "10-99"
    if count < 1000:
        return "100-999"
    return "1000+"


def byte_bucket(size: int) -> str:
    """Map a byte size to its frozen bucket label."""
    if size <= 0:
        return "0"
    if size < 4_096:
        return "<4k"
    if size < 65_536:
        return "<64k"
    if size < 1_048_576:
        return "<1m"
    if size < 16_777_216:
        return "<16m"
    return "16m+"
