"""Measure lossless OTLP round-trips against local observability backends.

The O0.3 spike sends the committed synthetic interaction corpus through an
OpenInference-shaped OpenTelemetry span, queries it through each backend's
official Python client, and records raw acceptance evidence. Phoenix's CLI also
starts an IPv6-any gRPC listener, so the ``serve-phoenix`` subcommand disables
that unused listener before invoking the pinned server entry point; HTTP remains
bound to the explicit loopback address.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypedDict, cast
from unittest.mock import patch

from openinference.semconv.resource import ResourceAttributes
from openinference.semconv.trace import (
    MessageAttributes,
    OpenInferenceMimeTypeValues,
    OpenInferenceSpanKindValues,
    SpanAttributes,
    ToolAttributes,
    ToolCallAttributes,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.id_generator import IdGenerator
from opentelemetry.trace import Status, StatusCode

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "tests/fixtures/observability"
INTERACTIONS_ROOT = FIXTURE_ROOT / "interactions"
PROJECT_NAME = "therapy-observability-spike"
CANONICAL_RECORD_ATTRIBUTE = "therapy.canonical_record"
INTERACTION_ID_ATTRIBUTE = "therapy.interaction_id"
QUERY_REPETITIONS = 20

type JsonScalar = None | str | bool | int | float
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type AttributeValue = str | bool | int | float
type Attributes = dict[str, AttributeValue]
type Query = Callable[[], list["ActualSpan"]]


@dataclass(frozen=True)
class Fixture:
    """One validated interaction fixture."""

    case: str
    record: dict[str, JsonValue]


@dataclass(frozen=True)
class ActualSpan:
    """Backend-independent projection of a queried span."""

    trace_id: str
    span_id: str
    attributes: dict[str, JsonValue]


class ProcessRow(TypedDict):
    """One typed row from the macOS ``ps`` process table."""

    pid: int
    ppid: int
    rss_bytes: int
    command: str


class CollisionOutcome(TypedDict):
    """Observed survivor for one duplicate fixture transport ID."""

    trace_id: str
    span_id: str
    fixture_order: list[str]
    retained_cases: list[str]


class CanaryScan(TypedDict):
    """Raw byte-scan evidence for one canary class."""

    counts: dict[str, int]
    files_scanned: int
    bytes_scanned: int
    unreadable_files: list[str]


@dataclass(frozen=True)
class ProcessUsage:
    """RSS snapshot for a server process tree."""

    root_rss_bytes: int
    tree_rss_bytes: int
    processes: list[ProcessRow]


class FixtureIdGenerator(IdGenerator):
    """Yield the trace and span IDs encoded in the fixture records."""

    def __init__(self, fixtures: Sequence[Fixture]) -> None:
        self._trace_ids = iter(
            int(_required_str(f.record, "trace_id"), 16) for f in fixtures
        )
        self._span_ids = iter(
            int(_required_str(f.record, "span_id"), 16) for f in fixtures
        )

    def generate_trace_id(self) -> int:
        """Return the next fixture trace ID."""
        return next(self._trace_ids)

    def generate_span_id(self) -> int:
        """Return the next fixture span ID."""
        return next(self._span_ids)


class CollectingExporter(SpanExporter):
    """Collect finished SDK spans so one explicit OTLP batch can be measured."""

    def __init__(self) -> None:
        self.spans: list[ReadableSpan] = []

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Append finished spans without I/O."""
        self.spans.extend(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        """Release no-op collector resources."""


def _required_str(value: Mapping[str, JsonValue], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"{key} must be a non-empty string")
    return item


def _string(value: Mapping[str, JsonValue], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str):
        raise ValueError(f"{key} must be a string")
    return item


def _required_dict(value: Mapping[str, JsonValue], key: str) -> dict[str, JsonValue]:
    item = value.get(key)
    if not isinstance(item, dict):
        raise ValueError(f"{key} must be an object with string keys")
    return item


def _optional_list(value: Mapping[str, JsonValue], key: str) -> list[JsonValue]:
    item = value.get(key, [])
    if not isinstance(item, list):
        raise ValueError(f"{key} must be an array")
    return item


def _canonical_json(value: JsonValue) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_value(value: object, path: str = "$") -> JsonValue:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, list):
        items = cast(list[object], value)
        return [
            _json_value(item, f"{path}[{index}]") for index, item in enumerate(items)
        ]
    if isinstance(value, dict):
        items = cast(dict[object, object], value)
        result: dict[str, JsonValue] = {}
        for key, item in items.items():
            if not isinstance(key, str):
                raise ValueError(f"{path}: JSON object key must be a string")
            result[key] = _json_value(item, f"{path}.{key}")
        return result
    raise ValueError(f"{path}: unsupported JSON value {type(value).__name__}")


def _json_object(value: object, path: str = "$") -> dict[str, JsonValue]:
    validated = _json_value(value, path)
    if not isinstance(validated, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return validated


def _string_mapping(value: Mapping[str, JsonValue], path: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(item, str):
            raise ValueError(f"{path}.{key} must be a string")
        result[key] = item
    return result


def _fixture_hash(root: Path = FIXTURE_ROOT) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x01")
    return digest.hexdigest()


def load_fixtures() -> list[Fixture]:
    """Load and validate the committed interaction fixture boundary."""
    fixtures: list[Fixture] = []
    for path in sorted(INTERACTIONS_ROOT.glob("*.json")):
        raw_payload: object = json.loads(path.read_text(encoding="utf-8"))
        payload = _json_object(raw_payload, str(path))
        case = _required_str(payload, "case")
        record = _required_dict(payload, "record")
        trace_id = _required_str(record, "trace_id")
        span_id = _required_str(record, "span_id")
        if len(trace_id) != 32 or len(span_id) != 16:
            raise ValueError(f"{path}: invalid OTLP trace/span ID length")
        int(trace_id, 16)
        int(span_id, 16)
        fixtures.append(Fixture(case=case, record=record))
    if len(fixtures) != 15:
        raise ValueError(f"expected 15 interaction fixtures, found {len(fixtures)}")
    return fixtures


def _set_message_attributes(
    attributes: Attributes,
    prefix: str,
    messages: Iterable[JsonValue],
) -> None:
    for index, item in enumerate(messages):
        if not isinstance(item, dict):
            raise ValueError(f"{prefix}[{index}] must be an object")
        role = item.get("role")
        content = item.get("content")
        base = f"{prefix}.{index}"
        if isinstance(role, str):
            attributes[f"{base}.{MessageAttributes.MESSAGE_ROLE}"] = role
        if isinstance(content, str):
            attributes[f"{base}.{MessageAttributes.MESSAGE_CONTENT}"] = content
        elif content is not None:
            attributes[f"{base}.{MessageAttributes.MESSAGE_CONTENT}"] = _canonical_json(
                content
            )


def build_attributes(record: dict[str, JsonValue]) -> Attributes:
    """Map a canonical record to OpenInference and TheraPy OTel attributes."""
    request = _required_dict(record, "request")
    response = _required_dict(record, "response")
    attributes: Attributes = {
        SpanAttributes.OPENINFERENCE_SPAN_KIND: OpenInferenceSpanKindValues.LLM.value,
        SpanAttributes.INPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
        SpanAttributes.OUTPUT_MIME_TYPE: OpenInferenceMimeTypeValues.JSON.value,
        SpanAttributes.INPUT_VALUE: _canonical_json(request),
        SpanAttributes.OUTPUT_VALUE: _canonical_json(response),
        SpanAttributes.SESSION_ID: _required_str(record, "session_id"),
        SpanAttributes.LLM_PROVIDER: _required_str(record, "provider"),
        SpanAttributes.LLM_SYSTEM: _required_str(record, "provider"),
        SpanAttributes.LLM_MODEL_NAME: _string(record, "actual_model"),
        SpanAttributes.LLM_PROMPT_TEMPLATE_VERSION: _required_str(
            record, "prompt_template_version"
        ),
        CANONICAL_RECORD_ATTRIBUTE: _canonical_json(record),
        INTERACTION_ID_ATTRIBUTE: _required_str(record, "interaction_id"),
        "therapy.turn_id": _required_turn_id(record),
        "therapy.operation": _required_str(record, "operation"),
        "therapy.provider": _required_str(record, "provider"),
        "therapy.requested_model": _required_str(record, "requested_model"),
        "therapy.actual_model": _string(record, "actual_model"),
        "therapy.status": _required_str(record, "status"),
    }

    _set_message_attributes(
        attributes,
        SpanAttributes.LLM_INPUT_MESSAGES,
        _optional_list(request, "messages"),
    )
    _set_message_attributes(
        attributes,
        SpanAttributes.LLM_OUTPUT_MESSAGES,
        _optional_list(response, "messages"),
    )

    parameters = request.get("parameters")
    if parameters is not None:
        attributes[SpanAttributes.LLM_INVOCATION_PARAMETERS] = _canonical_json(
            parameters
        )
    finish_reason = response.get("finish_reason")
    if isinstance(finish_reason, str):
        attributes[SpanAttributes.LLM_FINISH_REASON] = finish_reason

    for index, tool in enumerate(_optional_list(request, "tools")):
        attributes[
            f"{SpanAttributes.LLM_TOOLS}.{index}.{ToolAttributes.TOOL_JSON_SCHEMA}"
        ] = _canonical_json(tool)

    output_messages = _optional_list(response, "messages")
    tool_message_index = max(len(output_messages) - 1, 0)
    if not output_messages and _optional_list(response, "tool_calls"):
        role_key = (
            f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{tool_message_index}."
            f"{MessageAttributes.MESSAGE_ROLE}"
        )
        attributes[role_key] = "assistant"
    for index, tool_call in enumerate(_optional_list(response, "tool_calls")):
        if not isinstance(tool_call, dict):
            raise ValueError(f"response.tool_calls[{index}] must be an object")
        base = (
            f"{SpanAttributes.LLM_OUTPUT_MESSAGES}.{tool_message_index}."
            f"{MessageAttributes.MESSAGE_TOOL_CALLS}.{index}"
        )
        call_id = tool_call.get("id")
        name = tool_call.get("name")
        arguments = tool_call.get("arguments")
        if isinstance(call_id, str):
            attributes[f"{base}.{ToolCallAttributes.TOOL_CALL_ID}"] = call_id
        if isinstance(name, str):
            attributes[f"{base}.{ToolCallAttributes.TOOL_CALL_FUNCTION_NAME}"] = name
        if arguments is not None:
            attributes[
                f"{base}.{ToolCallAttributes.TOOL_CALL_FUNCTION_ARGUMENTS_JSON}"
            ] = _canonical_json(arguments)

    usage = response.get("usage")
    if isinstance(usage, dict):
        prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
        completion = usage.get("completion_tokens", usage.get("output_tokens"))
        total = usage.get("total_tokens")
        if isinstance(prompt, int):
            attributes[SpanAttributes.LLM_TOKEN_COUNT_PROMPT] = prompt
        if isinstance(completion, int):
            attributes[SpanAttributes.LLM_TOKEN_COUNT_COMPLETION] = completion
        if isinstance(total, int):
            attributes[SpanAttributes.LLM_TOKEN_COUNT_TOTAL] = total

    error = record.get("error")
    if isinstance(error, dict):
        attributes["error.type"] = str(error.get("provider_type") or "provider_error")
        attributes["therapy.error"] = _canonical_json(error)
        message = error.get("provider_error_body")
        if isinstance(message, str):
            attributes["error.message"] = message
    return attributes


def _required_turn_id(record: Mapping[str, JsonValue]) -> str | int:
    turn_id = record.get("turn_id")
    if isinstance(turn_id, bool) or not isinstance(turn_id, (str, int)):
        raise ValueError("turn_id must be a string or integer")
    return turn_id


def _unix_nanos(value: str) -> int:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp must include a timezone: {value}")
    return int(parsed.timestamp() * 1_000_000_000)


def build_spans(fixtures: Sequence[Fixture]) -> list[ReadableSpan]:
    """Build finished SDK spans while preserving fixture transport IDs."""
    collector = CollectingExporter()
    provider = TracerProvider(
        id_generator=FixtureIdGenerator(fixtures),
        resource=Resource.create(
            {
                "service.name": "therapy-observability-spike",
                ResourceAttributes.PROJECT_NAME: PROJECT_NAME,
            }
        ),
        span_limits=SpanLimits(
            max_span_attributes=10_000,
            max_span_attribute_length=None,
            max_events=64,
            max_event_attributes=128,
        ),
    )
    provider.add_span_processor(SimpleSpanProcessor(collector))
    tracer = provider.get_tracer("therapy.observability.spike", "O0.3")
    for fixture in fixtures:
        record = fixture.record
        span = tracer.start_span(
            name=f"llm.{_required_str(record, 'operation')}",
            start_time=_unix_nanos(_required_str(record, "started_at")),
        )
        span.set_attributes(build_attributes(record))
        error = record.get("error")
        if isinstance(error, dict):
            exception_type = str(error.get("provider_type") or "provider_error")
            event_attributes: Attributes = {
                "exception.type": exception_type,
                "exception.message": _canonical_json(error),
            }
            span.add_event("exception", event_attributes)
            span.set_status(Status(StatusCode.ERROR, exception_type))
        else:
            span.set_status(Status(StatusCode.OK))
        completed_at = record.get("completed_at")
        end_time = (
            _unix_nanos(completed_at)
            if isinstance(completed_at, str)
            else _unix_nanos(_required_str(record, "started_at"))
        )
        span.end(end_time=end_time)
    provider.shutdown()
    if len(collector.spans) != len(fixtures):
        raise RuntimeError("OTel SDK did not finish every fixture span")
    return collector.spans


def _base_url(endpoint: str) -> str:
    suffix = "/v1/traces"
    if not endpoint.endswith(suffix):
        raise ValueError(f"OTLP endpoint must end with {suffix}")
    return endpoint[: -len(suffix)]


def _phoenix_query(endpoint: str) -> tuple[Query, dict[str, object]]:
    from phoenix.client import Client

    client = Client(base_url=_base_url(endpoint))

    def query() -> list[ActualSpan]:
        spans = client.spans.get_spans(project_identifier=PROJECT_NAME, limit=1_000)
        actual: list[ActualSpan] = []
        for span in spans:
            context = span["context"]
            raw_attributes = span.get("attributes", {})
            actual.append(
                ActualSpan(
                    trace_id=str(context.get("trace_id", "")),
                    span_id=str(context.get("span_id", "")),
                    attributes=_json_object(dict(raw_attributes), "Phoenix attributes"),
                )
            )
        return actual

    return query, {"project_name": PROJECT_NAME, "query_api": "Client.spans.get_spans"}


def _mlflow_query(endpoint: str, storage_dir: Path) -> tuple[Query, dict[str, object]]:
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri=_base_url(endpoint))
    experiment = client.get_experiment_by_name(PROJECT_NAME)
    if experiment is None:
        artifact_dir = storage_dir / "artifacts" / PROJECT_NAME
        artifact_dir.mkdir(parents=True, exist_ok=True)
        experiment_id = client.create_experiment(
            PROJECT_NAME, artifact_location=artifact_dir.resolve().as_uri()
        )
    else:
        experiment_id = experiment.experiment_id

    def query() -> list[ActualSpan]:
        traces = client.search_traces(
            locations=[experiment_id], max_results=100, include_spans=True
        )
        actual: list[ActualSpan] = []
        for trace in traces:
            for span in trace.data.spans:
                raw_span: object = span.to_dict()
                span_payload = _json_object(raw_span, "MLflow span")
                encoded_trace_id = _required_str(span_payload, "trace_id")
                try:
                    trace_id = base64.b64decode(encoded_trace_id, validate=True).hex()
                except ValueError as exc:
                    raise ValueError(
                        "MLflow returned an invalid base64 trace ID"
                    ) from exc
                raw_attributes: object = span.attributes
                actual.append(
                    ActualSpan(
                        trace_id=trace_id,
                        span_id=span.span_id,
                        attributes=_json_object(raw_attributes, "MLflow attributes"),
                    )
                )
        return actual

    return query, {
        "experiment_id": experiment_id,
        "experiment_name": PROJECT_NAME,
        "query_api": "MlflowClient.search_traces",
    }


def _export(
    spans: Sequence[ReadableSpan],
    endpoint: str,
    headers: Mapping[str, str],
    *,
    timeout_seconds: float = 10.0,
) -> tuple[str, float]:
    exporter = OTLPSpanExporter(
        endpoint=endpoint,
        headers=dict(headers),
        timeout=timeout_seconds,
    )
    started = time.perf_counter()
    result = exporter.export(spans)
    elapsed_ms = (time.perf_counter() - started) * 1_000
    exporter.shutdown()
    return result.name, elapsed_ms


def _await_ingest(query: Query, timeout_seconds: float = 15.0) -> list[ActualSpan]:
    deadline = time.monotonic() + timeout_seconds
    previous_count = -1
    stable_samples = 0
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            spans = query()
            last_error = None
        except Exception as exc:  # query endpoint can race project creation
            last_error = exc
            time.sleep(0.2)
            continue
        count = len(spans)
        stable_samples = (
            stable_samples + 1 if count == previous_count and count > 0 else 0
        )
        if stable_samples >= 2:
            return spans
        previous_count = count
        time.sleep(0.2)
    if last_error is not None:
        raise TimeoutError("backend query never became available") from last_error
    return query()


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile * len(ordered)))
    return ordered[rank - 1]


def _measure_queries(query: Query) -> tuple[list[ActualSpan], list[float]]:
    latencies_ms: list[float] = []
    last: list[ActualSpan] = []
    for _ in range(QUERY_REPETITIONS):
        started = time.perf_counter()
        last = query()
        latencies_ms.append((time.perf_counter() - started) * 1_000)
    return last, latencies_ms


def _leaf_paths(value: JsonValue, path: str = "$") -> list[str]:
    if isinstance(value, dict):
        if not value:
            return [path]
        return [
            leaf
            for key in sorted(value)
            for leaf in _leaf_paths(value[key], f"{path}.{key}")
        ]
    if isinstance(value, list):
        if not value:
            return [path]
        return [
            leaf
            for index, item in enumerate(value)
            for leaf in _leaf_paths(item, f"{path}[{index}]")
        ]
    return [path]


def _record_diffs(
    expected: JsonValue, actual: JsonValue, path: str = "$"
) -> list[dict[str, str]]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        issues: list[dict[str, str]] = []
        for key in sorted(expected.keys() - actual.keys()):
            issues.extend(
                {"path": leaf, "kind": "lost"}
                for leaf in _leaf_paths(expected[key], f"{path}.{key}")
            )
        for key in sorted(expected.keys() & actual.keys()):
            issues.extend(_record_diffs(expected[key], actual[key], f"{path}.{key}"))
        return issues
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) == len(actual) and expected != actual:
            expected_items = Counter(_canonical_json(item) for item in expected)
            actual_items = Counter(_canonical_json(item) for item in actual)
            if expected_items == actual_items:
                return [{"path": path, "kind": "reordered"}]
        issues = []
        for index, item in enumerate(expected):
            if index >= len(actual):
                issues.extend(
                    {"path": leaf, "kind": "lost"}
                    for leaf in _leaf_paths(item, f"{path}[{index}]")
                )
            else:
                issues.extend(_record_diffs(item, actual[index], f"{path}[{index}]"))
        return issues
    if expected == actual and type(expected) is type(actual):
        return []
    kind = "changed"
    if (
        isinstance(expected, str)
        and isinstance(actual, str)
        and expected.startswith(actual)
    ):
        kind = "truncated"
    return [{"path": path, "kind": kind}]


def _round_trip_report(
    fixtures: Sequence[Fixture], actual_spans: Sequence[ActualSpan]
) -> dict[str, object]:
    by_interaction: dict[str, list[ActualSpan]] = defaultdict(list)
    for span in actual_spans:
        interaction_id = span.attributes.get(INTERACTION_ID_ATTRIBUTE)
        if isinstance(interaction_id, str):
            by_interaction[interaction_id].append(span)

    field_losses: list[dict[str, str]] = []
    attribute_losses: list[dict[str, str]] = []
    attribute_transformations: list[dict[str, str]] = []
    missing_cases: list[str] = []
    identity_mismatches: list[dict[str, str]] = []
    for fixture in fixtures:
        record = fixture.record
        interaction_id = _required_str(record, "interaction_id")
        candidates = by_interaction.get(interaction_id, [])
        if not candidates:
            missing_cases.append(fixture.case)
            field_losses.extend(
                {"case": fixture.case, "path": path, "kind": "lost"}
                for path in _leaf_paths(record)
            )
            continue
        actual = candidates[-1]
        expected_attributes = build_attributes(record)
        for key, expected in sorted(expected_attributes.items()):
            received = actual.attributes.get(key)
            if received != expected or type(received) is not type(expected):
                if isinstance(expected, str) and isinstance(received, (dict, list)):
                    if _canonical_json(received) == expected:
                        attribute_transformations.append(
                            {"case": fixture.case, "path": key, "kind": "json_decoded"}
                        )
                        continue
                kind = "lost" if key not in actual.attributes else "changed"
                if (
                    isinstance(expected, str)
                    and isinstance(received, str)
                    and expected.startswith(received)
                ):
                    kind = "truncated"
                attribute_losses.append(
                    {"case": fixture.case, "path": key, "kind": kind}
                )

        envelope = actual.attributes.get(CANONICAL_RECORD_ATTRIBUTE)
        if isinstance(envelope, str):
            try:
                raw_received_record: object = json.loads(envelope)
                received_record = _json_value(raw_received_record)
            except json.JSONDecodeError:
                field_losses.append(
                    {"case": fixture.case, "path": "$", "kind": "invalid_json"}
                )
            else:
                field_losses.extend(
                    {"case": fixture.case, **issue}
                    for issue in _record_diffs(record, received_record)
                )
        elif isinstance(envelope, dict):
            field_losses.extend(
                {"case": fixture.case, **issue}
                for issue in _record_diffs(record, envelope)
            )
        else:
            field_losses.extend(
                {"case": fixture.case, "path": path, "kind": "lost"}
                for path in _leaf_paths(record)
            )

        expected_trace = _required_str(record, "trace_id")
        expected_span = _required_str(record, "span_id")
        if actual.trace_id != expected_trace or actual.span_id != expected_span:
            identity_mismatches.append(
                {
                    "case": fixture.case,
                    "expected_trace_id": expected_trace,
                    "actual_trace_id": actual.trace_id,
                    "expected_span_id": expected_span,
                    "actual_span_id": actual.span_id,
                }
            )

    return {
        "queried_span_count": len(actual_spans),
        "matched_record_count": len(fixtures) - len(missing_cases),
        "missing_cases": missing_cases,
        "field_loss_count": len(field_losses),
        "field_losses": field_losses,
        "attribute_loss_count": len(attribute_losses),
        "attribute_losses": attribute_losses,
        "attribute_transformation_count": len(attribute_transformations),
        "attribute_transformations": attribute_transformations,
        "identity_mismatch_count": len(identity_mismatches),
        "identity_mismatches": identity_mismatches,
    }


def _collision_outcomes(
    fixtures: Sequence[Fixture], actual_spans: Sequence[ActualSpan]
) -> list[CollisionOutcome]:
    retained_ids = {
        value
        for span in actual_spans
        if isinstance((value := span.attributes.get(INTERACTION_ID_ATTRIBUTE)), str)
    }
    grouped: dict[tuple[str, str], list[Fixture]] = defaultdict(list)
    for fixture in fixtures:
        grouped[
            (
                _required_str(fixture.record, "trace_id"),
                _required_str(fixture.record, "span_id"),
            )
        ].append(fixture)
    outcomes: list[CollisionOutcome] = []
    for (trace_id, span_id), group in sorted(grouped.items()):
        if len(group) < 2:
            continue
        retained = [
            fixture.case
            for fixture in group
            if _required_str(fixture.record, "interaction_id") in retained_ids
        ]
        outcomes.append(
            {
                "trace_id": trace_id,
                "span_id": span_id,
                "fixture_order": [fixture.case for fixture in group],
                "retained_cases": retained,
            }
        )
    return outcomes


def _scan_storage(storage_dir: Path, canaries: Mapping[str, str]) -> CanaryScan:
    counts = {name: 0 for name in canaries}
    files_scanned = 0
    bytes_scanned = 0
    unreadable: list[str] = []
    for path in sorted(storage_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()
        except OSError:
            unreadable.append(str(path))
            continue
        files_scanned += 1
        bytes_scanned += len(data)
        for name, value in canaries.items():
            counts[name] += data.count(value.encode())
    return {
        "counts": counts,
        "files_scanned": files_scanned,
        "bytes_scanned": bytes_scanned,
        "unreadable_files": unreadable,
    }


def _disk_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _process_rows() -> list[ProcessRow]:
    output = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,rss=,command="],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout
    rows: list[ProcessRow] = []
    for line in output.splitlines():
        parts = line.strip().split(maxsplit=3)
        if len(parts) < 3:
            continue
        rows.append(
            {
                "pid": int(parts[0]),
                "ppid": int(parts[1]),
                "rss_bytes": int(parts[2]) * 1_024,
                "command": parts[3] if len(parts) == 4 else "",
            }
        )
    return rows


def _process_usage(root_pid: int) -> ProcessUsage:
    rows = _process_rows()
    by_parent: dict[int, list[int]] = defaultdict(list)
    by_pid: dict[int, ProcessRow] = {}
    for row in rows:
        pid = row["pid"]
        by_pid[pid] = row
        by_parent[row["ppid"]].append(pid)
    descendants: set[int] = set()
    pending = [root_pid]
    while pending:
        pid = pending.pop()
        if pid in descendants:
            continue
        descendants.add(pid)
        pending.extend(by_parent.get(pid, []))
    selected = [by_pid[pid] for pid in sorted(descendants) if pid in by_pid]
    root = by_pid.get(root_pid)
    return ProcessUsage(
        root_rss_bytes=root["rss_bytes"] if root else 0,
        tree_rss_bytes=sum(row["rss_bytes"] for row in selected),
        processes=selected,
    )


def _terminate_process_tree(root_pid: int) -> dict[str, object]:
    usage = _process_usage(root_pid)
    pids = [row["pid"] for row in usage.processes]
    for pid in reversed(pids):
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5
    alive = set(pids)
    while alive and time.monotonic() < deadline:
        alive = {pid for pid in alive if _pid_exists(pid)}
        time.sleep(0.05)
    for pid in alive:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return {"requested_pids": pids, "sigkill_pids": sorted(alive)}


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _packages(backend: str) -> dict[str, str]:
    backend_package = "arize-phoenix" if backend == "phoenix" else "mlflow"
    names = [
        backend_package,
        "opentelemetry-sdk",
        "opentelemetry-exporter-otlp-proto-http",
        "openinference-semantic-conventions",
    ]
    return {name: importlib.metadata.version(name) for name in names}


def _duplicate_fixture_ids(fixtures: Sequence[Fixture]) -> list[dict[str, object]]:
    cases: dict[tuple[str, str], list[str]] = defaultdict(list)
    for fixture in fixtures:
        cases[
            (
                _required_str(fixture.record, "trace_id"),
                _required_str(fixture.record, "span_id"),
            )
        ].append(fixture.case)
    return [
        {"trace_id": trace_id, "span_id": span_id, "cases": names}
        for (trace_id, span_id), names in sorted(cases.items())
        if len(names) > 1
    ]


def _environment(backend: str) -> dict[str, str]:
    keys = (
        [
            "PHOENIX_TELEMETRY_ENABLED",
            "PHOENIX_ALLOW_EXTERNAL_RESOURCES",
            "PHOENIX_DISABLE_AGENT_ASSISTANT",
            "PHOENIX_AGENTS_DISABLE_WEB_ACCESS",
            "PHOENIX_HOST",
            "PHOENIX_PORT",
            "PHOENIX_WORKING_DIR",
            "PHOENIX_SQL_DATABASE_URL",
        ]
        if backend == "phoenix"
        else ["MLFLOW_DISABLE_TELEMETRY", "DO_NOT_TRACK", "MLFLOW_TRACKING_URI"]
    )
    return {key: os.environ[key] for key in keys if key in os.environ}


def _write_result(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def measure(args: argparse.Namespace) -> int:
    """Execute the complete ingest/query/duplicate/leak/outage measurement."""
    backend = str(args.backend)
    endpoint = str(args.endpoint)
    storage_dir = Path(args.storage_dir).resolve()
    output = Path(args.output).resolve()
    server_pid = int(args.server_pid)
    storage_dir.mkdir(parents=True, exist_ok=True)
    fixtures = load_fixtures()
    spans = build_spans(fixtures)
    canonical_bytes = sum(
        len(_canonical_json(fixture.record).encode()) for fixture in fixtures
    )
    raw_canary_payload: object = json.loads(
        (FIXTURE_ROOT / "canaries.json").read_text()
    )
    canary_payload = _json_object(raw_canary_payload)
    content_canaries = _string_mapping(
        _required_dict(canary_payload, "content"), "$.content"
    )
    forbidden_canaries = _string_mapping(
        _required_dict(canary_payload, "forbidden"), "$.forbidden"
    )

    if backend == "phoenix":
        query, query_metadata = _phoenix_query(endpoint)
        headers: dict[str, str] = {}
    else:
        query, query_metadata = _mlflow_query(endpoint, storage_dir)
        headers = {"x-mlflow-experiment-id": str(query_metadata["experiment_id"])}

    first_export_result, first_export_ms = _export(spans, endpoint, headers)
    _await_ingest(query)
    disk_after_first = _disk_bytes(storage_dir)
    queried_before_duplicate, query_latencies = _measure_queries(query)
    round_trip = _round_trip_report(fixtures, queried_before_duplicate)

    duplicate_result, duplicate_export_ms = _export(spans, endpoint, headers)
    queried_after_duplicate = _await_ingest(query)
    disk_after_duplicate = _disk_bytes(storage_dir)
    collision_outcomes = _collision_outcomes(fixtures, queried_before_duplicate)
    retained_last_write = any(
        outcome["retained_cases"] == [outcome["fixture_order"][-1]]
        for outcome in collision_outcomes
    )
    duplicate_behavior = (
        "upserted_no_row_growth" if retained_last_write else "ignored_no_row_growth"
    )
    if len(queried_after_duplicate) > len(queried_before_duplicate):
        duplicate_behavior = "duplicated"
    elif duplicate_result != SpanExportResult.SUCCESS.name:
        duplicate_behavior = "rejected"
    elif [span.attributes for span in queried_after_duplicate] != [
        span.attributes for span in queried_before_duplicate
    ]:
        duplicate_behavior = "upserted_or_reordered"

    time.sleep(1)
    process_usage = _process_usage(server_pid)
    content_scan = _scan_storage(storage_dir, content_canaries)
    forbidden_scan = _scan_storage(storage_dir, forbidden_canaries)
    content_counts = content_scan["counts"]
    forbidden_counts = forbidden_scan["counts"]

    checks: dict[str, dict[str, object]] = {
        "fixture_count": {"passed": len(fixtures) == 15, "actual": len(fixtures)},
        "first_export": {
            "passed": first_export_result == SpanExportResult.SUCCESS.name,
            "actual": first_export_result,
        },
        "all_records_queryable": {
            "passed": not round_trip["missing_cases"],
            "missing_cases": round_trip["missing_cases"],
        },
        # Documented acceptance criterion (ADR, docs/evidence/
        # observability-backend-spike.md): a backend consuming
        # `openinference.span.kind` into its own top-level span-kind column
        # is a STRUCTURAL promotion — the value is preserved and recoverable
        # — not content loss. Everything else remains a hard failure.
        "attributes_round_trip": {
            "passed": all(
                loss.get("kind") == "lost"
                and loss.get("path") == "openinference.span.kind"
                for loss in round_trip["attribute_losses"]
            ),
            "loss_count": round_trip["attribute_loss_count"],
            "structural_promotions": sum(
                1
                for loss in round_trip["attribute_losses"]
                if loss.get("path") == "openinference.span.kind"
            ),
        },
        "attribute_representation_exact": {
            "passed": round_trip["attribute_transformation_count"] == 0,
            "transformation_count": round_trip["attribute_transformation_count"],
        },
        "canonical_fields_round_trip": {
            "passed": round_trip["field_loss_count"] == 0,
            "loss_count": round_trip["field_loss_count"],
        },
        "transport_ids_round_trip": {
            "passed": round_trip["identity_mismatch_count"] == 0,
            "mismatch_count": round_trip["identity_mismatch_count"],
        },
        "duplicate_probe_observed": {
            "passed": duplicate_behavior
            in {
                "ignored_no_row_growth",
                "upserted_no_row_growth",
                "duplicated",
                "rejected",
                "upserted_or_reordered",
            },
            "actual": duplicate_behavior,
        },
        "content_canaries_present": {
            "passed": all(int(count) > 0 for count in content_counts.values()),
            "counts": content_counts,
        },
        "forbidden_canaries_absent": {
            "passed": all(int(count) == 0 for count in forbidden_counts.values()),
            "counts": forbidden_counts,
        },
    }

    result: dict[str, object] = {
        "schema_version": 1,
        "backend": backend,
        "captured_at": datetime.now(UTC).isoformat(),
        "fixture_sha256": _fixture_hash(),
        "fixture_count": len(fixtures),
        "fixture_duplicate_transport_ids": _duplicate_fixture_ids(fixtures),
        "canonical_json_bytes": canonical_bytes,
        "package_versions": _packages(backend),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "endpoint": endpoint,
        "storage_path": str(storage_dir),
        "server_pid": server_pid,
        "server_command": str(args.server_command),
        "environment_variables": _environment(backend),
        "query": query_metadata,
        "measurements": {
            "first_export_wall_ms": first_export_ms,
            "duplicate_export_wall_ms": duplicate_export_ms,
            "query_repetitions": QUERY_REPETITIONS,
            "query_latency_ms": query_latencies,
            "query_p50_ms": _percentile(query_latencies, 0.50),
            "query_p95_ms": _percentile(query_latencies, 0.95),
            "server_root_rss_bytes": process_usage.root_rss_bytes,
            "server_tree_rss_bytes": process_usage.tree_rss_bytes,
            "server_processes": process_usage.processes,
            "disk_bytes_after_first_ingest": disk_after_first,
            "disk_bytes_after_duplicate_ingest": disk_after_duplicate,
            "disk_amplification_after_first": disk_after_first / canonical_bytes,
            "disk_amplification_after_duplicate": disk_after_duplicate
            / canonical_bytes,
        },
        "round_trip": round_trip,
        "duplicate_handling": {
            "first_query_span_count": len(queried_before_duplicate),
            "second_query_span_count": len(queried_after_duplicate),
            "second_export_result": duplicate_result,
            "behavior": duplicate_behavior,
            "first_batch_collision_outcomes": collision_outcomes,
        },
        "canary_scan": {
            "content": content_scan,
            "forbidden": forbidden_scan,
        },
        "checks": checks,
    }
    result["overall_pass"] = all(bool(check["passed"]) for check in checks.values())
    _write_result(output, result)

    if args.outage_probe:
        termination = _terminate_process_tree(server_pid)
        outage_result, outage_ms = _export(
            spans[:1], endpoint, headers, timeout_seconds=1.0
        )
        result["outage_probe"] = {
            "method": "SIGTERM server tree immediately before a one-span export",
            "termination": termination,
            "export_result": outage_result,
            "export_wall_ms": outage_ms,
            "exporter_reported_failure": outage_result == SpanExportResult.FAILURE.name,
        }
        checks["outage_failure_visible"] = {
            "passed": outage_result == SpanExportResult.FAILURE.name,
            "actual": outage_result,
        }
        result["overall_pass"] = all(bool(check["passed"]) for check in checks.values())
        _write_result(output, result)

    print(
        json.dumps(
            {
                "backend": backend,
                "output": str(output),
                "overall_pass": result["overall_pass"],
            }
        )
    )
    # A failing acceptance run must FAIL the shell gate (audit F-03).
    return 0 if result["overall_pass"] else 1


def serve_phoenix(args: argparse.Namespace) -> int:
    """Run Phoenix HTTP on loopback with its unused any-address gRPC server disabled."""
    if args.host != "127.0.0.1":
        raise ValueError("Phoenix spike server must bind to 127.0.0.1")
    from phoenix.server.grpc_server import GrpcServer
    from phoenix.server.main import main as phoenix_main

    class HttpOnlyGrpcServer(GrpcServer):
        async def __aenter__(self) -> None:
            return None

    sys.argv = [
        "phoenix",
        "serve",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--grpc-port",
        "0",
    ]
    with patch("phoenix.server.app.GrpcServer", HttpOnlyGrpcServer):
        phoenix_main()
    return 0


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse the server shim and measurement commands."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser("serve-phoenix")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, required=True)
    serve.set_defaults(handler=serve_phoenix)

    spike = subparsers.add_parser("measure")
    spike.add_argument("--backend", choices=("phoenix", "mlflow"), required=True)
    spike.add_argument("--endpoint", required=True)
    spike.add_argument("--storage-dir", type=Path, required=True)
    spike.add_argument("--server-pid", type=int, required=True)
    spike.add_argument("--server-command", required=True)
    spike.add_argument("--output", type=Path, required=True)
    spike.add_argument("--outage-probe", action="store_true")
    spike.set_defaults(handler=measure)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected O0.3 command."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
