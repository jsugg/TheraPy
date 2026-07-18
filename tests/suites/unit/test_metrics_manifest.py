"""Frozen instrument manifest and cardinality rules (plan §8, O2 gate)."""

import ast
import asyncio
import sqlite3
from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path

import pytest

from therapy.observability.metrics import INSTRUMENT_INDEX, INSTRUMENTS

type MetricCall = tuple[str, float, dict[str, str]]


def _metric_recorder(
    calls: list[MetricCall],
) -> Callable[[str, float, dict[str, str] | None], None]:
    def record(
        name: str, value: float, attrs: dict[str, str] | None = None
    ) -> None:
        calls.append((name, value, attrs or {}))

    return record


def test_names_are_unique_and_stable() -> None:
    assert len(INSTRUMENT_INDEX) == len(INSTRUMENTS)
    for spec in INSTRUMENTS:
        assert spec.name.startswith("therapy_"), spec.name
        assert spec.name == spec.name.lower()


def test_manifest_and_literal_metric_call_sites_stay_in_sync(repo_root: Path) -> None:
    """Every frozen metric is emitted, and no literal call bypasses the manifest."""
    called: set[str] = set()
    for path in (repo_root / "src" / "therapy").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            function = node.func
            is_metric_call = (
                isinstance(function, ast.Name)
                and function.id in {"record_metric", "_record_metric"}
            ) or (
                isinstance(function, ast.Attribute)
                and function.attr == "record_metric"
            )
            first = node.args[0]
            if (
                is_metric_call
                and isinstance(first, ast.Constant)
                and isinstance(first.value, str)
            ):
                called.add(first.value)

    assert called == set(INSTRUMENT_INDEX)


def test_enumerated_attribute_sets_stay_bounded() -> None:
    """Every label has an explicit finite value set (plan §8)."""
    for spec in INSTRUMENTS:
        for attr, values in spec.attributes.items():
            assert attr == attr.lower()
            assert values
            assert len(values) == len(set(values)), f"{spec.name}.{attr}"


def test_shape_valid_unknown_value_cannot_mint_a_series(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A label-looking value is still unknown unless the manifest lists it."""
    from therapy.observability import telemetry

    captured: dict[str, str] = {}

    class FakeCounter:
        def add(self, value: float, attributes: dict[str, str]) -> None:
            del value
            captured.update(attributes)

    monkeypatch.setattr(
        telemetry.state(), "instruments", {"therapy_llm_requests_total": FakeCounter()}
    )
    telemetry.record_metric(
        "therapy_llm_requests_total",
        1,
        {"provider": "synthetic-provider-123", "operation": "summary", "outcome": "success"},
    )
    assert captured["provider"] == "unknown"


def test_invalid_closed_label_or_missing_dimension_drops_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Labels without an unknown bucket fail closed instead of minting a series."""
    from therapy.observability import telemetry

    observations: list[dict[str, str]] = []

    class FakeHistogram:
        def record(self, value: float, attributes: dict[str, str]) -> None:
            del value
            observations.append(attributes)

    monkeypatch.setattr(
        telemetry.state(),
        "instruments",
        {"therapy_webrtc_rtt_seconds": FakeHistogram()},
    )
    telemetry.record_metric(
        "therapy_webrtc_rtt_seconds", 0.01, {"candidate_type": "private-canary"}
    )
    telemetry.record_metric("therapy_webrtc_rtt_seconds", 0.01)
    assert observations == []


def test_no_forbidden_label_dimensions() -> None:
    """Labels never carry IDs, models under dynamic routing, URLs, paths,
    timestamps, or content (plan §8)."""
    forbidden = {
        "session_id", "turn_id", "interaction_id", "job_id", "document_id",
        "model", "actual_model", "url", "path", "endpoint", "timestamp",
        "message", "text", "error_message",
    }
    for spec in INSTRUMENTS:
        overlap = forbidden & set(spec.attributes)
        assert not overlap, f"{spec.name}: {overlap}"


def test_record_metric_drops_undeclared_attributes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from therapy.observability import telemetry

    captured: dict[str, dict[str, str]] = {}

    class FakeCounter:
        def add(self, value: float, attributes: dict[str, str]) -> None:
            del value
            captured["attrs"] = attributes

    monkeypatch.setattr(
        telemetry.state(), "instruments", {"therapy_llm_requests_total": FakeCounter()}
    )
    telemetry.record_metric(
        "therapy_llm_requests_total",
        1,
        {
            "provider": "ollama",
            "operation": "summary",
            "outcome": "success",
            "session_id": "sess-leak",  # undeclared: must be dropped
        },
    )
    assert captured["attrs"] == {
        "provider": "ollama",
        "operation": "summary",
        "outcome": "success",
    }

    # exact plan enums without an unknown bucket fail closed
    class FakeCounter2(FakeCounter):
        pass

    previous = captured["attrs"]
    monkeypatch.setattr(
        telemetry.state(), "instruments", {"therapy_llm_output_total": FakeCounter2()}
    )
    telemetry.record_metric(
        "therapy_llm_output_total",
        1,
        {"provider": "ollama", "operation": "summary", "result": "made-up"},
    )
    assert captured["attrs"] is previous

    # unknown instruments are ignored, never created ad hoc
    telemetry.record_metric("therapy_not_in_manifest", 1, {})


def test_instrumented_thread_records_queue_active_completion_and_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from therapy.observability import telemetry
    from therapy.observability.model import WorkloadClass

    calls: list[tuple[str, float, dict[str, str]]] = []
    monkeypatch.setattr(telemetry, "record_metric", _metric_recorder(calls))

    def _identity(value: int) -> int:
        return value

    async def scenario() -> None:
        assert (
            await telemetry.run_in_thread(WorkloadClass.BATCH, _identity, 7)
            == 7
        )
        with pytest.raises(RuntimeError, match="private-thread-canary"):
            await telemetry.run_in_thread(
                WorkloadClass.BATCH,
                lambda: (_ for _ in ()).throw(RuntimeError("private-thread-canary")),
            )

    asyncio.run(scenario())

    names = [name for name, _, _ in calls]
    assert names.count("therapy_executor_queue_wait_seconds") == 2
    assert names.count("therapy_executor_active") == 4
    completed = [
        attrs["outcome"]
        for name, _, attrs in calls
        if name == "therapy_executor_completed_total"
    ]
    assert completed == ["success", "error"]
    assert all(
        attrs.get("workload_class") == "batch"
        for _, _, attrs in calls
        if attrs
    )
    assert "private-thread-canary" not in repr(calls)


def test_storage_operation_has_span_outcomes_and_bounded_result_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from therapy.observability import telemetry

    metrics: list[tuple[str, float, dict[str, str]]] = []
    spans: list[tuple[str, str, str]] = []

    @contextmanager
    def capture_span(
        name: str, *, component: str, operation: str
    ) -> Generator[None, None, None]:
        spans.append((name, component, operation))
        yield

    monkeypatch.setattr(telemetry, "broad_span", capture_span)
    monkeypatch.setattr(telemetry, "record_metric", _metric_recorder(metrics))

    with telemetry.storage_operation("memory", "query"):
        result = [1, 2, 3]
    telemetry.record_storage_result("memory", "query", result)
    with pytest.raises(sqlite3.OperationalError, match="private-busy-canary"):
        with telemetry.storage_operation("memory", "query"):
            raise sqlite3.OperationalError("database busy private-busy-canary")

    assert spans == [
        ("db.operation", "memory", "query"),
        ("db.operation", "memory", "query"),
    ]
    assert (
        "therapy_storage_result_rows_total",
        1,
        {"component": "memory", "operation": "query", "bucket": "1-9"},
    ) in metrics
    assert any(
        name == "therapy_storage_operations_total"
        and attrs["outcome"] == "error"
        for name, _, attrs in metrics
    )
    assert "private-busy-canary" not in repr(metrics)
