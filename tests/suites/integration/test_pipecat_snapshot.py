"""Installed-Pipecat telemetry surface vs. the committed snapshot (plan O0.1).

Runs only where Pipecat is installed (the therapy container). An upgrade that
changes emitted span attributes, tracer scopes, or MetricsFrame payloads must
fail here before it can silently change what reaches either telemetry plane.
"""

import inspect
import json
from pathlib import Path

import pytest

from therapy.observability.interactions import JsonValue, require_json_object

pipecat = pytest.importorskip("pipecat")

SNAPSHOT = (
    Path(__file__).resolve().parents[2]
    / "fixtures/observability/pipecat/snapshot-1.5.0.json"
)


@pytest.fixture(scope="module")
def snapshot() -> dict[str, JsonValue]:
    payload: object = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    return require_json_object(payload, "pipecat snapshot")


def test_installed_version_matches_snapshot(snapshot: dict[str, JsonValue]) -> None:
    import importlib.metadata as metadata

    assert metadata.version("pipecat-ai") == snapshot["pipecat_version"]


def test_pipeline_worker_still_accepts_required_telemetry_parameters(
    snapshot: dict[str, JsonValue],
) -> None:
    from pipecat.pipeline.worker import PipelineWorker

    initializer = vars(PipelineWorker).get("__init__")
    assert callable(initializer)
    params = set(inspect.signature(initializer).parameters)
    required = {
        "enable_tracing",
        "enable_turn_tracking",
        "observers",
        "conversation_id",
        "additional_span_attributes",
    }
    assert required <= params
    assert sorted(params - {"self"}) == snapshot["pipeline_worker_parameters"]


def test_metrics_data_classes_match_snapshot(snapshot: dict[str, JsonValue]) -> None:
    import pipecat.metrics.metrics as metrics_module

    installed = {}
    for name, cls in inspect.getmembers(metrics_module, inspect.isclass):
        if cls.__module__ != metrics_module.__name__:
            continue
        fields = getattr(cls, "model_fields", None)
        installed[name] = (
            sorted(fields) if fields else sorted(vars(cls).get("__annotations__", {}))
        )
    assert installed == snapshot["metrics_data_classes"]


def test_tracer_scopes_match_snapshot(snapshot: dict[str, JsonValue]) -> None:
    """The routing processor's audited scope list (`pipecat`, `pipecat.turn`)
    must cover every scope the installed Pipecat requests tracers under."""
    recorded: set[str] = set()
    tracing = require_json_object(snapshot["tracing"], "snapshot.tracing")
    for module_value in tracing.values():
        module = require_json_object(module_value, "snapshot.tracing.module")
        scopes = module.get("tracer_scopes", [])
        assert isinstance(scopes, list)
        assert all(isinstance(scope, str) for scope in scopes)
        recorded.update(scope for scope in scopes if isinstance(scope, str))
    assert recorded == {"pipecat", "pipecat.turn"}


def test_content_bearing_attribute_keys_are_known(
    snapshot: dict[str, JsonValue],
) -> None:
    """Every content-carrying span attribute Pipecat can emit is on the
    audited list; a new one appearing in an upgrade must be classified."""
    content_keys = {
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
    }
    observed: set[str] = set()
    tracing = require_json_object(snapshot["tracing"], "snapshot.tracing")
    for module_value in tracing.values():
        module = require_json_object(module_value, "snapshot.tracing.module")
        attribute_keys = module.get("attribute_keys", [])
        assert isinstance(attribute_keys, list)
        assert all(isinstance(key, str) for key in attribute_keys)
        observed.update(key for key in attribute_keys if isinstance(key, str))
    assert content_keys <= observed
