"""Installed-Pipecat telemetry surface vs. the committed snapshot (plan O0.1).

Runs only where Pipecat is installed (the therapy container). An upgrade that
changes emitted span attributes, tracer scopes, or MetricsFrame payloads must
fail here before it can silently change what reaches either telemetry plane.
"""

import inspect
import json
from pathlib import Path

import pytest

pipecat = pytest.importorskip("pipecat")

SNAPSHOT = (
    Path(__file__).resolve().parents[2]
    / "fixtures/observability/pipecat/snapshot-1.5.0.json"
)


@pytest.fixture(scope="module")
def snapshot() -> dict:
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


def test_installed_version_matches_snapshot(snapshot: dict) -> None:
    import importlib.metadata as metadata

    assert metadata.version("pipecat-ai") == snapshot["pipecat_version"]


def test_pipeline_task_still_accepts_required_telemetry_parameters(
    snapshot: dict,
) -> None:
    from pipecat.pipeline.task import PipelineTask

    params = set(inspect.signature(PipelineTask.__init__).parameters)
    required = {
        "enable_tracing",
        "enable_turn_tracking",
        "observers",
        "conversation_id",
        "additional_span_attributes",
    }
    assert required <= params
    assert sorted(params - {"self"}) == snapshot["pipeline_task_parameters"]


def test_metrics_data_classes_match_snapshot(snapshot: dict) -> None:
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


def test_tracer_scopes_match_snapshot(snapshot: dict) -> None:
    """The routing processor's audited scope list (`pipecat`, `pipecat.turn`)
    must cover every scope the installed Pipecat requests tracers under."""
    recorded: set[str] = set()
    for module in snapshot["tracing"].values():
        recorded.update(module.get("tracer_scopes", []))
    assert recorded == {"pipecat", "pipecat.turn"}


def test_content_bearing_attribute_keys_are_known(snapshot: dict) -> None:
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
    for module in snapshot["tracing"].values():
        observed.update(module.get("attribute_keys", []))
    assert content_keys <= observed
