"""Snapshot Pipecat's installed tracing/metrics surface (plan O0.1 item 4).

Runs inside the therapy container (the only environment with Pipecat
installed) and prints a sanitized JSON snapshot of:

- the installed Pipecat version;
- every tracing span-attribute builder and the exact attribute keys it can
  emit (parsed from the installed source, not from documentation);
- the instrumentation scope names Pipecat requests tracers under;
- every `MetricsData` payload class and its fields;
- the `PipelineWorker`/`PipelineParams` telemetry-relevant signature.

The output is committed as
`tests/fixtures/observability/pipecat/snapshot-<version>.json` and diffed by
tests so an upgrade that changes emitted telemetry fails loudly instead of
silently leaking new content attributes.

Usage:
    docker compose exec -T therapy uv run python \
        scripts/observability/snapshot_pipecat.py
"""

from __future__ import annotations

import ast
import inspect
import json
import re
import sys


def _attribute_keys_from_source(source: str) -> list[str]:
    """Every literal span-attribute key set in a module's source."""
    keys: set[str] = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        # attributes["key"] = ... / attributes.update({"key": ...})
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if isinstance(node.slice.value, str):
                keys.add(node.slice.value)
        elif isinstance(node, ast.Dict):
            for key in node.keys:
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    keys.add(key.value)
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "set_attribute":
                if node.args and isinstance(node.args[0], ast.Constant):
                    if isinstance(node.args[0].value, str):
                        keys.add(node.args[0].value)
    # Only keep plausible attribute keys (dotted or snake_case identifiers).
    return sorted(
        k for k in keys if re.fullmatch(r"[a-z][a-z0-9_.]*[a-z0-9]", k) and len(k) > 2
    )


def _span_names_from_source(source: str) -> list[str]:
    """Literal span names passed to start_span/start_as_current_span."""
    names: set[str] = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            attr = getattr(func, "attr", None)
            if attr in {"start_span", "start_as_current_span"}:
                if node.args and isinstance(node.args[0], ast.Constant):
                    if isinstance(node.args[0].value, str):
                        names.add(node.args[0].value)
    return sorted(names)


def _tracer_scopes_from_source(source: str) -> list[str]:
    """Literal instrumentation scope names passed to get_tracer()."""
    scopes: set[str] = set()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if getattr(func, "attr", None) == "get_tracer" or (
                isinstance(func, ast.Name) and func.id == "get_tracer"
            ):
                if node.args and isinstance(node.args[0], ast.Constant):
                    if isinstance(node.args[0].value, str):
                        scopes.add(node.args[0].value)
    return sorted(scopes)


def main() -> int:
    import importlib.metadata as metadata

    import pipecat.metrics.metrics as metrics_module
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker

    snapshot: dict[str, object] = {
        "pipecat_version": metadata.version("pipecat-ai"),
        "python_version": ".".join(map(str, sys.version_info[:3])),
    }

    tracing_modules = {}
    for name in (
        "pipecat.utils.tracing.service_attributes",
        "pipecat.utils.tracing.service_decorators",
        "pipecat.utils.tracing.setup",
        "pipecat.utils.tracing.tracing_context",
        "pipecat.utils.tracing.turn_trace_observer",
    ):
        try:
            module = __import__(name, fromlist=["_"])
        except Exception as exc:  # pragma: no cover - container only
            tracing_modules[name] = {"import_error": type(exc).__name__}
            continue
        source = inspect.getsource(module)
        tracing_modules[name] = {
            "attribute_keys": _attribute_keys_from_source(source),
            "span_names": _span_names_from_source(source),
            "tracer_scopes": _tracer_scopes_from_source(source),
            "functions": sorted(
                fn
                for fn, obj in inspect.getmembers(module, inspect.isfunction)
                if obj.__module__ == name
            ),
        }
    snapshot["tracing"] = tracing_modules

    metrics_classes = {}
    for cls_name, cls in sorted(
        inspect.getmembers(metrics_module, inspect.isclass), key=lambda kv: kv[0]
    ):
        if cls.__module__ != metrics_module.__name__:
            continue
        fields = getattr(cls, "model_fields", None)
        metrics_classes[cls_name] = (
            sorted(fields) if fields else sorted(vars(cls).get("__annotations__", {}))
        )
    snapshot["metrics_data_classes"] = metrics_classes

    worker_initializer = vars(PipelineWorker).get("__init__")
    if not callable(worker_initializer):
        raise RuntimeError("PipelineWorker.__init__ is unavailable")
    task_params = inspect.signature(worker_initializer).parameters
    snapshot["pipeline_worker_parameters"] = sorted(
        name for name in task_params if name != "self"
    )
    snapshot["pipeline_params_fields"] = sorted(PipelineParams.model_fields)

    json.dump(snapshot, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
