"""Validate the generated §9 Grafana provisioning surface."""

from __future__ import annotations

import json
import re
from importlib import import_module
from pathlib import Path
from typing import Protocol, TextIO, cast

from therapy.observability.interactions import JsonValue, require_json_object
from therapy.observability.metrics import INSTRUMENTS, InstrumentKind

ROOT = Path(__file__).resolve().parents[3]
DASHBOARD_DIR = ROOT / "deploy/observability/dashboards"
ALERTS_PATH = (
    ROOT
    / "deploy/observability/grafana/provisioning/alerting/therapy-alerts.yaml"
)
RUNBOOK_DIR = ROOT / "deploy/observability/runbooks"
EXTERNAL_PREFIXES = ("process_", "http_", "scrape_", "stun_", "turn_")
METRIC_RE = re.compile(
    r"\b(?:therapy|process|http|scrape|stun|turn)_[A-Za-z0-9_:]+\b|\bup\b"
)
METRIC_FAMILY_RE = re.compile(r'__name__=~"(?:therapy|stun|turn)_')


class _YamlModule(Protocol):
    def safe_load(self, stream: TextIO) -> object: ...


_yaml_module = import_module("yaml")
if not callable(getattr(_yaml_module, "safe_load", None)):
    raise RuntimeError("PyYAML safe_load is unavailable")
YAML = cast(_YamlModule, _yaml_module)


def _prometheus_instrument_names() -> set[str]:
    """Return names produced by the LGTM OTLP Prometheus translation."""
    names: set[str] = set()
    for spec in INSTRUMENTS:
        if spec.kind is InstrumentKind.HISTOGRAM:
            names.update(f"{spec.name}_{suffix}" for suffix in ("bucket", "sum", "count"))
        elif spec.kind is InstrumentKind.GAUGE and spec.name.endswith("_unixtime"):
            names.add(f"{spec.name}_seconds")
        else:
            names.add(spec.name)
    return names


def _assert_known_metrics(expr: str, *, context: str) -> None:
    """Assert all concrete metric tokens are manifest or approved external names."""
    names = set(METRIC_RE.findall(expr))
    assert names or METRIC_FAMILY_RE.search(expr), (
        f"{context} has no recognizable metric: {expr}"
    )
    known = _prometheus_instrument_names()
    unknown = {
        name
        for name in names
        if name != "up"
        and name not in known
        and not name.startswith(EXTERNAL_PREFIXES)
    }
    assert not unknown, f"{context} references unknown metrics: {sorted(unknown)}"


def _load_mapping(path: Path) -> dict[str, JsonValue]:
    """Load a JSON or YAML document and validate its object boundary."""
    with path.open(encoding="utf-8") as stream:
        payload: object = (
            json.load(stream) if path.suffix == ".json" else YAML.safe_load(stream)
        )
    return require_json_object(payload, str(path))


def test_dashboard_promql_uses_declared_or_external_metrics() -> None:
    dashboard_paths = sorted(DASHBOARD_DIR.glob("*.json"))
    assert len(dashboard_paths) == 6

    sli_panels = 0
    for path in dashboard_paths:
        dashboard = _load_mapping(path)
        templating = dashboard.get("templating")
        assert templating == {"list": []}, f"{path} must not expose variables"
        panels = dashboard.get("panels")
        assert isinstance(panels, list)
        for panel in panels:
            assert isinstance(panel, dict)
            title = panel.get("title")
            assert isinstance(title, str)
            if title.startswith("SLI:"):
                sli_panels += 1
                description = panel.get("description")
                assert isinstance(description, str)
                assert "Definition:" in description
                assert "Target:" in description
            targets = panel.get("targets", [])
            assert isinstance(targets, list)
            for target in targets:
                assert isinstance(target, dict)
                expr = target.get("expr")
                assert isinstance(expr, str)
                assert expr
                _assert_known_metrics(expr, context=f"{path.name}: {title}")

    assert sli_panels >= 8


def test_alert_promql_and_runbook_provisioning() -> None:
    provisioning = _load_mapping(ALERTS_PATH)
    groups = provisioning.get("groups")
    assert isinstance(groups, list)
    assert groups

    alert_count = 0
    alert_uids: set[str] = set()
    for group in groups:
        assert isinstance(group, dict)
        rules = group.get("rules")
        assert isinstance(rules, list)
        assert rules
        group_runbooks: set[str] = set()
        for rule in rules:
            assert isinstance(rule, dict)
            uid = rule.get("uid")
            assert isinstance(uid, str)
            assert uid not in alert_uids
            alert_uids.add(uid)
            title = rule.get("title")
            assert isinstance(title, str)
            annotations = rule.get("annotations")
            assert isinstance(annotations, dict)
            runbook = annotations.get("runbook")
            assert isinstance(runbook, str)
            group_runbooks.add(runbook)
            runbook_path = ROOT / runbook
            assert runbook_path.parent == RUNBOOK_DIR
            assert runbook_path.is_file(), f"{title}: missing {runbook}"

            data = rule.get("data")
            assert isinstance(data, list)
            prometheus_expressions = 0
            for query in data:
                assert isinstance(query, dict)
                if query.get("datasourceUid") != "prometheus":
                    continue
                model = query.get("model")
                assert isinstance(model, dict)
                assert model.get("instant") is True
                assert model.get("range") is False
                expr = model.get("expr")
                assert isinstance(expr, str)
                assert expr
                _assert_known_metrics(expr, context=f"alert: {title}")
                prometheus_expressions += 1
            assert prometheus_expressions > 0
            alert_count += 1
        assert len(group_runbooks) == 1

    assert alert_count >= 20
