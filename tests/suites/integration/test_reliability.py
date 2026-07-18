"""Reliability regressions: TURN config and deployment wiring."""

import socket
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.type_contracts import HttpTestClient


def test_ice_config_defaults(client: HttpTestClient) -> None:
    response = client.get("/api/ice-config")
    assert response.status_code == 200
    config = response.json()
    assert config["username"] == "therapy"
    assert config["credential"] == "therapy-local"
    assert config["port"] == 3478


def test_ice_config_env_override(
    client: HttpTestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("THERAPY_TURN_PASSWORD", "s3cret")
    monkeypatch.setenv("THERAPY_TURN_PORT", "3479")
    config = client.get("/api/ice-config").json()
    assert config["credential"] == "s3cret"
    assert config["port"] == 3479


def test_turn_probe_reports_unreachable_without_blocking_ice_config(
    client: HttpTestClient,
    monkeypatch: pytest.MonkeyPatch,
    free_port: Callable[[], int],
) -> None:
    port = free_port()
    monkeypatch.setenv("THERAPY_TURN_PROBE_HOST", "127.0.0.1")
    monkeypatch.setenv("THERAPY_TURN_PORT", str(port))

    readiness = client.get("/ready")
    ice_config = client.get("/api/ice-config")

    assert readiness.status_code == 200
    assert readiness.json()["checks"]["turn"] == "unreachable"
    assert "127.0.0.1" not in readiness.text
    assert str(port) not in readiness.text
    assert ice_config.status_code == 200
    assert ice_config.json()["port"] == port


def test_turn_probe_reports_ready_for_listening_socket(
    client: HttpTestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = int(listener.getsockname()[1])
        monkeypatch.setenv("THERAPY_TURN_PROBE_HOST", "127.0.0.1")
        monkeypatch.setenv("THERAPY_TURN_PORT", str(port))

        readiness = client.get("/ready")

    assert readiness.status_code == 200
    assert readiness.json()["checks"]["turn"] == "ready"
    assert "127.0.0.1" not in readiness.text
    assert str(port) not in readiness.text


def test_compose_declares_reliability_and_turn(repo_root: Path) -> None:
    compose = (repo_root / "compose.yaml").read_text()
    # Restart policies + healthcheck keep the stack self-recovering; one
    # policy per service (therapy, turn, and the opt-in phoenix/collector/
    # lgtm observability profiles).
    assert compose.count("restart: unless-stopped") == compose.count("mem_limit:")
    assert compose.count("restart: unless-stopped") == 5
    assert "healthcheck" in compose
    assert "/health" in compose
    assert "stop_grace_period" in compose
    # TURN relay for clients that can't reach container host candidates.
    assert "coturn" in compose
    assert "3478:3478/udp" in compose
    assert "--lt-cred-mech" in compose


def test_turn_has_real_healthcheck_and_metrics_without_username_labels(
    repo_root: Path,
) -> None:
    """O3 audit: TURN needs a behavior healthcheck and a 9641 scrape path,
    and must never emit per-username metric labels."""
    compose = (repo_root / "compose.yaml").read_text()
    assert "turnutils_stunclient" in compose  # STUN round-trip, not port-open
    assert "--prometheus" in compose
    assert "--prometheus-username-labels" not in compose

    collector = (
        repo_root / "deploy/observability/collector.yaml"
    ).read_text()
    assert "prometheus/turn" in collector
    assert "turn:9641" in collector
    assert "{key: username, action: delete}" in collector
    # The scrape must feed the BROAD pipeline (denylist applied).
    import re

    metrics_broad = re.search(
        r"metrics/broad:\n(?:\s+.+\n)+", collector
    )
    assert metrics_broad is not None
    assert "prometheus/turn" in metrics_broad.group(0)
    assert "attributes/broad_denylist" in metrics_broad.group(0)


def test_reliability_dashboard_has_relay_and_supervision_panels(
    repo_root: Path,
) -> None:
    import json

    dashboard = json.loads(
        (repo_root / "deploy/observability/dashboards/reliability.json").read_text()
    )
    titles = [panel["title"] for panel in dashboard["panels"]]
    assert "TURN relay up" in titles
    assert any("STUN bindings" in title for title in titles)
    assert any("Process uptime" in title for title in titles)


def test_compose_caps_memory_per_service(repo_root: Path) -> None:
    # Uncapped containers exhaust the Docker VM under load; a wedged VM
    # hangs the docker CLI and every port-forward (hypervisor stall,
    # observed 2026-07-10). Caps convert that into a container OOM-kill
    # that the restart policy recovers from. One cap per service
    # (therapy, turn, phoenix, collector, lgtm).
    assert (repo_root / "compose.yaml").read_text().count("mem_limit:") == 5


def test_dockerfile_runs_watchdog(repo_root: Path) -> None:
    assert "watchdog.py" in (repo_root / "Dockerfile").read_text()
