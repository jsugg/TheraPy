"""Executable contracts for launcher and planned session boundaries."""

from __future__ import annotations

import importlib.metadata
import logging
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import Mock, call

import pytest
import uvicorn

from therapy import __version__
from therapy.observability import logging as observability_logging
from therapy.observability import telemetry
from therapy.observability.config import ObservabilityConfig
from therapy.observability.model import CaptureMode
from therapy.perception.emotion import EmotionFrame
from therapy.server import __main__ as server_main
from therapy.session import timeline


def test_server_main_wires_observability_before_uvicorn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ObservabilityConfig(
        capture_mode=CaptureMode.RUNTIME,
        journal_path=Path("/tmp/test-interaction-journal.sqlite3"),
        interaction_backend="journal",
        remote_export_enabled=False,
        retention_days=30,
        queue_size=256,
        group_commit_ms=50,
        otel_enabled=True,
        otlp_broad_endpoint="http://localhost:4318",
        otlp_restricted_endpoint=None,
        otel_export_timeout_secs=3.0,
        environment="test",
        log_level="WARNING",
        client_telemetry_enabled=False,
    )
    from_env = Mock(return_value=config)
    configure_logging = Mock()
    initialize_telemetry = Mock(return_value=True)
    emit_event = Mock()
    def package_version(package: str) -> str:
        return {
            "pipecat-ai": "1.2.3",
            "opentelemetry-semantic-conventions": "4.5.6",
        }[package]

    metadata_version = Mock(side_effect=package_version)
    run = Mock()

    monkeypatch.setattr(ObservabilityConfig, "from_env", from_env)
    monkeypatch.setattr(
        observability_logging,
        "configure_stdout_json_logging",
        configure_logging,
    )
    monkeypatch.setattr(observability_logging, "emit_event", emit_event)
    monkeypatch.setattr(telemetry, "initialize", initialize_telemetry)
    monkeypatch.setattr(importlib.metadata, "version", metadata_version)
    monkeypatch.setattr(uvicorn, "run", run)
    monkeypatch.setenv("THERAPY_HOST", "127.0.0.9")
    monkeypatch.setenv("THERAPY_PORT", "8765")

    assert server_main.main() == 0

    from_env.assert_called_once_with()
    assert metadata_version.call_args_list == [
        call("pipecat-ai"),
        call("opentelemetry-semantic-conventions"),
    ]
    configure_logging.assert_called_once_with(
        level="WARNING",
        service_version=__version__,
        environment="test",
        resource={
            "pipecat.version": "1.2.3",
            "capture.mode": "runtime",
            "capture.backend": "journal",
            "schema.genai": "4.5.6",
            "capture.enabled": "true",
            "config.fingerprint": config.fingerprint(),
        },
    )
    initialize_telemetry.assert_called_once_with(config, service_version=__version__)
    emit_event.assert_called_once_with(
        "app.starting",
        severity=logging.INFO,
        component="server",
        operation="bootstrap",
        outcome="success",
    )
    run.assert_called_once_with(
        "therapy.server.app:app",
        host="127.0.0.9",
        port=8765,
        access_log=False,
        log_config=None,
    )


def test_emotion_frame_is_frozen_with_independent_raw_labels() -> None:
    # Replace when the `ser` boundary lands.
    first = EmotionFrame("turn-1", 0.0, 1.0, "pending")
    second = EmotionFrame("turn-2", 1.0, 2.0, "pending")

    with pytest.raises(FrozenInstanceError):
        first.__setattr__("turn_id", "changed")

    first.raw_labels["calm"] = 0.9
    assert first.raw_labels == {"calm": 0.9}
    assert second.raw_labels == {}
    assert first.raw_labels is not second.raw_labels


def test_timeline_exports_no_public_names() -> None:
    # Replace when the `ser` boundary lands.
    public_names = {name for name in vars(timeline) if not name.startswith("_")}
    assert public_names == set()
