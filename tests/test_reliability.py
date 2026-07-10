"""Reliability regressions: TURN config, connection preemption, compose wiring."""

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from therapy.server.app import app, launch_bot

client = TestClient(app)
COMPOSE = (Path(__file__).parents[1] / "compose.yaml").read_text()
DOCKERFILE = (Path(__file__).parents[1] / "Dockerfile").read_text()


def test_ice_config_defaults() -> None:
    response = client.get("/api/ice-config")
    assert response.status_code == 200
    config = response.json()
    assert config["username"] == "therapy"
    assert config["credential"] == "therapy-local"
    assert config["port"] == 3478


def test_ice_config_env_override(monkeypatch) -> None:
    monkeypatch.setenv("THERAPY_TURN_PASSWORD", "s3cret")
    monkeypatch.setenv("THERAPY_TURN_PORT", "3479")
    config = client.get("/api/ice-config").json()
    assert config["credential"] == "s3cret"
    assert config["port"] == 3479


def test_new_connection_preempts_previous_pipeline() -> None:
    async def scenario() -> tuple[bool, bool]:
        started = asyncio.Event()

        async def bot(connection: object) -> None:
            started.set()
            await asyncio.sleep(30)

        first = launch_bot("conn-1", bot)
        await started.wait()
        second = launch_bot("conn-2", bot)
        await asyncio.sleep(0.05)
        first_cancelled = first.cancelled()
        second.cancel()
        return first_cancelled, second.cancelled() or not second.done()

    first_cancelled, second_alive_then_cancelled = asyncio.run(scenario())
    assert first_cancelled  # single-user: newest connection wins (memory!)
    assert second_alive_then_cancelled


def test_compose_declares_reliability_and_turn() -> None:
    # Restart policies + healthcheck keep the stack self-recovering.
    assert COMPOSE.count("restart: unless-stopped") == 2
    assert "healthcheck" in COMPOSE and "/health" in COMPOSE
    assert "stop_grace_period" in COMPOSE
    # TURN relay for clients that can't reach container host candidates.
    assert "coturn" in COMPOSE
    assert "3478:3478/udp" in COMPOSE
    assert "--lt-cred-mech" in COMPOSE


def test_dockerfile_runs_watchdog() -> None:
    assert "watchdog.py" in DOCKERFILE
