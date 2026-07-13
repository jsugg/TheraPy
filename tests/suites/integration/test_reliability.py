"""Reliability regressions: TURN config, connection preemption, compose wiring."""

import asyncio

from therapy.server.app import launch_bot


def test_ice_config_defaults(client) -> None:
    response = client.get("/api/ice-config")
    assert response.status_code == 200
    config = response.json()
    assert config["username"] == "therapy"
    assert config["credential"] == "therapy-local"
    assert config["port"] == 3478


def test_ice_config_env_override(client, monkeypatch) -> None:
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


def test_compose_declares_reliability_and_turn(repo_root) -> None:
    compose = (repo_root / "compose.yaml").read_text()
    # Restart policies + healthcheck keep the stack self-recovering.
    assert compose.count("restart: unless-stopped") == 2
    assert "healthcheck" in compose
    assert "/health" in compose
    assert "stop_grace_period" in compose
    # TURN relay for clients that can't reach container host candidates.
    assert "coturn" in compose
    assert "3478:3478/udp" in compose
    assert "--lt-cred-mech" in compose


def test_compose_caps_memory_per_service(repo_root) -> None:
    # Uncapped containers exhaust the Docker VM under load; a wedged VM
    # hangs the docker CLI and every port-forward (hypervisor stall,
    # observed 2026-07-10). Caps convert that into a container OOM-kill
    # that the restart policy recovers from. One cap per service.
    assert (repo_root / "compose.yaml").read_text().count("mem_limit:") == 2


def test_dockerfile_runs_watchdog(repo_root) -> None:
    assert "watchdog.py" in (repo_root / "Dockerfile").read_text()
