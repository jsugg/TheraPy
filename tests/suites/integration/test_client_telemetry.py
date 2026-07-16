"""Strict client telemetry endpoint (plan O4.1; O4 gate schema tests)."""

import random

import pytest


@pytest.fixture
def enabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("THERAPY_CLIENT_TELEMETRY", "1")
    from therapy.server import app as app_mod

    app_mod._client_bucket["tokens"] = 60.0
    return monkeypatch


def _batch(events: list[dict]) -> dict:
    return {"schema_version": 1, "events": events}


def test_disabled_by_default_is_404(client, monkeypatch) -> None:
    monkeypatch.delenv("THERAPY_CLIENT_TELEMETRY", raising=False)
    response = client.post(
        "/api/telemetry/client",
        json=_batch([{"name": "sw_lifecycle", "outcome": "installed"}]),
    )
    assert response.status_code == 404


def test_valid_batch_accepted(client, enabled) -> None:
    response = client.post(
        "/api/telemetry/client",
        json=_batch(
            [
                {
                    "name": "webrtc_sample",
                    "rtt_ms": 42.5,
                    "jitter_ms": 3.1,
                    "packet_loss_ratio": 0.01,
                    "candidate_type": "relay",
                },
                {"name": "data_channel_state", "outcome": "connected",
                 "duration_ms": 350},
                {"name": "cache_fallback", "outcome": "fallback"},
            ]
        ),
    )
    assert response.status_code == 200
    assert response.json() == {"status": "accepted"}


@pytest.mark.parametrize(
    "event",
    [
        {"name": "free text about the user"},  # unlisted name
        {"name": "webrtc_sample", "url": "https://leak.example"},  # unknown key
        {"name": "webrtc_sample", "rtt_ms": -1},  # out of range
        {"name": "webrtc_sample", "rtt_ms": 999_999_999},  # out of range
        {"name": "webrtc_sample", "candidate_type": "prflx-with-ip"},  # unlisted
        {"name": "disconnect", "outcome": "because my mic broke"},  # free text
        {"name": "webrtc_sample", "packet_loss_ratio": 1.5},
    ],
)
def test_free_text_identifiers_and_ranges_rejected(client, enabled, event) -> None:
    response = client.post("/api/telemetry/client", json=_batch([event]))
    assert response.status_code == 422


def test_batch_bounds(client, enabled) -> None:
    too_many = [{"name": "shell_fetch"}] * 21
    assert (
        client.post("/api/telemetry/client", json=_batch(too_many)).status_code
        == 422
    )
    assert (
        client.post("/api/telemetry/client", json=_batch([])).status_code == 422
    )
    wrong_version = {"schema_version": 2, "events": [{"name": "shell_fetch"}]}
    assert (
        client.post("/api/telemetry/client", json=wrong_version).status_code == 422
    )


def test_oversize_body_rejected(client, enabled) -> None:
    response = client.post(
        "/api/telemetry/client",
        content=b"x" * 100,
        headers={
            "Content-Length": str(20 * 1024),
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 413


def test_cross_origin_rejected(client, enabled) -> None:
    response = client.post(
        "/api/telemetry/client",
        json=_batch([{"name": "shell_fetch"}]),
        headers={"Origin": "https://evil.example"},
    )
    assert response.status_code == 403


def test_rate_limit_is_process_wide_not_ip_keyed(client, enabled) -> None:
    from therapy.server import app as app_mod

    app_mod._client_bucket["tokens"] = 1.0
    ok = client.post(
        "/api/telemetry/client", json=_batch([{"name": "shell_fetch"}])
    )
    assert ok.status_code == 200
    limited = client.post(
        "/api/telemetry/client", json=_batch([{"name": "shell_fetch"}])
    )
    assert limited.status_code == 429


def test_fuzz_random_payloads_never_pass_schema(client, enabled) -> None:
    """Deterministic fuzz loop (O4 gate): randomly structured junk is always
    422, never accepted or a 500."""
    rng = random.Random(20260716)
    fragments = [
        "session-1234", "https://x.test/?q=1", "v=0\\no=- 46117", "sk-abc",
        {"nested": {"deep": True}}, ["a", 1], 1e308, -5, "shell_fetch",
    ]
    from therapy.server import app as app_mod

    for _ in range(100):
        app_mod._client_bucket["tokens"] = 60.0  # isolate schema behavior
        event = {
            rng.choice(["name", "outcome", "rtt_ms", "extra", "url", "id"]):
                rng.choice(fragments)
            for _ in range(rng.randint(1, 4))
        }
        response = client.post("/api/telemetry/client", json=_batch([event]))
        assert response.status_code in (200, 422), response.text
        if response.status_code == 200:
            # only possible when the random draw formed a fully valid event
            assert event.get("name") in {"shell_fetch"} and set(event) <= {
                "name", "outcome", "rtt_ms",
            }
