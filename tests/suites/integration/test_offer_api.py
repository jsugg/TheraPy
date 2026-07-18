"""Framework-free HTTP contract tests for WebRTC offer negotiation."""

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from tests.type_contracts import HttpTestClient
from therapy.memory import MemoryStore
from therapy.server import app as app_module
from therapy.voice.contracts import (
    ConnectionConflict,
    SessionTarget,
    VoiceUnavailable,
    WebRTCAnswer,
    WebRTCOffer,
)


@dataclass
class FakeVoiceGateway:
    """Record owned gateway calls without importing the Pipecat integration."""

    answer: WebRTCAnswer = WebRTCAnswer(sdp="answer-sdp", type="answer", pc_id="peer-1")
    error: Exception | None = None
    calls: list[tuple[WebRTCOffer, SessionTarget]] = field(
        default_factory=lambda: []
    )
    disconnected: list[str] = field(default_factory=lambda: [])
    closed: bool = False

    async def negotiate(
        self, offer: WebRTCOffer, target: SessionTarget
    ) -> WebRTCAnswer:
        self.calls.append((offer, target))
        if self.error is not None:
            raise self.error
        return self.answer

    async def close(self) -> None:
        self.closed = True

    async def disconnect(self, peer_id: str) -> bool:
        self.disconnected.append(peer_id)
        return True


@pytest.fixture
def gateway_client(
    client: HttpTestClient,
) -> Iterator[tuple[HttpTestClient, FakeVoiceGateway]]:
    """Inject an owned fake gateway into the live FastAPI test application."""
    gateway = FakeVoiceGateway()
    app_module.app.dependency_overrides[app_module.get_voice_gateway] = lambda: gateway
    try:
        yield client, gateway
    finally:
        app_module.app.dependency_overrides.clear()


def test_offer_new_session_preserves_response_shape(
    gateway_client: tuple[HttpTestClient, FakeVoiceGateway],
) -> None:
    client, gateway = gateway_client

    response = client.post(
        "/api/offer?new_session=1", json={"sdp": "offer-sdp", "type": "offer"}
    )

    assert response.status_code == 200
    assert response.json() == {
        "sdp": "answer-sdp",
        "type": "answer",
        "pc_id": "peer-1",
        "session_id": None,
        "resumed": False,
        "turns": [],
    }
    assert gateway.calls == [
        (
            WebRTCOffer(sdp="offer-sdp", type="offer"),
            SessionTarget(session_id=None, new_session=True),
        )
    ]


def test_disconnect_voice_uses_owned_gateway(
    gateway_client: tuple[HttpTestClient, FakeVoiceGateway],
) -> None:
    client, gateway = gateway_client

    response = client.post("/api/voice/disconnect", params={"pc_id": "peer-1"})

    assert response.status_code == 200
    assert response.json() == {"disconnected": True}
    assert gateway.disconnected == ["peer-1"]


def test_offer_automatically_resumes_and_enriches_transcript_in_order(
    data_dir: Path, gateway_client: tuple[HttpTestClient, FakeVoiceGateway]
) -> None:
    client, gateway = gateway_client
    store = MemoryStore(data_dir)
    session_id = store.create_session()
    store.add_turn(session_id, "user", "text", "es", "Primero")
    store.add_turn(session_id, "assistant", "text", "es", "Segundo")
    store.end_session(session_id, "Resumen")

    response = client.post("/api/offer", json={"sdp": "offer-sdp", "type": "offer"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == session_id
    assert payload["resumed"] is True
    assert [turn["text"] for turn in payload["turns"]] == ["Primero", "Segundo"]
    assert gateway.calls[0][1] == SessionTarget(
        session_id=session_id, new_session=False
    )


def test_offer_explicit_session_wins_over_automatic_resume(
    data_dir: Path, gateway_client: tuple[HttpTestClient, FakeVoiceGateway]
) -> None:
    client, gateway = gateway_client
    store = MemoryStore(data_dir)
    explicit = store.create_session()
    store.add_turn(explicit, "user", "text", "pt", "Escolhida")
    store.end_session(explicit, "Resumo")
    newest = store.create_session()
    store.add_turn(newest, "user", "text", "en", "Newer")
    store.end_session(newest, "Summary")

    response = client.post(
        f"/api/offer?session={explicit}",
        json={"sdp": "offer-sdp", "type": "offer"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["session_id"] == explicit
    assert payload["resumed"] is True
    assert [turn["text"] for turn in payload["turns"]] == ["Escolhida"]
    assert gateway.calls[0][1] == SessionTarget(session_id=explicit, new_session=False)


def test_offer_unknown_explicit_session_opens_fresh_session(
    gateway_client: tuple[HttpTestClient, FakeVoiceGateway],
) -> None:
    client, gateway = gateway_client

    response = client.post(
        "/api/offer?session=unknown",
        json={"sdp": "offer-sdp", "type": "offer"},
    )

    assert response.status_code == 200
    assert response.json()["session_id"] is None
    assert response.json()["resumed"] is False
    assert response.json()["turns"] == []
    assert gateway.calls[0][1] == SessionTarget(session_id=None, new_session=True)


@pytest.mark.parametrize(
    ("content", "content_type", "status"),
    [
        (b"not-json", "application/json", 400),
        (b"[]", "application/json", 422),
        (b'{"type":"offer"}', "application/json", 422),
        (b'{"sdp":"x","type":"answer"}', "application/json", 422),
        (b'{"sdp":1,"type":"offer"}', "application/json", 422),
        (b'{"sdp":"x","type":"offer","unexpected":true}', "application/json", 422),
    ],
)
def test_offer_rejects_invalid_envelopes(
    gateway_client: tuple[HttpTestClient, FakeVoiceGateway],
    content: bytes,
    content_type: str,
    status: int,
) -> None:
    client, gateway = gateway_client

    response = client.post(
        "/api/offer", content=content, headers={"Content-Type": content_type}
    )

    assert response.status_code == status
    assert gateway.calls == []


def test_offer_rejects_oversized_request_before_parsing(
    gateway_client: tuple[HttpTestClient, FakeVoiceGateway],
) -> None:
    client, gateway = gateway_client
    oversized = b"{" + b'"sdp":"' + b"x" * (256 * 1024) + b'","type":"offer"}'

    response = client.post(
        "/api/offer",
        content=oversized,
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 413
    assert gateway.calls == []


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (ConnectionConflict("peer conflict"), 409),
        (VoiceUnavailable("voice unavailable"), 503),
    ],
)
def test_offer_maps_owned_gateway_errors_to_stable_responses(
    gateway_client: tuple[HttpTestClient, FakeVoiceGateway],
    error: Exception,
    status: int,
) -> None:
    client, gateway = gateway_client
    gateway.error = error

    response = client.post("/api/offer", json={"sdp": "offer-sdp", "type": "offer"})

    assert response.status_code == status
    assert response.json() == {"detail": str(error)}
