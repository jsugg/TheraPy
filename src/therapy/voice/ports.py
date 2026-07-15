"""Application-facing port for voice signaling and runtime lifecycle."""

from typing import Protocol

from therapy.voice.contracts import SessionTarget, WebRTCAnswer, WebRTCOffer


class VoiceGateway(Protocol):
    """Negotiate voice connections without exposing framework-owned objects."""

    async def negotiate(
        self, offer: WebRTCOffer, target: SessionTarget
    ) -> WebRTCAnswer: ...

    async def disconnect(self, peer_id: str) -> bool: ...

    async def close(self) -> None: ...
