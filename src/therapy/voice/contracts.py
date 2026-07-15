"""Owned signaling values and failures at the voice-framework boundary."""

from dataclasses import dataclass
from typing import Literal, cast

type JsonValue = (
    None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
)
type OfferType = Literal["offer"]
type AnswerType = Literal["answer"]


class VoiceGatewayError(Exception):
    """Base failure raised through the owned voice gateway contract."""


class InvalidOffer(VoiceGatewayError):
    """The client supplied an invalid WebRTC signaling envelope."""


class ConnectionConflict(VoiceGatewayError):
    """The offer conflicts with the active peer or conversation lifecycle."""


class VoiceUnavailable(VoiceGatewayError):
    """The voice runtime cannot safely accept or maintain the connection."""


def _is_json_value(value: object) -> bool:
    """Return whether a value is representable by the owned JSON type."""
    if value is None or isinstance(value, bool | int | float | str):
        return True
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_json_value(item) for key, item in value.items()
        )
    return False


@dataclass(frozen=True, slots=True)
class WebRTCOffer:
    """Framework-independent WebRTC offer accepted by TheraPy's HTTP API."""

    sdp: str
    type: OfferType
    pc_id: str | None = None
    restart_pc: bool = False
    request_data: JsonValue = None

    def __post_init__(self) -> None:
        if not isinstance(self.sdp, str) or not self.sdp:
            raise InvalidOffer("sdp must be a non-empty string")
        if self.type != "offer":
            raise InvalidOffer("type must be 'offer'")
        if self.pc_id is not None and (
            not isinstance(self.pc_id, str) or not self.pc_id or len(self.pc_id) > 256
        ):
            raise InvalidOffer(
                "pc_id must be a non-empty string of at most 256 characters"
            )
        if not isinstance(self.restart_pc, bool):
            raise InvalidOffer("restart_pc must be a boolean")
        if not _is_json_value(self.request_data):
            raise InvalidOffer("request_data must contain only JSON values")

    @classmethod
    def from_payload(cls, payload: object) -> "WebRTCOffer":
        """Validate an untrusted JSON value and build an owned offer."""
        if not isinstance(payload, dict):
            raise InvalidOffer("request body must be a JSON object")
        if not all(isinstance(key, str) for key in payload):
            raise InvalidOffer("offer field names must be strings")
        data = cast(dict[str, object], payload)
        allowed = {"sdp", "type", "pc_id", "restart_pc", "request_data", "requestData"}
        unknown = sorted(key for key in data if key not in allowed)
        if unknown:
            raise InvalidOffer(f"unsupported offer fields: {', '.join(unknown)}")
        if "request_data" in data and "requestData" in data:
            raise InvalidOffer("provide only one of request_data or requestData")

        sdp = data.get("sdp")
        offer_type = data.get("type")
        pc_id = data.get("pc_id")
        restart_pc = data.get("restart_pc", False)
        if not isinstance(sdp, str):
            raise InvalidOffer("sdp must be a string")
        if offer_type != "offer":
            raise InvalidOffer("type must be 'offer'")
        if pc_id is not None and not isinstance(pc_id, str):
            raise InvalidOffer("pc_id must be a string or null")
        if not isinstance(restart_pc, bool):
            raise InvalidOffer("restart_pc must be a boolean")

        request_data = data.get("request_data", data.get("requestData"))
        if not _is_json_value(request_data):
            raise InvalidOffer("request_data must contain only JSON values")
        return cls(
            sdp=sdp,
            type="offer",
            pc_id=pc_id,
            restart_pc=restart_pc,
            request_data=cast(JsonValue, request_data),
        )


@dataclass(frozen=True, slots=True)
class WebRTCAnswer:
    """Framework-independent WebRTC answer returned by the voice gateway."""

    sdp: str
    type: AnswerType
    pc_id: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.sdp, str)
            or not self.sdp
            or self.type != "answer"
            or not isinstance(self.pc_id, str)
            or not self.pc_id
        ):
            raise VoiceUnavailable("voice runtime returned an invalid WebRTC answer")

    def as_payload(self) -> dict[str, str]:
        """Return the stable public signaling response fields."""
        return {"sdp": self.sdp, "type": self.type, "pc_id": self.pc_id}


@dataclass(frozen=True, slots=True)
class SessionTarget:
    """Resolved conversation that a negotiated voice pipeline must join."""

    session_id: str | None
    new_session: bool

    def __post_init__(self) -> None:
        if self.session_id is not None and (
            not isinstance(self.session_id, str) or not self.session_id
        ):
            raise ValueError("session_id must be a non-empty string or None")
        if not isinstance(self.new_session, bool):
            raise ValueError("new_session must be a boolean")
        if self.new_session != (self.session_id is None):
            raise ValueError(
                "new_session must be true exactly when session_id is absent"
            )
