"""Data-channel payload shapes shared with the PWA (server/protocol.py)."""

from therapy.server import live
from therapy.server.protocol import session_state_message


def _turn(i: int, text: str | None = None) -> dict:
    return {
        "role": "user" if i % 2 == 0 else "assistant",
        "text": f"t{i}" if text is None else text,
        "language": "es",
        "modality": "text",
    }


def test_session_state_carries_turns_verbatim() -> None:
    message = session_state_message("abc123", True, [_turn(0), _turn(1)])
    assert message["type"] == "session"
    assert message["session_id"] == "abc123"
    assert message["resumed"] is True
    assert message["turns"] == [
        {"role": "user", "text": "t0", "language": "es", "modality": "text"},
        {"role": "assistant", "text": "t1", "language": "es", "modality": "text"},
    ]


def test_session_state_not_resumed_without_turns() -> None:
    # A fresh session must not tell the client to clear-and-render nothing
    # as a "resume" — resumed implies there is a transcript to show.
    message = session_state_message("abc123", True, [])
    assert message["resumed"] is False
    assert message["turns"] == []


def test_session_state_caps_to_most_recent_and_drops_empty() -> None:
    turns = [_turn(i) for i in range(50)] + [_turn(50, text="")]
    message = session_state_message("abc123", True, turns, limit=40)
    # Slice first (bounds the payload), then drop empties — same semantics
    # as policy.rehydrate_messages: last 40 of 51, minus the empty one.
    assert len(message["turns"]) == 39
    assert message["turns"][0]["text"] == "t11"
    assert message["turns"][-1]["text"] == "t49"


def test_live_ownership_tokens() -> None:
    first = live.claim("sess")
    assert live.is_active("sess")
    assert live.owns("sess", first)

    second = live.claim("sess")  # reconnect takes over
    assert not live.owns("sess", first)
    assert live.owns("sess", second)

    live.release("sess", first)  # stale release must not evict the new owner
    assert live.is_active("sess")
    live.release("sess", second)
    assert not live.is_active("sess")
