"""Data-channel message payloads shared by the pipeline and its clients.

Framework-free: the Pipecat adapter wraps these dicts in transport frames, the PWA
and test clients parse them as JSON. Keeping the shapes here (with tests)
is what keeps the client and server from drifting apart.
"""


# Presence states the *server* can actually witness (phase C). The pipeline
# knows these exactly — VAD opens/closes the user turn and the transport
# reports the bot's audio — so it pushes them as authoritative. offline,
# connecting and mic-off stay client-owned: they are connection- and
# local-mute facts the server never sees.
PRESENCE_STATES = frozenset({"listening", "thinking", "speaking"})


def presence_message(state: str) -> dict:
    """Authoritative companion presence, pushed as the machine changes state.

    The client can *infer* presence (companion.js observers), but the pipeline
    knows it exactly — above all 'thinking', the user-stopped→reply gap the
    client could only guess at. A dropped message self-heals (the next
    transition re-sends) and the client falls back to inference if none ever
    arrive, so this only ever refines the existing behaviour.
    """
    if state not in PRESENCE_STATES:
        raise ValueError(f"Not a server-emittable presence state: {state!r}")
    return {"type": "presence", "state": state}


def session_state_message(
    session_id: str,
    resumed: bool,
    turns: list[dict],
    limit: int = 40,
) -> dict:
    """The server-truth chat state, sent when a client reports ready.

    The client replaces its rendered conversation with these turns, so a
    reconnect (or a page reload mid-session) shows the resumed transcript
    instead of an empty pane. Capped like LLM rehydration — the full
    record stays in the transcript browser.
    """
    return {
        "type": "session",
        "session_id": session_id,
        "resumed": resumed and bool(turns),
        "turns": [
            {
                "role": turn["role"],
                "text": turn["text"],
                "language": turn["language"],
                "modality": turn["modality"],
            }
            for turn in turns[-limit:]
            if turn.get("text")
        ],
    }
