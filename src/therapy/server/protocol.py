"""Data-channel message payloads shared by the pipeline and its clients.

Framework-free: agent.py wraps these dicts in transport frames, the PWA
and test clients parse them as JSON. Keeping the shapes here (with tests)
is what keeps the client and server from drifting apart.
"""


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
