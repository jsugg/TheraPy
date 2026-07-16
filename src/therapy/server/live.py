"""Live-pipeline session ownership (framework-free runtime state).

A reconnect can resume a session while the previous pipeline's
finalization is still pending — ownership tokens make stale finalizers
no-ops, and let the review API refuse to delete a session a pipeline is
actively writing to. Single event loop: plain dicts, no locking.
"""

_owners: dict[str, object] = {}


def _record(operation: str, outcome: str) -> None:
    from therapy.observability.telemetry import record_metric

    record_metric(
        "therapy_voice_pipeline_transitions_total",
        1,
        {"transition": operation, "outcome": outcome},
    )
    record_metric("therapy_voice_active_connections", len(_owners))


def claim(session_id: str) -> object:
    """Register the calling pipeline as the session's current owner."""
    token = object()
    preempted = session_id in _owners
    _owners[session_id] = token
    _record("claim", "preemption" if preempted else "success")
    return token


def owns(session_id: str, token: object) -> bool:
    """Whether this token is still the session's current owner."""
    return _owners.get(session_id) is token


def release(session_id: str, token: object) -> None:
    """Drop ownership — only if this token still holds it."""
    if _owners.get(session_id) is token:
        del _owners[session_id]
        _record("release", "success")
    else:
        _record("release", "mismatch")


def is_active(session_id: str) -> bool:
    """Whether any live pipeline currently owns this session."""
    return session_id in _owners
