"""PlaneRoutingSpanProcessor behavior with real-ish spans (plan §5.5, O2 gate).

Unknown scopes are dropped (never fanned out); broad spans are scrubbed of
denylisted/content keys; restricted spans pass intact; shutdown/flush stay
bounded and never raise.
"""

import time
from types import SimpleNamespace

from therapy.observability.telemetry import PlaneRoutingSpanProcessor, state


class RecordingProcessor:
    def __init__(self) -> None:
        self.ended: list[SimpleNamespace] = []

    def on_start(
        self, span: SimpleNamespace, parent_context: object | None = None
    ) -> None:
        pass

    def on_end(self, span: SimpleNamespace) -> None:
        self.ended.append(span)

    def shutdown(self) -> None:
        self.was_shutdown = True

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def _span(scope_name: str | None, attributes: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        instrumentation_scope=(
            SimpleNamespace(name=scope_name) if scope_name else None
        ),
        attributes=attributes,
        status=None,
    )


def test_restricted_scope_routes_only_restricted() -> None:
    broad, restricted = RecordingProcessor(), RecordingProcessor()
    router = PlaneRoutingSpanProcessor(broad=broad, restricted=restricted)
    span = _span("pipecat.turn", {"transcript": "exact words"})
    router.on_end(span)
    assert restricted.ended == [span]
    assert broad.ended == []
    # content attributes survive on the restricted path
    assert restricted.ended[0].attributes["transcript"] == "exact words"


def test_broad_scope_is_scrubbed() -> None:
    broad, restricted = RecordingProcessor(), RecordingProcessor()
    router = PlaneRoutingSpanProcessor(broad=broad, restricted=restricted)
    router.on_end(
        _span(
            "opentelemetry.instrumentation.fastapi",
            {
                "http.route": "/api/sessions/{session_id}",
                "url.query": "token=abc",
                "transcript": "should never be here",
            },
        )
    )
    assert restricted.ended == []
    assert len(broad.ended) == 1
    attrs = dict(broad.ended[0].attributes)
    assert attrs == {"http.route": "/api/sessions/{session_id}"}


def test_unknown_scope_dropped_and_counted() -> None:
    broad, restricted = RecordingProcessor(), RecordingProcessor()
    router = PlaneRoutingSpanProcessor(broad=broad, restricted=restricted)
    before = state().dropped_unknown_scopes
    router.on_end(_span("random.vendor.sdk", {"anything": "x"}))
    router.on_end(_span(None, {}))
    assert broad.ended == []
    assert restricted.ended == []
    assert state().dropped_unknown_scopes == before + 2


def test_shutdown_and_flush_are_bounded() -> None:
    broad, restricted = RecordingProcessor(), RecordingProcessor()
    router = PlaneRoutingSpanProcessor(broad=broad, restricted=restricted)
    started = time.monotonic()
    assert router.force_flush(1000) is True
    router.shutdown()
    assert time.monotonic() - started < 1.0
    assert broad.was_shutdown
    assert restricted.was_shutdown
