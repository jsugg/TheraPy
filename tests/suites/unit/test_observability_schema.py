"""Canonical record schema contract (plan §5.2, O1 test list).

Covers boundary validation, canonical serialization determinism, checksum
stability, stream ordering, and a deterministic fuzz loop over malformed
payloads — arbitrary objects must never slip into the `json-v1` shape.
"""

import json
import random
from dataclasses import replace
from typing import cast

import pytest

from therapy.observability.capture import build_stream_tuple
from therapy.observability.interactions import (
    InteractionRecord,
    InteractionRequest,
    InteractionResponse,
    JsonValue,
    Message,
    ProviderNative,
    StreamEvent,
    canonical_json,
    checksum,
    require_json_object,
)
from therapy.observability.model import (
    InteractionOperation,
    InteractionStatus,
    Provider,
)


def _record(**overrides: object) -> InteractionRecord:
    base = InteractionRecord(
        interaction_id="itx-test-0001",
        trace_id="a" * 32,
        span_id="b" * 16,
        operation=InteractionOperation.SUMMARY,
        provider=Provider.OLLAMA,
        requested_model="m",
        actual_model="m",
        prompt_template_version="v1",
        request=InteractionRequest(
            system_instructions="sys",
            messages=(Message(role="user", content="hello"),),
        ),
        response=InteractionResponse(completion="world"),
        stream=(StreamEvent(sequence=0, observed_at="t0", delta="wor"),
                StreamEvent(sequence=1, observed_at="t1", delta="ld")),
        error=None,
        provider_native=ProviderNative(request={"model": "m"}),
        language="en",
        modality="text",
        build_version="0.1.0",
        policy_version="p1",
        config_version="c1",
        started_at="2026-07-15T00:00:00+00:00",
        completed_at="2026-07-15T00:00:01+00:00",
        status=InteractionStatus.SUCCEEDED,
    )
    return replace(base, **overrides)


def test_build_stream_tuple_accepts_only_text_deltas() -> None:
    stream = build_stream_tuple(
        [
            {
                "observed_at": "2026-07-17T00:00:00+00:00",
                "delta": "hello",
                "tool_delta": '{"name":"lookup"}',
            }
        ]
    )

    assert stream[0].delta == "hello"
    assert stream[0].tool_delta == '{"name":"lookup"}'

    with pytest.raises(TypeError, match="tool delta must be text"):
        build_stream_tuple([{"tool_delta": {"name": "lookup"}}])


def test_to_json_dict_round_trips_and_is_canonical() -> None:
    record = _record()
    payload = record.to_json_dict()
    assert payload["operation"] == "summary"
    assert payload["provider"] == "ollama"
    assert payload["status"] == "succeeded"
    request = require_json_object(payload["request"], "record.request")
    messages = request["messages"]
    assert isinstance(messages, list)
    assert messages[0] == {"role": "user", "content": "hello"}
    # canonical serialization is deterministic and key-sorted
    assert canonical_json(payload) == canonical_json(json.loads(record.canonical()))
    assert record.checksum() == checksum(payload)


def test_checksum_changes_with_content() -> None:
    first = _record().checksum()
    second = _record(response=InteractionResponse(completion="tampered")).checksum()
    assert first != second


def test_malformed_ids_rejected() -> None:
    with pytest.raises(ValueError, match="W3C"):
        _record(trace_id="short")
    with pytest.raises(ValueError, match="W3C"):
        _record(span_id="tiny")
    with pytest.raises(ValueError, match="interaction_id"):
        _record(interaction_id="")


def test_stream_ordering_enforced() -> None:
    events = (
        StreamEvent(sequence=1, observed_at="t", delta="a"),
        StreamEvent(sequence=0, observed_at="t", delta="b"),
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        _record(stream=events)
    duplicated = (
        StreamEvent(sequence=0, observed_at="t", delta="a"),
        StreamEvent(sequence=0, observed_at="t", delta="b"),
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        _record(stream=duplicated)


def test_non_json_values_rejected_at_the_boundary() -> None:
    record = _record(
        provider_native=ProviderNative(
            request={"weird": cast(JsonValue, object())}
        )
    )
    with pytest.raises(TypeError, match="not JSON-serializable"):
        record.to_json_dict()


def test_fuzz_loop_never_accepts_undeclared_shapes() -> None:
    """Deterministic fuzz: random junk inside native envelopes must either
    serialize as pure JSON or raise TypeError — never pass through as-is."""
    rng = random.Random(20260715)
    junk_factory = [
        lambda: object(),
        lambda: {1: "non-string-key"},
        lambda: {"nested": {"deep": object()}},
        lambda: [object()],
        lambda: {"x": bytes(3)},
    ]
    for _ in range(200):
        junk = rng.choice(junk_factory)()
        record = _record(
            provider_native=ProviderNative(
                request={"payload": cast(JsonValue, junk)}
            )
        )
        with pytest.raises(TypeError):
            record.to_json_dict()


def test_provider_native_extra_flattens_into_envelope() -> None:
    record = _record(
        provider_native=ProviderNative(
            request={"model": "m"},
            extra={"generation_id": "gen-1", "fallback_attempts": 2},
        )
    )
    native = require_json_object(
        record.to_json_dict()["provider_native"], "record.provider_native"
    )
    assert native["generation_id"] == "gen-1"
    assert native["fallback_attempts"] == 2
    assert "extra" not in native


def test_non_finite_floats_rejected() -> None:
    """Audit F-16: NaN/Infinity never reach canonical JSON."""
    record = _record(
        provider_native=ProviderNative(request={"score": float("nan")})
    )
    with pytest.raises(TypeError, match="non-finite"):
        record.to_json_dict()


def test_non_hex_ids_rejected() -> None:
    """Audit F-16: W3C IDs must be lowercase hex, not merely length-shaped."""
    with pytest.raises(ValueError, match="hex"):
        _record(trace_id="z" * 32)
    with pytest.raises(ValueError, match="hex"):
        _record(span_id="G" * 16)
