"""Frozen instrument manifest and cardinality rules (plan §8, O2 gate)."""

from therapy.observability.metrics import INSTRUMENT_INDEX, INSTRUMENTS


def test_names_are_unique_and_stable() -> None:
    assert len(INSTRUMENT_INDEX) == len(INSTRUMENTS)
    for spec in INSTRUMENTS:
        assert spec.name.startswith("therapy_"), spec.name
        assert spec.name == spec.name.lower()


def test_enumerated_attribute_sets_stay_bounded() -> None:
    """Explicit value sets stay under ten values (plan §8); None means the
    values come from a bounded enum in model.py."""
    for spec in INSTRUMENTS:
        for attr, values in spec.attributes.items():
            assert attr == attr.lower()
            if values is not None:
                assert 0 < len(values) < 10, f"{spec.name}.{attr}"


def test_no_forbidden_label_dimensions() -> None:
    """Labels never carry IDs, models under dynamic routing, URLs, paths,
    timestamps, or content (plan §8)."""
    forbidden = {
        "session_id", "turn_id", "interaction_id", "job_id", "document_id",
        "model", "actual_model", "url", "path", "endpoint", "timestamp",
        "message", "text", "error_message",
    }
    for spec in INSTRUMENTS:
        overlap = forbidden & set(spec.attributes)
        assert not overlap, f"{spec.name}: {overlap}"


def test_record_metric_drops_undeclared_attributes(monkeypatch) -> None:
    from therapy.observability import telemetry

    captured: dict[str, dict] = {}

    class FakeCounter:
        def add(self, value, attributes):
            captured["attrs"] = attributes

    monkeypatch.setattr(
        telemetry.state(), "instruments", {"therapy_llm_requests_total": FakeCounter()}
    )
    telemetry.record_metric(
        "therapy_llm_requests_total",
        1,
        {
            "provider": "ollama",
            "operation": "summary",
            "outcome": "success",
            "session_id": "sess-leak",  # undeclared: must be dropped
        },
    )
    assert captured["attrs"] == {
        "provider": "ollama",
        "operation": "summary",
        "outcome": "success",
    }

    # enumerated sets normalize unknown values instead of minting labels
    class FakeCounter2(FakeCounter):
        pass

    monkeypatch.setattr(
        telemetry.state(), "instruments", {"therapy_llm_output_total": FakeCounter2()}
    )
    telemetry.record_metric(
        "therapy_llm_output_total",
        1,
        {"provider": "ollama", "operation": "summary", "result": "made-up"},
    )
    assert captured["attrs"]["result"] == "unknown"

    # unknown instruments are ignored, never created ad hoc
    telemetry.record_metric("therapy_not_in_manifest", 1, {})
