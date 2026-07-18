"""Crisis-config validation table (P1).

`crisis_contacts()` is the surface an owner uses to localize crisis hotlines
via `THERAPY_CRISIS_CONTACTS`. Every malformed shape must raise a specific
`CrisisConfigurationError`, and the independent response path
(`crisis_resources()`) must degrade to the safe default without raising — so a
mangled config can never crash the safety surface nor silently drop contacts.
"""

from __future__ import annotations

import json
import types

import pytest

from therapy.dialogue import policy
from therapy.dialogue.policy import (
    CRISIS_RESOURCES_DEFAULT,
    CrisisConfigurationError,
    build_system_prompt,
    continuity_note,
    crisis_contacts,
    crisis_resources,
    graph_continuity_note,
)
from therapy.knowledge.user_model import GraphContext

# Oversized raw config (> 10 KB) is rejected before JSON parsing (L49).
_OVERSIZE = '["' + "a" * 10_001 + '"]'
# 21 otherwise-valid entries trips the >20 cap (L62).
_TWENTY_ONE = json.dumps([{"label": f"L{i}", "value": str(i)} for i in range(21)])
# A control character inside an otherwise-valid label (L95).
_CONTROL = json.dumps([{"label": "a\x01b", "value": "y"}])

# One malformed shape per row: (case id, raw THERAPY_CRISIS_CONTACTS value,
# optional json.loads() override, expected CrisisConfigurationError substring).
# The override exists only for shapes JSON itself cannot express — a non-string
# key can arise solely from a parser that yields one, so it is injected there.
type _MalformedCase = tuple[str, str, object, str]

_MALFORMED: list[_MalformedCase] = [
    ("not-valid-json", "not-json", None, "invalid JSON"),
    ("not-a-json-array", '{"label": "x", "value": "y"}', None, "must be a JSON array"),
    ("over-20-entries", _TWENTY_ONE, None, "at most 20 entries"),
    ("entry-not-an-object", "[1, 2]", None, "only label and value"),
    ("non-string-keys", "[]", [{1: "x"}], "only label and value"),
    ("non-string-label-value", '[{"label": 1, "value": "y"}]', None, "must be strings"),
    ("empty-label", '[{"label": "", "value": "y"}]', None, "length is invalid"),
    ("over-long-label", '[{"label": "' + "x" * 101 + '", "value": "y"}]', None, "length is invalid"),
    ("control-characters", _CONTROL, None, "control characters"),
    ("extra-keys", '[{"label": "a", "value": "b", "extra": "c"}]', None, "only label and value"),
    ("missing-value", '[{"label": "Only label"}]', None, "only label and value"),
    ("oversize-over-10kb", _OVERSIZE, None, "exceeds 10 KB"),
]

# argnames stripped of the id column; ids kept in lock-step from the same rows.
_MALFORMED_ARGS: list[tuple[str, object, str]] = [
    (raw, loads_return, match) for _id, raw, loads_return, match in _MALFORMED
]
_MALFORMED_IDS: list[str] = [case[0] for case in _MALFORMED]


def _override_loads(monkeypatch: pytest.MonkeyPatch, value: object) -> None:
    """Replace only *policy*'s `json` view so `json.loads` yields `value`.

    Swapping the module reference (not the shared `json` module) keeps the
    override scoped to `crisis_contacts` and auto-restored by monkeypatch.
    """

    def _fake_loads(_raw: str) -> object:
        return value

    monkeypatch.setattr(
        policy,
        "json",
        types.SimpleNamespace(loads=_fake_loads, JSONDecodeError=json.JSONDecodeError),
    )


@pytest.mark.parametrize(("raw", "loads_return", "match"), _MALFORMED_ARGS, ids=_MALFORMED_IDS)
def test_malformed_crisis_contacts_raise_typed_error(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    loads_return: object,
    match: str,
) -> None:
    monkeypatch.setenv("THERAPY_CRISIS_CONTACTS", raw)
    if loads_return is not None:
        _override_loads(monkeypatch, loads_return)
    with pytest.raises(CrisisConfigurationError, match=match):
        crisis_contacts()


@pytest.mark.parametrize(("raw", "loads_return", "match"), _MALFORMED_ARGS, ids=_MALFORMED_IDS)
def test_malformed_crisis_config_degrades_to_default(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
    loads_return: object,
    match: str,  # noqa: ARG001 — same table as the raise test; the safe path ignores the message
) -> None:
    monkeypatch.setenv("THERAPY_CRISIS_CONTACTS", raw)
    monkeypatch.delenv("THERAPY_CRISIS_RESOURCES", raising=False)
    if loads_return is not None:
        _override_loads(monkeypatch, loads_return)
    # The response path swallows CrisisConfigurationError and never raises.
    assert crisis_resources() == CRISIS_RESOURCES_DEFAULT
    assert "emergency services" in build_system_prompt()


def test_valid_multi_contact_renders_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "THERAPY_CRISIS_CONTACTS",
        '[{"label": "Línea 135", "value": "135"}, {"label": "SAMU", "value": "192"}]',
    )
    assert crisis_contacts() == [
        {"label": "Línea 135", "value": "135"},
        {"label": "SAMU", "value": "192"},
    ]
    assert crisis_resources() == "Línea 135: 135; SAMU: 192"
    assert "Línea 135: 135; SAMU: 192" in build_system_prompt()


def test_duplicate_contacts_are_deduped_case_insensitively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "THERAPY_CRISIS_CONTACTS",
        '[{"label": "Aid", "value": "1"}, {"label": "aid", "value": "1"}]',
    )
    assert crisis_contacts() == [{"label": "Aid", "value": "1"}]


def test_legacy_resources_over_2000_chars_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("THERAPY_CRISIS_CONTACTS", raising=False)
    monkeypatch.setenv("THERAPY_CRISIS_RESOURCES", "x" * 2_001)
    assert crisis_resources() == CRISIS_RESOURCES_DEFAULT


def test_legacy_resources_within_limit_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("THERAPY_CRISIS_CONTACTS", raising=False)
    monkeypatch.setenv("THERAPY_CRISIS_RESOURCES", "Línea 135")
    assert crisis_resources() == "Línea 135"


def test_build_system_prompt_reraises_when_resource_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # crisis_resources() never raises in practice, but the prompt builder's
    # fallback-then-reraise guard must hold if it ever does.
    def _boom() -> str:
        raise RuntimeError("resource lookup failed")

    monkeypatch.setattr(policy, "crisis_resources", _boom)
    with pytest.raises(RuntimeError, match="resource lookup failed"):
        build_system_prompt()


def test_continuity_note_rejects_non_text_statement() -> None:
    with pytest.raises(TypeError, match="must be text"):
        continuity_note([], [{"statement": 123}])


class _EmptyGraphModel:
    """Graph provider whose walk yields nothing to inject."""

    def assemble_context(self, topic: str = "", *, k: int = 8) -> GraphContext:
        return GraphContext(
            identity=[],
            preferences=[],
            goals=[],
            threads=[],
            never_initiate=[],
            walk_nodes=[],
            walk_edges=[],
        )


def test_graph_continuity_note_appends_summaries_when_graph_is_empty() -> None:
    note = graph_continuity_note(
        _EmptyGraphModel(),
        summaries=[{"started_at": "2026-07-09T20:00:00+00:00", "summary": "Talked about work."}],
    )
    assert note is not None
    assert "# Previous conversations" in note
    assert "2026-07-09" in note
    assert "Talked about work." in note


def test_graph_continuity_note_returns_none_when_nothing_to_inject() -> None:
    assert graph_continuity_note(_EmptyGraphModel()) is None
