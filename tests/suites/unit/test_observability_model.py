"""Contracts for the framework-free observability model (plan O0.1, §5.1)."""

import ast
from pathlib import Path

import pytest

from therapy.observability import model


def test_every_external_facing_enum_has_a_bounded_unknown_member() -> None:
    """External input collapses to one bounded member, never a raw label."""
    for enum in (
        model.Provider,
        model.Outcome,
        model.Component,
        model.WorkloadClass,
        model.LanguageGroup,
        model.Modality,
        model.Destination,
    ):
        assert "unknown" in {member.value for member in enum}, enum


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("anthropic", model.Provider.ANTHROPIC),
        ("ANTHROPIC", model.Provider.ANTHROPIC),
        (" openrouter ", model.Provider.OPENROUTER),
        ("gpt-oss-labs", model.Provider.UNKNOWN),
        (None, model.Provider.UNKNOWN),
        (42, model.Provider.UNKNOWN),
        (model.Provider.OLLAMA, model.Provider.OLLAMA),
    ],
)
def test_normalize_enum_never_leaks_raw_values(
    raw: object, expected: model.Provider
) -> None:
    assert model.normalize_enum(raw, model.Provider, model.Provider.UNKNOWN) is expected


def test_interaction_operations_cover_required_vocabulary() -> None:
    required = {
        "reply",
        "summary",
        "distill",
        "judge",
        "recap",
        "title",
        "research_grounding",
        "tool",
        "evaluation",
    }
    assert required <= {op.value for op in model.InteractionOperation}


def test_status_transitions_are_monotonic() -> None:
    """Terminal states admit no further transitions; no cycles exist."""
    transitions = model.INTERACTION_STATUS_TRANSITIONS
    assert set(transitions) == set(model.InteractionStatus)
    for status in model.TERMINAL_INTERACTION_STATUSES:
        assert transitions[status] == frozenset(), status
    for status, nexts in transitions.items():
        assert status not in nexts, f"{status} may not self-transition"
        for nxt in nexts:
            assert status not in transitions[nxt], f"cycle {status}<->{nxt}"


def test_route_manifest_is_well_formed() -> None:
    routes = model.HTTP_ROUTE_MANIFEST
    assert len(routes) == 54  # O0's 52 + O3.1 /ready + O4.1 client telemetry
    names = [route.name for route in routes]
    assert len(set(names)) == len(names), "duplicate route names"
    pairs = [(route.method, route.path) for route in routes]
    assert len(set(pairs)) == len(pairs), "duplicate method/path pairs"
    for route in routes:
        assert route.method in {"GET", "POST", "PUT", "PATCH", "DELETE"}, route
        assert route.path.startswith("/"), route
    # Liveness and the static shell stay out of broad traces (plan O2.1).
    excluded = {route.name for route in routes if not route.broad_traced}
    assert excluded == {"health", "ready", "index", "client_telemetry"}
    test_only = {route.name for route in routes if route.test_only}
    assert test_only == {"acceptance_agent_turn", "acceptance_proactivity_run"}


def test_llm_boundary_manifest_is_well_formed() -> None:
    boundaries = model.LLM_BOUNDARY_MANIFEST
    names = [boundary.name for boundary in boundaries]
    assert len(set(names)) == len(names), "duplicate boundary names"
    for boundary in boundaries:
        assert boundary.expected_evidence, boundary.name
        if boundary.provider_path is model.ProviderPath.PIPECAT_LLM_SERVICE:
            assert model.InteractionEventKind.STREAM_DELTA in boundary.expected_evidence
        # Every boundary ends in explicit terminal evidence, success or error.
        assert (
            model.InteractionEventKind.TERMINAL_RESPONSE in boundary.expected_evidence
        )
        assert model.InteractionEventKind.TERMINAL_ERROR in boundary.expected_evidence


def test_retrieval_tool_boundary_manifest_is_well_formed() -> None:
    boundaries = model.RETRIEVAL_TOOL_BOUNDARY_MANIFEST
    names = [boundary.name for boundary in boundaries]
    assert len(boundaries) == 5
    assert len(set(names)) == len(names), "duplicate boundary names"
    assert {boundary.kind for boundary in boundaries} == {"retrieval", "tool"}
    assert {boundary.evidence for boundary in boundaries} <= {
        "restricted_capture",
        "product_store",
    }
    assert {boundary.entrypoint for boundary in boundaries} == {
        "ResearchKB.query",
        "ResearchKB.grounding_context",
        "ContextAssembler.assemble",
        "ContextAssembler._episodes",
        "_audit",
    }
    assert all(boundary.notes for boundary in boundaries)
    assert model.retrieval_tool_boundary_manifest_json() == [
        {
            "name": boundary.name,
            "kind": boundary.kind,
            "module": boundary.module,
            "entrypoint": boundary.entrypoint,
            "evidence": boundary.evidence,
            "notes": boundary.notes,
        }
        for boundary in boundaries
    ]


def _entrypoint_is_defined(path: Path, entrypoint: str) -> bool:
    """Resolve a dotted class/function path within one parsed module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    body = tree.body
    for name in entrypoint.split("."):
        match = next(
            (
                node
                for node in body
                if isinstance(
                    node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
                )
                and node.name == name
            ),
            None,
        )
        if match is None:
            return False
        body = match.body
    return True


def test_retrieval_tool_manifest_entrypoints_exist(repo_root: Path) -> None:
    """Every declared boundary resolves to an AST-located implementation."""
    for boundary in model.RETRIEVAL_TOOL_BOUNDARY_MANIFEST:
        parts = boundary.module.split(".")
        assert parts[0] == "therapy", boundary.module
        path = repo_root / "src" / Path(*parts).with_suffix(".py")
        assert path.is_file(), boundary.module
        assert _entrypoint_is_defined(path, boundary.entrypoint), boundary


def test_no_model_invoked_function_calling_tools(repo_root: Path) -> None:
    """Provider call sites expose no unclassified function-calling input."""
    paths = [repo_root / "src/therapy/memory/summarizer.py"]
    paths.extend(sorted((repo_root / "src/therapy/integrations/pipecat").rglob("*.py")))
    violations: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            for keyword in call.keywords:
                if keyword.arg == "tools":
                    violations.append(f"{path}:{call.lineno}: tools keyword")
                if keyword.arg == "json" and any(
                    isinstance(candidate, ast.Dict)
                    and any(
                        isinstance(key, ast.Constant) and key.value == "tools"
                        for key in candidate.keys
                    )
                    for candidate in ast.walk(keyword.value)
                ):
                    violations.append(f"{path}:{call.lineno}: tools JSON field")

    assert violations == []
    tool_boundaries = [
        boundary
        for boundary in model.RETRIEVAL_TOOL_BOUNDARY_MANIFEST
        if boundary.kind == "tool"
    ]
    assert [boundary.entrypoint for boundary in tool_boundaries] == ["_audit"]
    assert "no model runtime invokes them" in tool_boundaries[0].notes


def _call_sites(path: Path, callee: str) -> int:
    """Count calls to `callee` (bare name or attribute) in one module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == callee:
                count += 1
            elif isinstance(func, ast.Attribute) and func.attr == callee:
                count += 1
    return count


def test_llm_boundary_manifest_matches_current_call_sites(repo_root: Path) -> None:
    """Plan O0.1 item 3: the manifest tracks every reference to
    `summarizer.complete()` and `make_llm_service()`; a new call site must be
    classified here before it can ship."""
    completion_modules: dict[str, int] = {}
    for boundary in model.LLM_BOUNDARY_MANIFEST:
        if boundary.provider_path is model.ProviderPath.COMPLETION_CLIENT:
            completion_modules[boundary.module] = (
                completion_modules.get(boundary.module, 0) + 1
            )

    # Every module that calls complete() is declared, with one boundary per
    # call site (the summarizer module hosts the shared wrapper itself).
    source_root = repo_root / "src" / "therapy"
    found: dict[str, int] = {}
    for path in sorted(source_root.rglob("*.py")):
        count = _call_sites(path, "complete")
        if count:
            dotted = (
                path.relative_to(repo_root / "src").with_suffix("").as_posix()
            ).replace("/", ".")
            found[dotted] = count

    assert found == completion_modules, (
        "complete() call sites diverge from LLM_BOUNDARY_MANIFEST.\n"
        f"found={found}\nmanifest={completion_modules}"
    )

    # Realtime boundary: make_llm_service() is constructed exactly once, in
    # the declared pipeline module.
    realtime = [
        boundary
        for boundary in model.LLM_BOUNDARY_MANIFEST
        if boundary.provider_path is model.ProviderPath.PIPECAT_LLM_SERVICE
    ]
    assert len(realtime) == 1
    service_calls: dict[str, int] = {}
    for path in sorted(source_root.rglob("*.py")):
        count = _call_sites(path, "make_llm_service")
        if count:
            dotted = (
                path.relative_to(repo_root / "src").with_suffix("").as_posix()
            ).replace("/", ".")
            service_calls[dotted] = count
    assert service_calls == {realtime[0].module: 1}, service_calls
