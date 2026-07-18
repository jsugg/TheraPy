import asyncio
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

# The Pipecat integration is optional on framework-free development hosts;
# on hosts without the realtime stack (e.g. the Intel-Mac dev box, where
# onnxruntime/kokoro have no wheels) collection would fail. Skip cleanly there —
# the container test bed has the full stack.
pytest.importorskip("pipecat")

from therapy.integrations.pipecat.pipeline import (
    LANGUAGE_ENUM,
    generate_session_artifacts,
    make_llm_service,
)
from therapy.knowledge.distill import DistillResult
from therapy.knowledge.user_model import UserModel
from therapy.perception.stt import SUPPORTED_LANGUAGES


def test_language_enum_covers_supported_set() -> None:
    assert set(LANGUAGE_ENUM) == set(SUPPORTED_LANGUAGES)


def test_llm_factory_rejects_unknown_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("THERAPY_LLM", "clippy")
    with pytest.raises(ValueError, match="clippy"):
        make_llm_service()


def test_llm_factory_openrouter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("THERAPY_LLM", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    service = make_llm_service()
    assert type(service).__name__ == "OpenRouterLLMService"


def test_llm_factory_uses_deterministic_external_boundary_in_test_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("THERAPY_TEST_MODE", "1")

    assert type(make_llm_service()).__name__ == "DeterministicTestLLM"


def test_summary_failure_does_not_block_other_finalization_artifacts(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    async def broken_summary(_turns: Sequence[Mapping[str, object]]) -> str:
        calls.append("summary")
        raise RuntimeError("summary unavailable")

    async def distill(
        _model: UserModel,
        _turns: Sequence[Mapping[str, object]],
        _session_id: str,
    ) -> DistillResult:
        calls.append("distill")
        return DistillResult(run_id="run", promoted_nodes=[7])

    async def recap(_turns: Sequence[Mapping[str, object]]) -> str:
        calls.append("recap")
        return "Useful recap"

    async def title(_turns: Sequence[Mapping[str, object]]) -> str:
        calls.append("title")
        return "Useful title"

    artifacts = asyncio.run(
        generate_session_artifacts(
            [{"role": "user", "text": "hello"}],
            UserModel(tmp_path),
            "s1",
            summarize=broken_summary,
            distill=distill,
            recap=recap,
            title=title,
        )
    )

    assert calls == ["summary", "distill", "recap", "title"]
    assert artifacts.summary is None
    assert artifacts.distillation.promoted_nodes == [7]
    assert artifacts.recap == "Useful recap"
    assert artifacts.title == "Useful title"
