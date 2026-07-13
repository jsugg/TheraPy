import pytest

# agent.py is the only module that imports Pipecat (SPEC dependency boundary);
# on hosts without the realtime stack (e.g. the Intel-Mac dev box, where
# onnxruntime/kokoro have no wheels) collection would fail. Skip cleanly there —
# the container test bed has the full stack.
pytest.importorskip("pipecat")

from therapy.agent import LANGUAGE_ENUM, make_llm_service
from therapy.perception.stt import SUPPORTED_LANGUAGES


def test_language_enum_covers_supported_set() -> None:
    assert set(LANGUAGE_ENUM) == set(SUPPORTED_LANGUAGES)


def test_llm_factory_rejects_unknown_provider(monkeypatch) -> None:
    monkeypatch.setenv("THERAPY_LLM", "clippy")
    with pytest.raises(ValueError, match="clippy"):
        make_llm_service()


def test_llm_factory_openrouter(monkeypatch) -> None:
    monkeypatch.setenv("THERAPY_LLM", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    service = make_llm_service()
    assert type(service).__name__ == "OpenRouterLLMService"
