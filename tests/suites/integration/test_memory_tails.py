import asyncio
import json
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from types import TracebackType

import httpx
import pytest

from therapy.memory import distill as distill_module
from therapy.memory import summarizer as summarizer_module
from therapy.observability import telemetry as telemetry_module
from therapy.observability.model import InteractionOperation


def test_distill_facts_short_circuit_and_completion_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []

    async def fake_complete(
        system: str, user: str, **kwargs: object
    ) -> str:
        calls.append((system, user, kwargs))
        return "\nNONE\n" + "x" * 201 + "\n" + "\n".join(
            f"- fact {number}" for number in range(1, 7)
        )

    monkeypatch.setattr(distill_module, "complete", fake_complete)
    turns: list[dict[str, object]] = [
        {
            "role": "user",
            "modality": "text",
            "language": "en",
            "text": "I garden every weekend.",
        }
    ]

    assert asyncio.run(distill_module.distill_facts([])) == []
    assert asyncio.run(distill_module.distill_facts(turns)) == [
        "fact 1",
        "fact 2",
        "fact 3",
        "fact 4",
        "fact 5",
    ]
    assert calls == [
        (
            distill_module.FACTS_PROMPT,
            "user (en, text): I garden every weekend.",
            {
                "max_tokens": 300,
                "operation": InteractionOperation.DISTILL,
            },
        )
    ]


def test_entitle_short_circuit_and_unknown_language_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []

    async def fake_complete(
        system: str, user: str, **kwargs: object
    ) -> str:
        calls.append((system, user, kwargs))
        return "Daily stress.\nignored"

    monkeypatch.setattr(summarizer_module, "complete", fake_complete)
    turns: list[dict[str, object]] = [
        {
            "role": "user",
            "modality": "voice",
            "language": "xx",
            "text": "Work felt difficult.",
        }
    ]

    assert asyncio.run(summarizer_module.entitle([])) is None
    assert asyncio.run(summarizer_module.entitle(turns)) == "Daily stress"
    assert len(calls) == 1
    system, user, kwargs = calls[0]
    assert "the user's language" in system
    assert user == "user (xx, voice): Work felt difficult."
    assert kwargs == {
        "max_tokens": 40,
        "operation": InteractionOperation.TITLE,
    }


def test_openrouter_completion_uses_expected_request_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "completion-1",
                "model": "openrouter/test",
                "choices": [
                    {
                        "message": {"content": "  compact result  "},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 2},
            },
            request=request,
        )

    @asynccontextmanager
    async def fake_instrumented_client(
        provider: str, *, timeout: float
    ) -> AsyncGenerator[httpx.AsyncClient]:
        assert provider == "openrouter"
        assert timeout == 120.0
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            yield client

    def ignore_retry_count(_provider: str, _count: int) -> None:
        return None

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(summarizer_module, "capture_service", lambda: None)
    monkeypatch.setattr(
        telemetry_module, "instrumented_async_client", fake_instrumented_client
    )
    monkeypatch.setattr(
        telemetry_module,
        "record_outbound_retry_count",
        ignore_retry_count,
    )

    result = asyncio.run(
        summarizer_module.complete(
            "system prompt",
            "user prompt",
            provider="openrouter",
            model="openrouter/test",
        )
    )

    assert result == "compact result"
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == "https://openrouter.ai/api/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer test-key"
    assert json.loads(request.content) == {
        "model": "openrouter/test",
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "user prompt"},
        ],
        "stream": False,
    }


def test_complete_rejects_unknown_provider() -> None:
    with pytest.raises(
        ValueError, match="Unknown THERAPY_LLM provider: 'invalid'"
    ):
        asyncio.run(
            summarizer_module.complete(
                "system prompt", "user prompt", provider="invalid"
            )
        )


def test_anthropic_setup_failure_is_recorded_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failures: list[tuple[str, str, bool, bool]] = []
    retry_counts: list[tuple[str, int]] = []

    class FailingAsyncClient:
        async def __aenter__(self) -> httpx.AsyncClient:
            raise RuntimeError("mock client setup failed")

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: TracebackType | None,
        ) -> None:
            del exc_type, exc, traceback

    def fake_instrumented_client(
        _provider: str, *, timeout: httpx.Timeout | float
    ) -> FailingAsyncClient:
        del timeout
        return FailingAsyncClient()

    def record_failure(
        destination: str,
        operation: str,
        *,
        tls: bool,
        started_at: float,
        timed_out: bool,
    ) -> None:
        assert started_at > 0
        failures.append((destination, operation, tls, timed_out))

    def record_retry_count(destination: str, count: int) -> None:
        retry_counts.append((destination, count))

    monkeypatch.setattr(summarizer_module, "capture_service", lambda: None)
    monkeypatch.setattr(
        telemetry_module, "instrumented_async_client", fake_instrumented_client
    )
    monkeypatch.setattr(telemetry_module, "record_outbound_failure", record_failure)
    monkeypatch.setattr(
        telemetry_module, "record_outbound_retry_count", record_retry_count
    )

    with pytest.raises(RuntimeError, match="mock client setup failed"):
        asyncio.run(
            summarizer_module.complete(
                "system prompt", "user prompt", provider="anthropic"
            )
        )

    assert failures == [("anthropic", "post", True, False)]
    assert retry_counts == [("anthropic", 0)]


def test_llm_summarizer_and_factory_delegate_with_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, dict[str, object]]] = []

    async def fake_complete(
        system: str, user: str, **kwargs: object
    ) -> str:
        calls.append((system, user, kwargs))
        return "session summary"

    monkeypatch.setattr(summarizer_module, "complete", fake_complete)
    summarizer = summarizer_module.LLMSummarizer(
        provider="OLLAMA",
        model="local-test",
        base_url="http://ollama.test/v1",
    )
    turns: list[dict[str, object]] = [
        {
            "role": "assistant",
            "modality": "text",
            "language": "en",
            "text": "What helped?",
        }
    ]

    assert isinstance(summarizer_module.make_summarizer(), summarizer_module.LLMSummarizer)
    assert asyncio.run(summarizer.summarize(turns)) == "session summary"
    assert calls == [
        (
            summarizer_module.SUMMARY_PROMPT,
            "assistant (en): What helped?",
            {
                "provider": "ollama",
                "model": "local-test",
                "base_url": "http://ollama.test/v1",
                "operation": InteractionOperation.SUMMARY,
            },
        )
    ]
