"""End-to-end LLM boundary capture through `complete()` (plan O1.3, O1 gate).

A local fake OpenAI-style provider serves `/chat/completions`; the journal
must hold the exact pre-dispatch request, ordered native evidence, and the
terminal — for success, HTTP error, and transport failure — plus the
capture-unavailable policy per mode.
"""

import asyncio
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest

from therapy.memory.summarizer import complete
from therapy.observability.capture import (
    CaptureService,
    CaptureUnavailable,
    set_capture_service,
    start_capture,
)
from therapy.observability.config import ObservabilityConfig
from therapy.observability.journal import JournalStore
from therapy.observability.model import CaptureMode, InteractionOperation

COMPLETION_TEXT = "A captured synthetic completion."


class _FakeProvider(BaseHTTPRequestHandler):
    behavior = "ok"  # ok | rate_limit | error_body

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        if self.behavior == "rate_limit":
            body = json.dumps({"error": {"message": "slow down", "code": 429}})
            self.send_response(429)
        elif self.behavior == "error_body":
            body = json.dumps({"error": {"message": "exact provider error body"}})
            self.send_response(500)
        else:
            body = json.dumps(
                {
                    "id": "chatcmpl-fake-1",
                    "model": "fake-model",
                    "choices": [
                        {
                            "message": {"role": "assistant", "content": COMPLETION_TEXT},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 8},
                }
            )
            self.send_response(200)
        payload = body.encode()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        return


@pytest.fixture
def provider() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeProvider)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _FakeProvider.behavior = "ok"
    yield f"http://127.0.0.1:{server.server_port}/v1"
    server.shutdown()
    thread.join(timeout=5)


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_capture_requires_running_loop_setup(provider: str, tmp_path: Path,
                                             monkeypatch: pytest.MonkeyPatch) -> None:
    """Success path: pre-dispatch request, native event, and terminal are
    exactly journaled; the return value is unchanged."""
    monkeypatch.setenv("THERAPY_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("THERAPY_INTERACTION_JOURNAL", raising=False)
    config = ObservabilityConfig.from_env()

    async def scenario() -> None:
        runtime = await start_capture(config, build_version="test")
        try:
            result = await complete(
                "system prompt with exact content",
                "user words with exact content",
                provider="ollama",
                base_url=provider,
                model="fake-model",
                operation=InteractionOperation.SUMMARY,
                session_id="sess-capture-1",
            )
            assert result == COMPLETION_TEXT
            await runtime.writer.flush()

            store: JournalStore = runtime.store
            ids = list(store.iter_interaction_ids())
            assert len(ids) == 1
            loaded = store.load(ids[0])
            row = loaded["interaction"]
            assert row["status"] == "succeeded"
            assert row["operation"] == "summary"
            assert row["provider"] == "ollama"
            request = json.loads(row["canonical_request_json"])
            assert request["system_instructions"] == "system prompt with exact content"
            assert request["messages"][0]["content"] == "user words with exact content"
            native_request = json.loads(row["provider_request_json"])
            assert native_request["model"] == "fake-model"
            assert native_request["stream"] is False
            terminal = json.loads(row["terminal_json"])
            assert terminal["kind"] == "success"
            assert terminal["response"]["completion"] == COMPLETION_TEXT
            kinds = [event["kind"] for event in loaded["events"]]
            assert kinds == ["provider_event"]
            assert store.verify_checksums(ids[0])
        finally:
            await runtime.close()

    _run(scenario())


@pytest.mark.parametrize(
    ("behavior", "expected_status", "expected_type"),
    [("rate_limit", 429, "http_error"), ("error_body", 500, "http_error")],
)
def test_provider_error_body_is_captured_exactly(
    provider: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    behavior: str,
    expected_status: int,
    expected_type: str,
) -> None:
    monkeypatch.setenv("THERAPY_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("THERAPY_INTERACTION_JOURNAL", raising=False)
    config = ObservabilityConfig.from_env()
    _FakeProvider.behavior = behavior

    async def scenario() -> None:
        runtime = await start_capture(config, build_version="test")
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await complete(
                    "sys",
                    "user",
                    provider="ollama",
                    base_url=provider,
                    model="fake-model",
                    operation=InteractionOperation.DISTILL,
                )
            await runtime.writer.flush()
            store: JournalStore = runtime.store
            ids = list(store.iter_interaction_ids())
            assert len(ids) == 1
            row = store.load(ids[0])["interaction"]
            assert row["status"] == "failed"
            terminal = json.loads(row["terminal_json"])
            assert terminal["kind"] == "error"
            assert terminal["error"]["http_status"] == expected_status
            assert terminal["error"]["provider_type"] == expected_type
            # the exact body is preserved, not paraphrased
            assert "error" in json.loads(terminal["error"]["provider_error_body"])
        finally:
            await runtime.close()

    _run(scenario())


def test_transport_failure_is_terminal_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, free_port
) -> None:
    monkeypatch.setenv("THERAPY_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("THERAPY_INTERACTION_JOURNAL", raising=False)
    config = ObservabilityConfig.from_env()
    dead_endpoint = f"http://127.0.0.1:{free_port()}/v1"

    async def scenario() -> None:
        runtime = await start_capture(config, build_version="test")
        try:
            with pytest.raises(httpx.HTTPError):
                await complete(
                    "sys", "user", provider="ollama", base_url=dead_endpoint,
                    model="fake-model", operation=InteractionOperation.RECAP,
                )
            await runtime.writer.flush()
            store: JournalStore = runtime.store
            ids = list(store.iter_interaction_ids())
            row = store.load(ids[0])["interaction"]
            assert row["status"] == "failed"
            terminal = json.loads(row["terminal_json"])
            assert terminal["error"]["http_status"] is None
            assert terminal["error"]["provider_type"] == "ConnectError"
        finally:
            await runtime.close()

    _run(scenario())


def test_evaluation_mode_fails_closed_without_journal() -> None:
    service = CaptureService(None, mode=CaptureMode.EVALUATION)

    async def scenario() -> None:
        from therapy.observability.interactions import InteractionRequest, Message
        from therapy.observability.model import Provider

        with pytest.raises(CaptureUnavailable):
            await service.start_attempt(
                operation=InteractionOperation.EVALUATION,
                provider=Provider.OLLAMA,
                requested_model="m",
                request=InteractionRequest(
                    system_instructions="s", messages=(Message("user", "u"),)
                ),
                provider_request={"model": "m"},
            )

    _run(scenario())


def test_runtime_mode_without_journal_is_visible_gap_not_crash(
    provider: str,
) -> None:
    """Documented availability exception: journal missing at startup."""
    service = CaptureService(None, mode=CaptureMode.RUNTIME)
    set_capture_service(service)
    try:
        result = _run(
            complete(
                "sys", "user", provider="ollama", base_url=provider,
                model="fake-model", operation=InteractionOperation.SUMMARY,
            )
        )
        assert result == COMPLETION_TEXT
    finally:
        set_capture_service(None)
