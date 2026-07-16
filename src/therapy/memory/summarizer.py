"""Session-summary distillation, provider-swappable (SPEC §8).

Only transcript text reaches the LLM — raw audio never leaves the data
directory. Provider selection mirrors the pipeline's `THERAPY_LLM`
convention (the Pipecat pipeline), but this module stays framework-free: the
SDK or plain HTTP, no pipeline services. Summarization runs once, at session
end, off the realtime path.
"""

import os
from collections.abc import Mapping, Sequence
from typing import Protocol

import httpx

from therapy.observability.capture import capture_service
from therapy.observability.interactions import (
    InteractionError,
    InteractionRequest,
    InteractionResponse,
    Message,
)
from therapy.observability.model import (
    InteractionEventKind,
    InteractionOperation,
    Provider,
    normalize_enum,
)

SUMMARY_PROMPT = """Distill the session transcript into a compact English summary.
Write plain prose for a future conversation to rely on, using at most about
eight sentences. Capture the topics discussed, the user's emotional state,
distinctive personal facts the user shared (keep names and specifics verbatim),
and any commitments or open threads. Phrase claims as remembered context, e.g.
"The user said..." Do not add advice, diagnoses, or facts not present in the
transcript."""


TITLE_PROMPT = """Give this session a very short title: three to six words naming its
main topic. Write the title in {language}. Output only the title itself —
no quotes, no trailing punctuation, no explanation."""

_TITLE_LANGUAGES = {"en": "English", "es": "Spanish", "pt": "Portuguese"}


def dominant_turn_language(turns: Sequence[Mapping[str, object]]) -> str:
    """Most frequent user-turn language code; 'en' when there are none."""
    counts: dict[str, int] = {}
    for turn in turns:
        if turn.get("role") == "user":
            code = str(turn.get("language") or "")
            if code:
                counts[code] = counts.get(code, 0) + 1
    return max(counts, key=lambda code: counts[code]) if counts else "en"


def clean_title(raw: str) -> str | None:
    """First line of LLM output, dequoted and capped; None if unusable."""
    stripped = raw.strip()
    if not stripped:
        return None
    line = stripped.splitlines()[0].strip().strip("\"'“”«»").rstrip(".!…").strip()
    return line[:80] or None


async def entitle(turns: Sequence[Mapping[str, object]]) -> str | None:
    """A short topic title in the session's dominant language."""
    if not turns:
        return None
    language = dominant_turn_language(turns)
    raw = await complete(
        TITLE_PROMPT.format(
            language=_TITLE_LANGUAGES.get(language, "the user's language")
        ),
        render_transcript(turns),
        max_tokens=40,
        operation=InteractionOperation.TITLE,
    )
    return clean_title(raw)


class Summarizer(Protocol):
    """Async session summarizer interface."""

    async def summarize(self, turns: Sequence[Mapping[str, object]]) -> str:
        """Summarize transcript turns for future context."""
        ...


def render_transcript(turns: Sequence[Mapping[str, object]]) -> str:
    """One compact line per turn: `user (es, voice): …` / `assistant (en): …`."""
    lines: list[str] = []
    for turn in turns:
        role = str(turn.get("role") or "unknown")
        language = str(turn.get("language") or "unknown")
        text = str(turn.get("text") or "")
        if role == "user":
            modality = str(turn.get("modality") or "unknown")
            lines.append(f"{role} ({language}, {modality}): {text}")
        else:
            lines.append(f"{role} ({language}): {text}")
    return "\n".join(lines)


async def complete(
    system: str,
    user: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    max_tokens: int = 500,
    operation: InteractionOperation = InteractionOperation.SUMMARY,
    session_id: str | None = None,
    turn_id: int | None = None,
) -> str:
    """One non-streaming completion against the configured provider.

    Shared by summarization and fact distillation — both are offline,
    single-shot calls that must not depend on the realtime pipeline.

    Every attempt is captured through the interaction journal BEFORE
    dispatch (plan O1.3); `operation` defaults to SUMMARY only for direct
    compatibility callers — production callers pass it explicitly.
    """
    provider = (provider or os.environ.get("THERAPY_LLM", "anthropic")).lower()
    if provider == "anthropic":
        resolved_model = model or "claude-opus-4-8"
    elif provider == "ollama":
        base_url = (
            base_url
            or os.environ.get("OLLAMA_BASE_URL")
            or "http://localhost:11434/v1"
        )
        resolved_model = (
            model or os.environ.get("THERAPY_LLM_MODEL") or "pedrolucas/smollm3:3b-q4_k_m"
        )
    elif provider == "openrouter":
        base_url = base_url or "https://openrouter.ai/api/v1"
        resolved_model = (
            model or os.environ.get("THERAPY_LLM_MODEL") or "openrouter/free"
        )
    else:
        raise ValueError(f"Unknown THERAPY_LLM provider: {provider!r}")

    handle = None
    service = capture_service()
    if service is not None:
        request = InteractionRequest(
            system_instructions=system,
            messages=(Message(role="user", content=user),),
            parameters={"max_tokens": max_tokens},
        )
        provider_request: dict[str, object] = {
            "model": resolved_model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
            "stream": False,
        }
        handle = await service.start_attempt(
            operation=operation,
            provider=normalize_enum(provider, Provider, Provider.UNKNOWN),
            requested_model=resolved_model,
            request=request,
            provider_request=provider_request,  # type: ignore[arg-type]
            session_id=session_id,
            turn_id=turn_id,
        )

    if provider == "anthropic":
        return await _complete_anthropic(
            system, user, resolved_model, max_tokens, handle
        )
    assert base_url is not None  # both non-anthropic branches set it above
    return await _complete_openai_style(
        system, user, resolved_model, base_url, provider, handle
    )


async def _complete_anthropic(
    system: str, user: str, model: str, max_tokens: int, handle
) -> str:
    from anthropic import APIError, AsyncAnthropic

    try:
        message = await AsyncAnthropic().messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
    except APIError as exc:
        if handle is not None:
            await handle.fail(
                InteractionError(
                    http_status=getattr(exc, "status_code", None),
                    provider_type=getattr(exc, "type", None) or type(exc).__name__,
                    provider_code=None,
                    provider_error_body=getattr(exc, "body", None)
                    and str(exc.body),
                    retry_attempt=0,
                    provider_request_id=getattr(exc, "request_id", None),
                )
            )
        raise
    except Exception as exc:
        if handle is not None:
            await handle.fail(
                InteractionError(
                    http_status=None,
                    provider_type=type(exc).__name__,
                    provider_code=None,
                    provider_error_body=None,
                    retry_attempt=0,
                    provider_request_id=None,
                ),
                finish_class="transport_error",
            )
        raise
    parts = [getattr(block, "text", "") or "" for block in message.content]
    completion = "".join(parts).strip()
    if handle is not None:
        usage = message.usage.model_dump() if message.usage else None
        native = message.model_dump(mode="json")
        await handle.record_event(
            InteractionEventKind.PROVIDER_EVENT, {"response": native}
        )
        await handle.succeed(
            InteractionResponse(
                messages=(Message(role="assistant", content=completion),),
                completion=completion,
                finish_reason=message.stop_reason,
                usage=usage,
            ),
            native_terminal={
                "id": message.id,
                "stop_reason": message.stop_reason,
                "usage": usage,
            },
        )
    return completion


async def _complete_openai_style(
    system: str,
    user: str,
    model: str,
    base_url: str,
    provider: str,
    handle,
) -> str:
    headers = None
    if provider == "openrouter":
        headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{base_url.rstrip('/')}/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                },
                headers=headers,
            )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        if handle is not None:
            await handle.fail(
                InteractionError(
                    http_status=exc.response.status_code,
                    provider_type="http_error",
                    provider_code=str(exc.response.status_code),
                    provider_error_body=exc.response.text[:100_000],
                    retry_attempt=0,
                    provider_request_id=None,
                ),
                finish_class=(
                    "rate_limit" if exc.response.status_code == 429 else "provider_error"
                ),
            )
        raise
    except httpx.HTTPError as exc:
        if handle is not None:
            await handle.fail(
                InteractionError(
                    http_status=None,
                    provider_type=type(exc).__name__,
                    provider_code=None,
                    provider_error_body=None,
                    retry_attempt=0,
                    provider_request_id=None,
                ),
                finish_class="transport_error",
            )
        raise
    body = response.json()
    content = str(body["choices"][0]["message"]["content"]).strip()
    if handle is not None:
        await handle.record_event(
            InteractionEventKind.PROVIDER_EVENT, {"response": body}
        )
        await handle.succeed(
            InteractionResponse(
                messages=(Message(role="assistant", content=content),),
                completion=content,
                finish_reason=body["choices"][0].get("finish_reason"),
                usage=body.get("usage"),
            ),
            native_terminal={
                "id": body.get("id"),
                "model": body.get("model"),
                "finish_reason": body["choices"][0].get("finish_reason"),
                "usage": body.get("usage"),
            },
        )
    return content


class LLMSummarizer:
    """LLM-backed session summarizer (anthropic | ollama | openrouter)."""

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._provider = (provider or os.environ.get("THERAPY_LLM", "anthropic")).lower()
        self._model = model
        self._base_url = base_url

    async def summarize(self, turns: Sequence[Mapping[str, object]]) -> str:
        """Return a compact session summary; empty string for no turns."""
        if not turns:
            return ""
        return await complete(
            SUMMARY_PROMPT,
            render_transcript(turns),
            provider=self._provider,
            model=self._model,
            base_url=self._base_url,
            operation=InteractionOperation.SUMMARY,
        )


def make_summarizer() -> Summarizer:
    """The configured summarizer (kept as a factory so alternatives can swap in)."""
    return LLMSummarizer()
