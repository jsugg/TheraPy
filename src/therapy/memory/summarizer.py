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
) -> str:
    """One non-streaming completion against the configured provider.

    Shared by summarization and fact distillation — both are offline,
    single-shot calls that must not depend on the realtime pipeline.
    """
    provider = (provider or os.environ.get("THERAPY_LLM", "anthropic")).lower()
    if provider == "anthropic":
        from anthropic import AsyncAnthropic

        message = await AsyncAnthropic().messages.create(
            model=model or "claude-opus-4-8",
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts = [getattr(block, "text", "") or "" for block in message.content]
        return "".join(parts).strip()

    if provider == "ollama":
        base_url = (
            base_url
            or os.environ.get("OLLAMA_BASE_URL")
            or "http://localhost:11434/v1"
        )
        headers = None
        default_model = "pedrolucas/smollm3:3b-q4_k_m"
    elif provider == "openrouter":
        base_url = base_url or "https://openrouter.ai/api/v1"
        headers = {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}
        default_model = "openrouter/free"
    else:
        raise ValueError(f"Unknown THERAPY_LLM provider: {provider!r}")

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json={
                "model": model or os.environ.get("THERAPY_LLM_MODEL") or default_model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "stream": False,
            },
            headers=headers,
        )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return str(content).strip()


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
        )


def make_summarizer() -> Summarizer:
    """The configured summarizer (kept as a factory so alternatives can swap in)."""
    return LLMSummarizer()
