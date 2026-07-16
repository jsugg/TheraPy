"""User-model v1 distillation: session transcript → durable facts (SPEC §8).

The full property-graph model (SPEC Appendix A) lands in later phases; v1
keeps the core loop honest — distinctive personal facts are extracted at
session end as canonical-English statements and accumulated in the store,
so every new conversation starts knowing who it is talking to. Exact
re-statements reinforce (occurrence counts), never duplicate.
"""

from therapy.memory.summarizer import complete, render_transcript
from therapy.observability.model import InteractionOperation

FACTS_PROMPT = """From the session transcript, list the distinctive, stable
personal facts about the user worth remembering across conversations: names
and relationships, life circumstances, preferences, ongoing projects or
threads. At most five. One fact per line, as a short canonical English
statement (e.g. "Has a dog named Bruno."), keeping names and specifics
verbatim. No numbering, no bullets, no commentary. If the transcript holds
nothing worth keeping, output exactly NONE."""


def parse_facts(raw: str) -> list[str]:
    """Parse LLM output into clean fact statements (defensive by design)."""
    facts: list[str] = []
    for line in raw.splitlines():
        statement = line.strip().lstrip("-•*").strip()
        if not statement or statement.upper() == "NONE":
            continue
        if len(statement) > 200:  # a paragraph is not a fact
            continue
        facts.append(statement)
    return facts[:5]


async def distill_facts(turns: list[dict[str, object]]) -> list[str]:
    """Extract user-model v1 facts from a session's turns."""
    if not turns:
        return []
    raw = await complete(
        FACTS_PROMPT,
        render_transcript(turns),
        max_tokens=300,
        operation=InteractionOperation.DISTILL,
    )
    return parse_facts(raw)
