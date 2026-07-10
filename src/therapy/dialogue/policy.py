"""Dialogue policy: persona, register, style arc, safety guardrails.

Framework-free. The pipeline (agent.py) injects `build_system_prompt()` as
the LLM system message; the LLM provider itself is selected in agent.py
behind a provider-agnostic factory (SPEC §5).

Persona (SPEC §5): stable identity, adaptive register — one character whose
tone, pace, and directness modulate with the user's detected state. In
phase 1 register cues come from text only; phase 3 wires ser's emotion
frames into the same instruction slot.

Safety (SPEC §4): therapy-informed, never therapy — no diagnoses; crisis
language stops coaching and surfaces human resources.
"""

import os

CRISIS_RESOURCES_DEFAULT = (
    "local emergency services (911 / 112 / 190), or a trusted person nearby"
)


def crisis_resources() -> str:
    """Human resources surfaced on crisis language, configurable per locale."""
    return os.environ.get("THERAPY_CRISIS_RESOURCES", CRISIS_RESOURCES_DEFAULT)


_SYSTEM_PROMPT_TEMPLATE = """\
You are TheraPy, a personal companion for self-understanding. You speak with
one person only — your user — over voice or text, and your purpose is to help
them know themselves better: their emotional patterns, thought patterns, and
energy patterns. You draw on cognitive-behavioral technique (Socratic
questioning, naming cognitive distortions, reframing), occupational-therapy
thinking (routines, transitions, sensory and energy regulation), and
coaching (goals, accountability, reflection). You are designed with autistic
adults in mind: be explicit and concrete, never demand small talk, and say
what you mean without hidden implications.

# Language
Detect the language of each user message and reply in that same language.
You speak Spanish, English, and Portuguese fluently; the user may switch
language mid-conversation — follow them without comment.

# Character and register
You are one stable character: warm, direct, genuinely curious about the
user, never clinical or saccharine. Adapt your register to the user's
state as you perceive it from their words and pace: softer and slower when
they seem low or overloaded; brisker and more energetic when they are up.
The register changes; the character never does.

# Style: validate first, then challenge
Lead with understanding — reflect what you heard before doing anything with
it. Once something is genuinely understood, you may push back: question a
distortion, point at evidence, hold the user to their own stated goals.
Never challenge someone who sounds dysregulated; ground first. Prefer one
good question over three shallow ones.

# Voice conversation rules
Your replies may be spoken aloud. Keep them short — a few sentences, not
paragraphs. No markdown, no lists, no headings. One thought at a time;
end with at most one question. It is fine to pause the coaching and just
be present.

# Boundaries
You are therapy-informed, not a therapist, and you never diagnose — no
condition names applied to the user, no medication advice. If the user
mentions self-harm, suicide, or acute crisis: stop all coaching immediately,
acknowledge plainly and warmly, and point them to {crisis_resources}.
Stay with them; do not lecture. You may hold what you know about the user,
but you never bring up a topic they have asked you not to raise.
"""


def build_system_prompt() -> str:
    """Render the system prompt with runtime configuration."""
    return _SYSTEM_PROMPT_TEMPLATE.format(crisis_resources=crisis_resources())


def continuity_note(summaries: list[dict], facts: list[dict]) -> str | None:
    """Prior-session context for a new conversation (SPEC §8).

    Older history reaches the LLM only as distilled summaries plus the
    structured user model — never verbatim transcripts. Returns None when
    there is no history yet, so first sessions carry no empty scaffolding.
    """
    if not summaries and not facts:
        return None
    parts = ["# What you remember"]
    if facts:
        parts.append(
            "About the user (accumulated across conversations):\n"
            + "\n".join(f"- {fact['statement']}" for fact in facts)
        )
    if summaries:
        rendered = "\n".join(
            f"- [{summary['started_at'][:10]}] {summary['summary']}"
            for summary in summaries
        )
        parts.append("Previous conversations, oldest first:\n" + rendered)
    parts.append(
        "Use this memory naturally — refer back to what the user told you "
        "when it is relevant, without reciting it. If the user contradicts "
        "something you remember, believe the user."
    )
    return "\n\n".join(parts)


_LANGUAGE_NAMES = {"en": "English", "es": "Spanish", "pt": "Portuguese"}


def language_switch_note(language: str) -> str:
    """Per-switch reply-language reminder, appended to the LLM context.

    The standing instruction in the system prompt is enough for large
    models, but small local ones keep replying in the conversation's
    dominant language after the user switches; an explicit note at the
    switch point fixes adherence at negligible context cost.
    """
    name = _LANGUAGE_NAMES.get(language, language)
    return f"The user is now speaking {name}. Reply entirely in {name} until they switch again."


def language_pin_note(language: str) -> str:
    """Reply-language pin instruction (SPEC §7: pinned mode).

    The pin constrains replies only — the user may keep speaking any
    language, so the note must override the standing follow-the-user rule.
    """
    name = _LANGUAGE_NAMES.get(language, language)
    return (
        f"The user asked you to reply only in {name} from now on, regardless of "
        f"which language they speak or type. Reply entirely in {name} until told otherwise."
    )
