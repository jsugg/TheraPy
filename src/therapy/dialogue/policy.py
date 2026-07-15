"""Dialogue policy: persona, register, style arc, safety guardrails.

Framework-free. The Pipecat pipeline injects `build_system_prompt()` as
the LLM system message; the LLM provider itself is selected by that adapter
behind a provider-agnostic factory (SPEC §5).

Persona (SPEC §5): stable identity, adaptive register — one character whose
tone, pace, and directness modulate with the user's detected state. In
phase 1 register cues come from text only; phase 3 wires ser's emotion
frames into the same instruction slot.

Safety (SPEC §4): therapy-informed, never therapy — no diagnoses; crisis
language stops coaching and surfaces human resources.
"""

from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from therapy.knowledge.user_model import GraphContextProvider

CRISIS_RESOURCES_DEFAULT = (
    "local emergency services (911 / 112 / 190), or a trusted person nearby"
)
logger = logging.getLogger(__name__)


class CrisisConfigurationError(ValueError):
    """Malformed owner-provided crisis contact configuration."""


def crisis_contacts() -> list[dict[str, str]]:
    """Configured crisis hotlines/contacts as `[{label, value}]` (W8, SPEC §4).

    Read from `THERAPY_CRISIS_CONTACTS` (a JSON array of `{"label","value"}`),
    so the owner can localize the safety protocol's resources without code
    changes. Malformed configuration raises clearly; the independent crisis
    response path catches it and retains the safe default.
    """
    raw = os.environ.get("THERAPY_CRISIS_CONTACTS", "").strip()
    if not raw:
        return []
    if len(raw) > 10_000:
        raise CrisisConfigurationError("THERAPY_CRISIS_CONTACTS exceeds 10 KB")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CrisisConfigurationError(
            f"THERAPY_CRISIS_CONTACTS is invalid JSON: {exc.msg}"
        ) from exc
    if not isinstance(parsed, list) or len(parsed) > 20:
        raise CrisisConfigurationError(
            "THERAPY_CRISIS_CONTACTS must be a JSON array with at most 20 entries"
        )
    contacts: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for index, entry in enumerate(parsed):
        if not isinstance(entry, dict) or set(entry) != {"label", "value"}:
            raise CrisisConfigurationError(
                f"crisis contact {index} must contain only label and value"
            )
        label = entry["label"]
        value = entry["value"]
        if not isinstance(label, str) or not isinstance(value, str):
            raise CrisisConfigurationError(
                f"crisis contact {index} label/value must be strings"
            )
        label = label.strip()
        value = value.strip()
        if not 1 <= len(label) <= 100 or not 1 <= len(value) <= 300:
            raise CrisisConfigurationError(
                f"crisis contact {index} label/value length is invalid"
            )
        if any(ord(character) < 32 for character in label + value):
            raise CrisisConfigurationError(
                f"crisis contact {index} contains control characters"
            )
        key = (label.casefold(), value.casefold())
        if key not in seen:
            contacts.append({"label": label, "value": value})
            seen.add(key)
    return contacts


def crisis_resources() -> str:
    """Human resources surfaced on crisis language, configurable per locale.

    Priority: the structured `crisis_contacts()` config (rendered as a list),
    then the `THERAPY_CRISIS_RESOURCES` string override, then a safe default.
    """
    try:
        contacts = crisis_contacts()
    except CrisisConfigurationError as exc:
        logger.error("Invalid crisis contact configuration: %s", exc)
        contacts = []
    if contacts:
        return "; ".join(f"{c['label']}: {c['value']}" for c in contacts)
    fallback = os.environ.get("THERAPY_CRISIS_RESOURCES", "").strip()
    if fallback and len(fallback) <= 2_000:
        return fallback
    if fallback:
        logger.error("THERAPY_CRISIS_RESOURCES exceeds 2000 characters; using default")
    return CRISIS_RESOURCES_DEFAULT


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


def graph_continuity_note(
    model: GraphContextProvider, *, topic: str = "", summaries: list[dict] | None = None
) -> str | None:
    """Graph-aware continuity for a new conversation (SPEC §8, W3).

    Replaces the Phase-2 "flat facts + summaries" injection: identity,
    preferences and the `never_initiate` list are always present, and a graph
    walk from the opening `topic` pulls in the most relevant confirmed/pattern
    claims with their confirmed edges. The model is protocol-typed to keep this
    framework-free module free of a runtime user-model import. Recent session
    summaries, when supplied, are appended as narrative colour. Returns None
    when there is nothing to inject.
    """
    from therapy.knowledge.user_model import render_context

    note = render_context(model.assemble_context(topic))
    parts = [note] if note else []
    if summaries:
        rendered = "\n".join(
            f"- [{summary['started_at'][:10]}] {summary['summary']}"
            for summary in summaries
        )
        parts.append("# Previous conversations, oldest first\n" + rendered)
    return "\n\n".join(parts) if parts else None


def resume_note() -> str:
    """Marker preceding rehydrated turns after a reconnect (SPEC §8).

    A dropped WebRTC connection is not a session boundary: the prior turns
    re-enter the context verbatim, and this note keeps the model from
    greeting the user as if the conversation were starting over.
    """
    return (
        "The connection dropped briefly and was restored. The messages below "
        "are the same ongoing conversation — continue it naturally; do not "
        "greet the user again or treat their next message as an opener."
    )


def rehydrate_messages(turns: list[dict], limit: int = 40) -> list[dict]:
    """Verbatim chat messages rebuilt from a resumed session's stored turns.

    Within one session the conversation reaches the LLM verbatim (the
    summaries-only rule applies across sessions, not to an interrupted
    one). Capped to the most recent turns so a marathon session cannot
    blow up the context on reconnect.
    """
    return [
        {"role": str(turn["role"]), "content": str(turn["text"])}
        for turn in turns[-limit:]
        if turn.get("text")
    ]


# The anchors are written IN the target language: for a small model every
# English system note in the context is itself English evidence pulling
# generation back toward English (field test: replies tagged es came out
# in English despite the anchor). In the target language the note is both
# the instruction and a language prime.
_SWITCH_NOTES = {
    "en": "The user is now speaking English. Reply entirely in English until they switch again.",
    "es": "El usuario ahora está hablando en español. Responde íntegramente en español hasta que cambie de idioma.",
    "pt": "O usuário agora está falando em português. Responda inteiramente em português até que ele mude de idioma.",
}

_REMINDERS = {
    "en": "Reply to the user's next message entirely in English.",
    "es": "Responde al próximo mensaje del usuario completamente en español.",
    "pt": "Responda à próxima mensagem do usuário inteiramente em português.",
}

_PIN_NOTES = {
    "en": (
        "The user asked you to reply only in English from now on, regardless of "
        "which language they speak or type. Reply entirely in English until told otherwise."
    ),
    "es": (
        "El usuario pidió que respondas solo en español de ahora en adelante, sin "
        "importar en qué idioma hable o escriba. Responde íntegramente en español "
        "hasta nuevo aviso."
    ),
    "pt": (
        "O usuário pediu que você responda apenas em português de agora em diante, "
        "não importa em que idioma ele fale ou escreva. Responda inteiramente em "
        "português até novo aviso."
    ),
}


def language_switch_note(language: str) -> str:
    """Per-switch reply-language reminder, appended to the LLM context.

    The standing instruction in the system prompt is enough for large
    models, but small local ones keep replying in the conversation's
    dominant language after the user switches; an explicit note at the
    switch point fixes adherence at negligible context cost.
    """
    return _SWITCH_NOTES.get(language) or (
        f"The user is now speaking {language}. "
        f"Reply entirely in {language} until they switch again."
    )


def reply_language_reminder(language: str) -> str:
    """Per-turn reply-language nudge (SPEC §7 auto mode).

    Small local models drift back to English over a long context even
    without a switch — the tag and TTS voice stay right while the text
    goes wrong. A short reminder adjacent to each user turn keeps the
    generation anchored; large models simply ignore the redundancy.
    """
    return _REMINDERS.get(language) or (
        f"Reply to the user's next message entirely in {language}."
    )


def language_pin_note(language: str) -> str:
    """Reply-language pin instruction (SPEC §7: pinned mode).

    The pin constrains replies only — the user may keep speaking any
    language, so the note must override the standing follow-the-user rule.
    """
    return _PIN_NOTES.get(language) or (
        f"The user asked you to reply only in {language} from now on, regardless "
        f"of which language they speak or type. Reply entirely in {language} "
        f"until told otherwise."
    )
