"""Deterministic agent double that exercises production Phase 4 orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np

from therapy.knowledge.context import ContextAssembler, ContextBudget, TurnContext
from therapy.knowledge.distill import DistillResult, distill_session
from therapy.knowledge.embeddings import EmbeddingMetadata
from therapy.knowledge.research import ResearchKB
from therapy.knowledge.user_model import UserModel
from therapy.memory import MemoryStore


class AgentTurnResult(TypedDict):
    """HTTP-safe scripted-agent outcome."""

    session_id: str
    reply: str
    memory_note: str | None
    insight: dict[str, object] | None
    resolution: dict[str, object] | None
    distillation: dict[str, object] | None


class AcceptanceEmbedding:
    """Small deterministic multilingual semantic boundary for acceptance only."""

    @property
    def metadata(self) -> EmbeddingMetadata:
        return EmbeddingMetadata("acceptance-semantic", "v1", 4)

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        folded = text.casefold()
        concepts = (
            ("late meeting", "energy", "drain", "reunião", "energia"),
            ("task", "checklist", "planning", "lista", "planejamento"),
            ("brother", "irmão", "hermano", "private"),
        )
        values = [float(any(term in folded for term in group)) for group in concepts]
        values.append(float(not any(values)))
        vector = np.asarray(values, dtype=np.float32)
        return vector / np.linalg.norm(vector)

    def embed_documents(self, texts: list[str]) -> list[np.ndarray]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


class ScriptedLLM:
    """Deterministic extraction/judgment/response double, never used in production."""

    @staticmethod
    async def extract(_transcript: str, observations: list[str]) -> list[dict[str, object]]:
        if not observations:
            return []
        text = observations[-1]
        folded = text.casefold()
        quote = [{"text": text}]
        if "late meeting" in folded and ("drain" in folded or "energy" in folded):
            return [
                {
                    "kind": "node",
                    "type": "pattern",
                    "statement": "Late meetings recur.",
                    "aliases": ["reuniões tarde"],
                    "quotes": quote,
                },
                {
                    "kind": "node",
                    "type": "pattern",
                    "statement": "Energy drops after late meetings.",
                    "aliases": ["energia cai depois de reuniões"],
                    "quotes": quote,
                },
                {
                    "kind": "edge",
                    "type": "triggers",
                    "src": "Late meetings recur.",
                    "dst": "Energy drops after late meetings.",
                    "statement": "Late meetings trigger an energy drop.",
                    "quotes": quote,
                },
            ]
        if "brother" in folded and "sensitive" in folded:
            return [
                {
                    "kind": "node",
                    "type": "thread",
                    "statement": "My brother is a sensitive ongoing thread.",
                    "quotes": quote,
                }
            ]
        return []

    @staticmethod
    async def judge(
        _kind: Literal["node", "edge"], _claim: Mapping[str, object]
    ) -> bool:
        return True

    @staticmethod
    def respond(text: str, context: TurnContext) -> str:
        resolution = context["resolution"]
        if resolution is not None:
            state = resolution["state"]
            return f"Thanks — I recorded that reflection as {state}."
        insight = context["insight"]
        if isinstance(insight, dict):
            return str(insight["reflection"])
        if "visible checklist" in str(context.get("research") or "").casefold():
            return "A visible checklist may reduce planning load for this transition."
        return "I’m with you. What feels most useful to notice here?"


class AcceptanceAgent:
    """Persist turns, refresh real context, distill, and answer through a test LLM."""

    def __init__(self, data_dir: Path) -> None:
        self.store = MemoryStore(data_dir)
        self.model = UserModel(data_dir)
        self.embedder = AcceptanceEmbedding()
        self.research = ResearchKB(data_dir, embedder=self.embedder)
        self.context = ContextAssembler(
            self.model,
            self.store,
            embedder=self.embedder,
            research=self.research,
            budget=ContextBudget(
                max_nodes=10,
                max_edges=10,
                max_episodes=3,
                max_tokens=1_200,
                node_threshold=0.4,
                episode_threshold=0.4,
            ),
        )
        self.llm = ScriptedLLM()

    async def turn(
        self,
        text: str,
        language: str,
        *,
        session_id: str | None = None,
        finalize: bool = False,
    ) -> AgentTurnResult:
        """Run one production-shaped text turn against isolated persistent data."""
        active_session = session_id or self.store.create_session()
        if not self.store.has_session(active_session):
            raise KeyError(active_session)
        self.store.add_turn(active_session, "user", "text", language, text)
        self.model.add_observation(text, session_id=active_session, language=language)
        context = self.context.assemble(text, active_session)
        reply = self.llm.respond(text, context)
        self.store.add_turn(active_session, "assistant", "text", language, reply)
        distilled: DistillResult | None = None
        if finalize:
            summary = f"The user said: {text}"
            self.store.end_session(active_session, summary)
            self.store.ensure_recap(active_session, f"You reflected on: {text}")
            distilled = await distill_session(
                self.model,
                self.store.session_turns(active_session),
                active_session,
                extractor=self.llm.extract,
                judger=self.llm.judge,
                extractor_version="acceptance-script-v1",
            )
            self.store.ensure_title(active_session, "Acceptance conversation")
        insight = context["insight"]
        resolution = context["resolution"]
        return AgentTurnResult(
            session_id=active_session,
            reply=reply,
            memory_note=context["note"],
            insight=dict(insight) if insight is not None else None,
            resolution=dict(resolution) if resolution is not None else None,
            distillation=asdict(distilled) if distilled is not None else None,
        )


__all__ = ["AcceptanceAgent", "AgentTurnResult"]
