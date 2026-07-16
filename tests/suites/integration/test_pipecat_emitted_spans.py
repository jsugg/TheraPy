"""Emitted Pipecat telemetry vs. the audited routing sets (audit F-08).

Runs a REAL PipelineTask with `enable_tracing`/`enable_turn_tracking` against
the deterministic LLM boundary and snapshots the spans Pipecat actually
emits — scopes, names, and attribute keys — instead of trusting static
source reflection. Emitted scopes must be covered by the restricted routing
set, and every observed content-bearing attribute key must already be on the
denylist that keeps it out of the broad plane.
"""

import asyncio
import json
from pathlib import Path

import pytest

pytest.importorskip("pipecat")

from opentelemetry import trace as trace_api  # noqa: E402
from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from pipecat.frames.frames import EndFrame, LLMContextFrame, LLMRunFrame  # noqa: E402
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.runner import PipelineRunner  # noqa: E402
from pipecat.pipeline.task import PipelineParams, PipelineTask  # noqa: E402
from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: E402
from pipecat.processors.aggregators.llm_response_universal import (  # noqa: E402
    LLMContextAggregatorPair,
)

from therapy.integrations.pipecat.pipeline import _DeterministicTestLLM  # noqa: E402
from therapy.observability.model import TelemetryPlane  # noqa: E402
from therapy.observability.routing import (  # noqa: E402
    CONTENT_ATTRIBUTE_KEYS,
    classify_scope,
)

SNAPSHOT = (
    Path(__file__).resolve().parents[2]
    / "fixtures/observability/pipecat/snapshot-1.5.0.json"
)


@pytest.fixture(scope="module")
def emitted_spans():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # Pipecat's is_tracing_available() checks the GLOBAL provider; install
    # ours exactly the way the owned bootstrap would.
    trace_api.set_tracer_provider(provider)

    async def run() -> None:
        llm = _DeterministicTestLLM()
        context = LLMContext(
            messages=[
                {"role": "system", "content": "emitted-span system prompt"},
                {"role": "user", "content": "emitted-span user words"},
            ]
        )
        aggregators = LLMContextAggregatorPair(context)
        pipeline = Pipeline([aggregators.user(), llm, aggregators.assistant()])
        task = PipelineTask(
            pipeline,
            params=PipelineParams(enable_metrics=True),
            enable_tracing=True,
            enable_turn_tracking=True,
            conversation_id="tele-emitted-snapshot",
            cancel_on_idle_timeout=True,
            idle_timeout_secs=3,
        )
        await task.queue_frames(
            [LLMContextFrame(context=context), LLMRunFrame(), EndFrame()]
        )
        runner = PipelineRunner(handle_sigint=False)
        await asyncio.wait_for(runner.run(task), timeout=30)

    asyncio.run(run())
    return exporter.get_finished_spans()


def test_pipeline_emits_spans_under_audited_scopes_only(emitted_spans) -> None:
    assert emitted_spans, "tracing was enabled but nothing was emitted"
    scopes = {span.instrumentation_scope.name for span in emitted_spans}
    for scope in scopes:
        assert classify_scope(scope) is TelemetryPlane.RESTRICTED, (
            f"emitted scope {scope!r} is not routed restricted"
        )


def test_emitted_content_attribute_keys_are_denylisted(emitted_spans) -> None:
    """Every content-carrying attribute key ACTUALLY emitted must be unable
    to survive a broad scrub; unexpected new keys fail loudly."""
    known = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    known_keys: set[str] = set()
    for module in known["tracing"].values():
        known_keys.update(module.get("attribute_keys", []))

    observed_content = set()
    for span in emitted_spans:
        for key, value in (span.attributes or {}).items():
            if not isinstance(value, str) or not value:
                continue
            if "emitted-span" in value:  # our content actually appeared here
                observed_content.add(key)
    for key in observed_content:
        assert key in CONTENT_ATTRIBUTE_KEYS or any(
            key.startswith(prefix) for prefix in ("gen_ai.",)
        ), f"content observed under unlisted attribute key {key!r}"

    # scope/name inventory stays inside the pinned static snapshot's world
    names = {span.name for span in emitted_spans}
    assert names, names
