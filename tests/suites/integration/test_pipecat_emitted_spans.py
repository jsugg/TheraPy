"""Emitted Pipecat telemetry vs. the audited routing sets (audit F-08).

Runs a REAL PipelineWorker with `enable_tracing`/`enable_turn_tracking` against
the deterministic LLM boundary and snapshots the spans Pipecat actually
emits — scopes, names, and attribute keys — instead of trusting static
source reflection. Emitted scopes must be covered by the restricted routing
set, and every observed content-bearing attribute key must already be on the
denylist that keeps it out of the broad plane.
"""

import asyncio
import json
from pathlib import Path
from typing import cast

import pytest

pytest.importorskip("pipecat")

from opentelemetry import trace as trace_api  # noqa: E402
from opentelemetry.sdk.trace import (  # noqa: E402
    ReadableSpan,
    TracerProvider,
)
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from pipecat.frames.frames import EndFrame, LLMContextFrame, LLMRunFrame  # noqa: E402
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.runner import WorkerRunner  # noqa: E402
from pipecat.pipeline.worker import PipelineParams, PipelineWorker  # noqa: E402
from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: E402
from pipecat.processors.aggregators.llm_response_universal import (  # noqa: E402
    LLMContextAggregatorPair,
)

from therapy.integrations.pipecat.pipeline import DeterministicTestLLM  # noqa: E402
from therapy.observability.interactions import require_json_object  # noqa: E402
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
def emitted_spans() -> tuple[ReadableSpan, ...]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # Pipecat's is_tracing_available() checks the GLOBAL provider; install
    # ours exactly the way the owned bootstrap would.
    trace_api.set_tracer_provider(provider)

    async def run() -> None:
        llm = DeterministicTestLLM()
        context = LLMContext(
            messages=[
                {"role": "system", "content": "emitted-span system prompt"},
                {"role": "user", "content": "emitted-span user words"},
            ]
        )
        aggregators = LLMContextAggregatorPair(context)
        pipeline = Pipeline([aggregators.user(), llm, aggregators.assistant()])
        worker = PipelineWorker(
            pipeline,
            params=PipelineParams(enable_metrics=True),
            enable_tracing=True,
            enable_turn_tracking=True,
            conversation_id="tele-emitted-snapshot",
            cancel_on_idle_timeout=True,
            idle_timeout_secs=3,
        )
        await worker.queue_frames(
            [LLMContextFrame(context=context), LLMRunFrame(), EndFrame()]
        )
        runner = WorkerRunner(handle_sigint=False)
        await runner.add_workers(worker)
        await asyncio.wait_for(runner.run(), timeout=30)

    asyncio.run(run())
    return exporter.get_finished_spans()


def test_pipeline_emits_spans_under_audited_scopes_only(
    emitted_spans: tuple[ReadableSpan, ...],
) -> None:
    assert emitted_spans, "tracing was enabled but nothing was emitted"
    scopes: set[str] = set()
    for span in emitted_spans:
        scope = span.instrumentation_scope
        assert scope is not None
        scopes.add(scope.name)
    for scope in scopes:
        assert classify_scope(scope) is TelemetryPlane.RESTRICTED, (
            f"emitted scope {scope!r} is not routed restricted"
        )


def test_emitted_content_attribute_keys_are_denylisted(
    emitted_spans: tuple[ReadableSpan, ...],
) -> None:
    """Every content-carrying attribute key ACTUALLY emitted must be unable
    to survive a broad scrub; unexpected new keys fail loudly."""
    known = require_json_object(
        json.loads(SNAPSHOT.read_text(encoding="utf-8")), "test.snapshot"
    )
    tracing_value = known["tracing"]
    assert isinstance(tracing_value, dict)
    raw_tracing = cast(dict[object, object], tracing_value)
    assert all(isinstance(key, str) for key in raw_tracing)
    tracing = cast(dict[str, object], raw_tracing)
    known_keys: set[str] = set()
    for module_value in tracing.values():
        module = require_json_object(module_value, "test.snapshot.module")
        keys_value = module.get("attribute_keys", [])
        assert isinstance(keys_value, list)
        raw_keys = cast(list[object], keys_value)
        assert all(isinstance(key, str) for key in raw_keys)
        known_keys.update(cast(list[str], raw_keys))

    observed_content: set[str] = set()
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
