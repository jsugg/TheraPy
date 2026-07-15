"""Deterministic golden interaction fixtures + canaries (plan O0.2).

Regenerates `tests/fixtures/observability/{canaries.json,classification.json,
interactions/*.json}` byte-identically. Fixtures are committed; this script
exists so a reviewer can prove they were produced by code, not edited by hand.

Every fixture file has two top-level planes:

- `record`      — the restricted canonical interaction record (§5.2 shape),
                  carrying unique content canaries;
- `broad_twin`  — the bounded content-free projection the broad plane may see.

The O0 gate scanner asserts content canaries appear only under `record`,
never under `broad_twin`, and forbidden canaries appear nowhere at all.

Usage: .venv/bin/python scripts/observability/gen_interaction_fixtures.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "tests/fixtures/observability"

# ---------------------------------------------------------------------------
# Canaries. Unique, grep-able, deliberately NOT shaped like real secrets.
# Content canaries must each appear in at least one restricted record.
# Forbidden canaries must appear nowhere in any fixture plane.
# ---------------------------------------------------------------------------

CONTENT_CANARIES: dict[str, str] = {
    "typed_transcript": "OBS-CANARY-TYPED-TRANSCRIPT-a41f07",
    "voice_transcript": "OBS-CANARY-VOICE-TRANSCRIPT-b52e18",
    "system_prompt": "OBS-CANARY-SYSTEM-PROMPT-c63d29",
    "completion": "OBS-CANARY-COMPLETION-d74c3a",
    "memory_note": "OBS-CANARY-MEMORY-NOTE-e85b4b",
    "retrieval_passage": "OBS-CANARY-RETRIEVAL-PASSAGE-f96a5c",
    "tool_definition": "OBS-CANARY-TOOL-DEFINITION-0a796d",
    "tool_arguments": "OBS-CANARY-TOOL-ARGUMENTS-1b887e",
    "tool_result": "OBS-CANARY-TOOL-RESULT-2c798f",
    "provider_error": "OBS-CANARY-PROVIDER-ERROR-3d6a90",
}

FORBIDDEN_CANARIES: dict[str, str] = {
    "sdp": "OBS-CANARY-SDP-BLOB-4e5ba1",
    "query_string": "OBS-CANARY-QUERY-STRING-5f4cb2",
    "auth_header": "OBS-CANARY-AUTH-HEADER-604dc3",
    "api_key": "OBS-CANARY-API-KEY-713ed4",
    "turn_key": "OBS-CANARY-TURN-KEY-822fe5",
    "push_key": "OBS-CANARY-PUSH-KEY-9310f6",
}

# ---------------------------------------------------------------------------
# Destination matrix (§5.2 canonical record + §5.4 broad event fields).
# ---------------------------------------------------------------------------

CLASSIFICATION: dict[str, str] = {
    # correlation and bounded envelope — broad-safe
    "interaction_id": "broad",
    "trace_id": "broad",
    "span_id": "broad",
    "operation": "broad",
    "provider": "broad",
    "requested_model": "broad",
    "prompt_template_version": "broad",
    "language": "broad",
    "modality": "broad",
    "build_version": "broad",
    "policy_version": "broad",
    "config_version": "broad",
    "started_at": "broad",
    "completed_at": "broad",
    "status": "broad",
    "response.usage": "broad",
    "error.http_status": "broad",
    "error.retry_attempt": "broad",
    # product IDs and exact content — restricted journal/backend only
    "session_id": "restricted",
    "turn_id": "restricted",
    "actual_model": "restricted",
    "request.system_instructions": "restricted",
    "request.messages": "restricted",
    "request.transcript": "restricted",
    "request.memory_notes": "restricted",
    "request.retrieved_documents": "restricted",
    "request.tools": "restricted",
    "request.parameters": "restricted",
    "request.response_schema": "restricted",
    "request.context_order": "restricted",
    "request.truncation": "restricted",
    "response.messages": "restricted",
    "response.completion": "restricted",
    "response.tool_calls": "restricted",
    "response.tool_results": "restricted",
    "response.finish_reason": "restricted",
    "stream": "restricted",
    "error.provider_type": "restricted",
    "error.provider_code": "restricted",
    "error.provider_error_body": "restricted",
    "error.provider_request_id": "restricted",
    "provider_native": "restricted",
    # never in either plane
    "sdp": "forbidden",
    "ice_candidates": "forbidden",
    "ip_address": "forbidden",
    "user_agent": "forbidden",
    "device_id": "forbidden",
    "ssrc": "forbidden",
    "raw_audio": "forbidden",
    "provider_request_headers": "forbidden",
    "provider_response_headers": "forbidden",
    "authorization_header": "forbidden",
    "cookie_header": "forbidden",
    "api_key": "forbidden",
    "ssh_key": "forbidden",
    "turn_secret": "forbidden",
    "vapid_private_key": "forbidden",
    "signing_key": "forbidden",
    "tls_key": "forbidden",
    "environment_dump": "forbidden",
    "sql_statement": "forbidden",
    "db_path": "forbidden",
    "url_query": "forbidden",
    "exception_repr": "forbidden",
}


def _base_record(
    *,
    case: str,
    operation: str,
    provider: str,
    requested_model: str,
    actual_model: str,
    status: str,
) -> dict[str, object]:
    """Common canonical-record scaffolding (§5.2), deterministic per case."""
    suffix = case.replace("_", "-")
    digest = hashlib.sha256(case.encode("utf-8")).hexdigest()
    return {
        "interaction_id": f"itx-{suffix}",
        "trace_id": digest[:32],
        "span_id": digest[32:48],
        "session_id": f"sess-{suffix}",
        "turn_id": 3,
        "operation": operation,
        "provider": provider,
        "requested_model": requested_model,
        "actual_model": actual_model,
        "prompt_template_version": "v3",
        "request": {
            "system_instructions": CONTENT_CANARIES["system_prompt"],
            "messages": [
                {"role": "user", "content": CONTENT_CANARIES["typed_transcript"]},
            ],
            "transcript": [
                {
                    "role": "user",
                    "language": "es",
                    "modality": "voice",
                    "text": CONTENT_CANARIES["voice_transcript"],
                }
            ],
            "memory_notes": [CONTENT_CANARIES["memory_note"]],
            "retrieved_documents": [
                {
                    "source_type": "research",
                    "source_id": "doc-77",
                    "anchor": "p4.b2",
                    "score": 0.83,
                    "rank": 1,
                    "text": CONTENT_CANARIES["retrieval_passage"],
                }
            ],
            "tools": [],
            "parameters": {"max_tokens": 500, "temperature": 1.0},
            "response_schema": None,
            "context_order": ["system", "memory", "retrieval", "messages"],
            "truncation": {"applied": False, "dropped_messages": 0},
        },
        "response": {
            "messages": [],
            "completion": None,
            "tool_calls": [],
            "tool_results": [],
            "finish_reason": None,
            "usage": None,
        },
        "stream": [],
        "error": None,
        "provider_native": {
            "request": {},
            "ordered_events": [],
            "terminal_response": None,
            "terminal_error": None,
        },
        "language": "es",
        "modality": "voice",
        "build_version": "0.1.0",
        "policy_version": "p2",
        "config_version": "c1",
        "started_at": "2026-07-15T12:00:00.000000+00:00",
        "completed_at": "2026-07-15T12:00:02.500000+00:00",
        "status": status,
    }


def _broad_twin(
    record: dict[str, object],
    *,
    outcome: str,
    finish_class: str,
    retry_count: int = 0,
    status_class: str = "2xx",
) -> dict[str, object]:
    """The bounded content-free projection (plan O1.3 item 4)."""
    return {
        "event.name": "llm.attempt",
        "component": "llm",
        "operation": record["operation"],
        "provider": record["provider"],
        "requested_model_policy": record["requested_model"],
        "trace_id": record["trace_id"],
        "span_id": record["span_id"],
        "input_size_bucket": "1k-4k",
        "output_size_bucket": "0-1k",
        "input_tokens": 412,
        "output_tokens": 96,
        "finish_class": finish_class,
        "retry_count": retry_count,
        "status_class": status_class,
        "ttft_ms": 350,
        "duration_ms": 2500,
        "outcome": outcome,
    }


def _anthropic_events(*, with_tools: bool, error_event: dict | None) -> list[dict]:
    completion = CONTENT_CANARIES["completion"]
    events: list[dict] = [
        {"type": "message_start", "message": {"id": "msg_fixture_01", "usage": {"input_tokens": 412, "output_tokens": 0}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "ping"},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": completion[: len(completion) // 2]}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": completion[len(completion) // 2 :]}},
        {"type": "content_block_stop", "index": 0},
    ]
    if with_tools:
        events += [
            {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "toolu_fixture", "name": "lookup_research"}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"query": "'}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": CONTENT_CANARIES["tool_arguments"] + '"}'}},
            {"type": "content_block_stop", "index": 1},
        ]
    if error_event is not None:
        events.append(error_event)
    else:
        events += [
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 96}},
            {"type": "message_stop"},
        ]
    return events


def case_anthropic_success() -> dict[str, object]:
    record = _base_record(
        case="anthropic_success",
        operation="reply",
        provider="anthropic",
        requested_model="claude-opus-4-8",
        actual_model="claude-opus-4-8",
        status="succeeded",
    )
    completion = CONTENT_CANARIES["completion"]
    record["response"] = {
        "messages": [{"role": "assistant", "content": completion}],
        "completion": completion,
        "tool_calls": [],
        "tool_results": [],
        "finish_reason": "end_turn",
        "usage": {"input_tokens": 412, "output_tokens": 96},
    }
    record["stream"] = [
        {"sequence": i, "observed_at": f"2026-07-15T12:00:0{1 + i % 2}.{i:06d}+00:00", "delta": part, "tool_delta": None}
        for i, part in enumerate([completion[: len(completion) // 2], completion[len(completion) // 2 :]])
    ]
    record["provider_native"] = {
        "request": {
            "body_request_id": "req_body_fixture_01",
            "model": "claude-opus-4-8",
            "system": CONTENT_CANARIES["system_prompt"],
            "messages": [{"role": "user", "content": CONTENT_CANARIES["typed_transcript"]}],
            "max_tokens": 500,
        },
        "response_request_id": "req_hdr_fixture_01",
        "ordered_events": _anthropic_events(with_tools=False, error_event=None),
        "terminal_response": {"stop_reason": "end_turn", "usage": {"input_tokens": 412, "output_tokens": 96}},
        "terminal_error": None,
        "sdk_retry": {"attempt": 0, "parsed_delay_seconds": None},
    }
    return {
        "case": "anthropic_success",
        "scenario": "provider_success",
        "record": record,
        "broad_twin": _broad_twin(record, outcome="success", finish_class="stop"),
    }


def case_anthropic_tools() -> dict[str, object]:
    record = _base_record(
        case="anthropic_tools",
        operation="tool",
        provider="anthropic",
        requested_model="claude-opus-4-8",
        actual_model="claude-opus-4-8",
        status="succeeded",
    )
    record["request"]["tools"] = [  # type: ignore[index]
        {
            "definition": {
                "name": "lookup_research",
                "description": CONTENT_CANARIES["tool_definition"],
                "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
            "arguments": None,
            "authorization_outcome": "allowed",
        }
    ]
    record["response"] = {
        "messages": [],
        "completion": CONTENT_CANARIES["completion"],
        "tool_calls": [
            {"id": "toolu_fixture", "name": "lookup_research", "arguments": {"query": CONTENT_CANARIES["tool_arguments"]}}
        ],
        "tool_results": [
            {"tool_call_id": "toolu_fixture", "content": CONTENT_CANARIES["tool_result"], "is_error": False}
        ],
        "finish_reason": "tool_use",
        "usage": {"input_tokens": 512, "output_tokens": 64},
    }
    record["provider_native"] = {
        "request": {"body_request_id": "req_body_fixture_02", "model": "claude-opus-4-8", "tools": [{"name": "lookup_research", "description": CONTENT_CANARIES["tool_definition"]}]},
        "response_request_id": "req_hdr_fixture_02",
        "ordered_events": _anthropic_events(with_tools=True, error_event=None),
        "terminal_response": {"stop_reason": "tool_use", "usage": {"input_tokens": 512, "output_tokens": 64}},
        "terminal_error": None,
        "sdk_retry": {"attempt": 0, "parsed_delay_seconds": None},
    }
    return {
        "case": "anthropic_tools",
        "scenario": "tools",
        "record": record,
        "broad_twin": _broad_twin(record, outcome="success", finish_class="tool_use"),
    }


def case_anthropic_retry_then_success() -> dict[str, object]:
    record = case_anthropic_success()["record"]
    record["interaction_id"] = "itx-anthropic-retry"  # type: ignore[index]
    native = record["provider_native"]  # type: ignore[index]
    native["sdk_retry"] = {"attempt": 2, "parsed_delay_seconds": 1.5}
    return {
        "case": "anthropic_retry_then_success",
        "scenario": "retry_fallback",
        "record": record,
        "broad_twin": _broad_twin(record, outcome="success", finish_class="stop", retry_count=2),  # type: ignore[arg-type]
    }


def case_anthropic_instream_error_after_200() -> dict[str, object]:
    record = _base_record(
        case="anthropic_instream_error",
        operation="reply",
        provider="anthropic",
        requested_model="claude-opus-4-8",
        actual_model="claude-opus-4-8",
        status="failed",
    )
    partial = CONTENT_CANARIES["completion"][: len(CONTENT_CANARIES["completion"]) // 2]
    error_event = {"type": "error", "error": {"type": "overloaded_error", "message": CONTENT_CANARIES["provider_error"]}}
    record["stream"] = [
        {"sequence": 0, "observed_at": "2026-07-15T12:00:01.000001+00:00", "delta": partial, "tool_delta": None}
    ]
    record["response"]["completion"] = partial  # type: ignore[index]
    record["error"] = {
        "http_status": 200,
        "provider_type": "overloaded_error",
        "provider_code": None,
        "provider_error_body": json.dumps(error_event["error"]),
        "retry_attempt": 0,
        "provider_request_id": "req_hdr_fixture_03",
    }
    record["provider_native"] = {
        "request": {"body_request_id": "req_body_fixture_03", "model": "claude-opus-4-8"},
        "response_request_id": "req_hdr_fixture_03",
        "ordered_events": _anthropic_events(with_tools=False, error_event=error_event)[:5] + [error_event],
        "terminal_response": None,
        "terminal_error": error_event,
        "sdk_retry": {"attempt": 0, "parsed_delay_seconds": None},
    }
    return {
        "case": "anthropic_instream_error_after_200",
        "scenario": "in_stream_error_after_200",
        "record": record,
        "broad_twin": _broad_twin(record, outcome="error", finish_class="provider_error", status_class="2xx"),
    }


def case_anthropic_empty_output() -> dict[str, object]:
    record = _base_record(
        case="anthropic_empty",
        operation="title",
        provider="anthropic",
        requested_model="claude-opus-4-8",
        actual_model="claude-opus-4-8",
        status="succeeded",
    )
    record["response"] = {
        "messages": [{"role": "assistant", "content": ""}],
        "completion": "",
        "tool_calls": [],
        "tool_results": [],
        "finish_reason": "end_turn",
        "usage": {"input_tokens": 200, "output_tokens": 0},
    }
    record["provider_native"] = {
        "request": {"body_request_id": "req_body_fixture_04", "model": "claude-opus-4-8"},
        "response_request_id": "req_hdr_fixture_04",
        "ordered_events": [
            {"type": "message_start", "message": {"id": "msg_fixture_04", "usage": {"input_tokens": 200, "output_tokens": 0}}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 0}},
            {"type": "message_stop"},
        ],
        "terminal_response": {"stop_reason": "end_turn", "usage": {"input_tokens": 200, "output_tokens": 0}},
        "terminal_error": None,
        "sdk_retry": {"attempt": 0, "parsed_delay_seconds": None},
    }
    return {
        "case": "anthropic_empty_output",
        "scenario": "empty_output",
        "record": record,
        "broad_twin": _broad_twin(record, outcome="success", finish_class="empty"),
    }


def case_anthropic_structured_output() -> dict[str, object]:
    record = _base_record(
        case="anthropic_structured",
        operation="distill",
        provider="anthropic",
        requested_model="claude-opus-4-8",
        actual_model="claude-opus-4-8",
        status="succeeded",
    )
    schema = {"type": "object", "properties": {"facts": {"type": "array", "items": {"type": "string"}}}, "required": ["facts"]}
    structured = json.dumps({"facts": [CONTENT_CANARIES["completion"]]})
    record["request"]["response_schema"] = schema  # type: ignore[index]
    record["response"] = {
        "messages": [{"role": "assistant", "content": structured}],
        "completion": structured,
        "tool_calls": [],
        "tool_results": [],
        "finish_reason": "end_turn",
        "usage": {"input_tokens": 300, "output_tokens": 40},
    }
    record["provider_native"] = {
        "request": {"body_request_id": "req_body_fixture_05", "model": "claude-opus-4-8"},
        "response_request_id": "req_hdr_fixture_05",
        "ordered_events": [
            {"type": "message_start", "message": {"id": "msg_fixture_05", "usage": {"input_tokens": 300, "output_tokens": 0}}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": structured}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 40}},
            {"type": "message_stop"},
        ],
        "terminal_response": {"stop_reason": "end_turn", "usage": {"input_tokens": 300, "output_tokens": 40}},
        "terminal_error": None,
        "sdk_retry": {"attempt": 0, "parsed_delay_seconds": None},
    }
    return {
        "case": "anthropic_structured_output",
        "scenario": "structured_output",
        "record": record,
        "broad_twin": _broad_twin(record, outcome="success", finish_class="stop"),
    }


def case_openrouter_success() -> dict[str, object]:
    record = _base_record(
        case="openrouter_success",
        operation="summary",
        provider="openrouter",
        requested_model="openrouter/free",
        actual_model="meta-llama/llama-3.3-70b-instruct",
        status="succeeded",
    )
    completion = CONTENT_CANARIES["completion"]
    record["response"] = {
        "messages": [{"role": "assistant", "content": completion}],
        "completion": completion,
        "tool_calls": [],
        "tool_results": [],
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 380, "completion_tokens": 88, "total_tokens": 468, "cost": 0.0},
    }
    record["provider_native"] = {
        "request": {"model": "openrouter/free", "routing_summary": {"route": "fallback", "allow_fallbacks": True}},
        "generation_id": "gen-fixture-or-01",
        "response_id": "chatcmpl-fixture-or-01",
        "ordered_events": [
            {"choices": [{"delta": {"content": completion}, "finish_reason": None}]},
            {"choices": [{"delta": {}, "finish_reason": "stop", "native_finish_reason": "eos"}], "usage": {"prompt_tokens": 380, "completion_tokens": 88, "cost": 0.0}},
        ],
        "terminal_response": {"finish_reason": "stop", "native_finish_reason": "eos", "actual_model": "meta-llama/llama-3.3-70b-instruct"},
        "terminal_error": None,
        "fallback_attempts": 0,
        "parsed_retry_after_seconds": None,
    }
    return {
        "case": "openrouter_success",
        "scenario": "provider_success",
        "record": record,
        "broad_twin": _broad_twin(record, outcome="success", finish_class="stop"),
    }


def case_openrouter_truncation() -> dict[str, object]:
    fixture = case_openrouter_success()
    record = fixture["record"]
    record["interaction_id"] = "itx-openrouter-truncation"  # type: ignore[index]
    record["response"]["finish_reason"] = "length"  # type: ignore[index]
    record["request"]["truncation"] = {"applied": True, "dropped_messages": 2}  # type: ignore[index]
    native = record["provider_native"]  # type: ignore[index]
    native["terminal_response"] = {"finish_reason": "length", "native_finish_reason": "MAX_TOKENS", "actual_model": "meta-llama/llama-3.3-70b-instruct"}
    return {
        "case": "openrouter_truncation",
        "scenario": "truncation",
        "record": record,
        "broad_twin": _broad_twin(record, outcome="success", finish_class="length"),  # type: ignore[arg-type]
    }


def case_openrouter_fallback_midstream_error() -> dict[str, object]:
    record = _base_record(
        case="openrouter_fallback",
        operation="summary",
        provider="openrouter",
        requested_model="openrouter/free",
        actual_model="mistralai/mistral-small",
        status="failed",
    )
    partial = CONTENT_CANARIES["completion"][:12]
    record["stream"] = [
        {"sequence": 0, "observed_at": "2026-07-15T12:00:01.000001+00:00", "delta": partial, "tool_delta": None}
    ]
    record["response"]["completion"] = partial  # type: ignore[index]
    record["error"] = {
        "http_status": 200,
        "provider_type": "provider_error",
        "provider_code": "502",
        "provider_error_body": json.dumps({"error": {"message": CONTENT_CANARIES["provider_error"], "code": 502}}),
        "retry_attempt": 1,
        "provider_request_id": "gen-fixture-or-02",
    }
    record["provider_native"] = {
        "request": {"model": "openrouter/free", "routing_summary": {"route": "fallback", "allow_fallbacks": True}},
        "generation_id": "gen-fixture-or-02",
        "response_id": "chatcmpl-fixture-or-02",
        "ordered_events": [
            {"choices": [{"delta": {"content": partial}, "finish_reason": None}]},
            {"error": {"message": CONTENT_CANARIES["provider_error"], "code": 502}},
        ],
        "terminal_response": None,
        "terminal_error": {"canonical_type": "provider_error", "provider_code": "502"},
        "fallback_attempts": 1,
        "parsed_retry_after_seconds": 30,
    }
    return {
        "case": "openrouter_fallback_midstream_error",
        "scenario": "retry_fallback",
        "record": record,
        "broad_twin": _broad_twin(record, outcome="error", finish_class="provider_error", retry_count=1),
    }


def case_openrouter_http_error() -> dict[str, object]:
    record = _base_record(
        case="openrouter_http_error",
        operation="summary",
        provider="openrouter",
        requested_model="openrouter/free",
        actual_model="",
        status="failed",
    )
    record["completed_at"] = "2026-07-15T12:00:00.900000+00:00"
    record["error"] = {
        "http_status": 429,
        "provider_type": "rate_limit",
        "provider_code": "429",
        "provider_error_body": json.dumps({"error": {"message": CONTENT_CANARIES["provider_error"], "code": 429}}),
        "retry_attempt": 0,
        "provider_request_id": None,
    }
    record["provider_native"] = {
        "request": {"model": "openrouter/free", "routing_summary": {"route": "default"}},
        "generation_id": None,
        "response_id": None,
        "ordered_events": [],
        "terminal_response": None,
        "terminal_error": {"canonical_type": "rate_limit", "provider_code": "429"},
        "fallback_attempts": 0,
        "parsed_retry_after_seconds": 5,
    }
    return {
        "case": "openrouter_http_error",
        "scenario": "http_error",
        "record": record,
        "broad_twin": _broad_twin(record, outcome="error", finish_class="rate_limit", status_class="4xx"),
    }


def case_ollama_success() -> dict[str, object]:
    record = _base_record(
        case="ollama_success",
        operation="recap",
        provider="ollama",
        requested_model="pedrolucas/smollm3:3b-q4_k_m",
        actual_model="pedrolucas/smollm3:3b-q4_k_m",
        status="succeeded",
    )
    completion = CONTENT_CANARIES["completion"]
    record["response"] = {
        "messages": [{"role": "assistant", "content": completion}],
        "completion": completion,
        "tool_calls": [],
        "tool_results": [],
        "finish_reason": "stop",
        "usage": {"prompt_eval_count": 250, "eval_count": 60},
    }
    record["provider_native"] = {
        "request": {"model": "pedrolucas/smollm3:3b-q4_k_m", "stream": True},
        "ordered_events": [
            {"model": "pedrolucas/smollm3:3b-q4_k_m", "message": {"role": "assistant", "content": completion[:14]}, "done": False},
            {"model": "pedrolucas/smollm3:3b-q4_k_m", "message": {"role": "assistant", "content": completion[14:]}, "done": False},
            {
                "model": "pedrolucas/smollm3:3b-q4_k_m",
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
                "total_duration": 2_513_402_111,
                "load_duration": 601_337_004,
                "prompt_eval_count": 250,
                "prompt_eval_duration": 812_004_512,
                "eval_count": 60,
                "eval_duration": 1_004_887_333,
            },
        ],
        "terminal_response": {
            "done_reason": "stop",
            "total_duration": 2_513_402_111,
            "load_duration": 601_337_004,
            "prompt_eval_count": 250,
            "prompt_eval_duration": 812_004_512,
            "eval_count": 60,
            "eval_duration": 1_004_887_333,
        },
        "terminal_error": None,
    }
    return {
        "case": "ollama_success",
        "scenario": "provider_success",
        "record": record,
        "broad_twin": _broad_twin(record, outcome="success", finish_class="stop"),
    }


def case_ollama_error_body() -> dict[str, object]:
    record = _base_record(
        case="ollama_error",
        operation="recap",
        provider="ollama",
        requested_model="pedrolucas/smollm3:3b-q4_k_m",
        actual_model="",
        status="failed",
    )
    body = {"error": f"model not loaded: {CONTENT_CANARIES['provider_error']}"}
    record["error"] = {
        "http_status": 500,
        "provider_type": "server_error",
        "provider_code": None,
        "provider_error_body": json.dumps(body),
        "retry_attempt": 0,
        "provider_request_id": None,
    }
    record["provider_native"] = {
        "request": {"model": "pedrolucas/smollm3:3b-q4_k_m", "stream": True},
        "ordered_events": [body],
        "terminal_response": None,
        "terminal_error": body,
    }
    return {
        "case": "ollama_error_body",
        "scenario": "http_error",
        "record": record,
        "broad_twin": _broad_twin(record, outcome="error", finish_class="provider_error", status_class="5xx"),
    }


def case_timeout() -> dict[str, object]:
    record = _base_record(
        case="ollama_timeout",
        operation="summary",
        provider="ollama",
        requested_model="pedrolucas/smollm3:3b-q4_k_m",
        actual_model="",
        status="failed",
    )
    record["error"] = {
        "http_status": None,
        "provider_type": "timeout",
        "provider_code": None,
        "provider_error_body": None,
        "retry_attempt": 0,
        "provider_request_id": None,
    }
    record["provider_native"] = {
        "request": {"model": "pedrolucas/smollm3:3b-q4_k_m", "stream": True},
        "ordered_events": [],
        "terminal_response": None,
        "terminal_error": {"canonical_type": "timeout", "timeout_seconds": 120.0},
    }
    return {
        "case": "timeout",
        "scenario": "timeout",
        "record": record,
        "broad_twin": _broad_twin(record, outcome="timeout", finish_class="timeout", status_class="none"),
    }


def case_cancellation() -> dict[str, object]:
    record = _base_record(
        case="anthropic_cancellation",
        operation="reply",
        provider="anthropic",
        requested_model="claude-opus-4-8",
        actual_model="claude-opus-4-8",
        status="incomplete",
    )
    partial = CONTENT_CANARIES["completion"][:10]
    record["stream"] = [
        {"sequence": 0, "observed_at": "2026-07-15T12:00:01.000001+00:00", "delta": partial, "tool_delta": None}
    ]
    record["response"]["completion"] = partial  # type: ignore[index]
    record["completed_at"] = None
    record["provider_native"] = {
        "request": {"body_request_id": "req_body_fixture_06", "model": "claude-opus-4-8"},
        "response_request_id": "req_hdr_fixture_06",
        "ordered_events": [
            {"type": "message_start", "message": {"id": "msg_fixture_06", "usage": {"input_tokens": 412, "output_tokens": 0}}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": partial}},
        ],
        "terminal_response": None,
        "terminal_error": {"canonical_type": "cancelled", "reason": "barge_in"},
        "sdk_retry": {"attempt": 0, "parsed_delay_seconds": None},
    }
    return {
        "case": "cancellation",
        "scenario": "cancellation",
        "record": record,
        "broad_twin": _broad_twin(record, outcome="cancelled", finish_class="cancelled"),
    }


def case_partial_stream_disconnect() -> dict[str, object]:
    record = _base_record(
        case="openrouter_partial_stream",
        operation="reply",
        provider="openrouter",
        requested_model="openrouter/free",
        actual_model="meta-llama/llama-3.3-70b-instruct",
        status="incomplete",
    )
    partial = CONTENT_CANARIES["completion"][:8]
    record["stream"] = [
        {"sequence": 0, "observed_at": "2026-07-15T12:00:01.000001+00:00", "delta": partial, "tool_delta": None}
    ]
    record["response"]["completion"] = partial  # type: ignore[index]
    record["completed_at"] = None
    record["provider_native"] = {
        "request": {"model": "openrouter/free", "routing_summary": {"route": "default"}},
        "generation_id": "gen-fixture-or-03",
        "response_id": "chatcmpl-fixture-or-03",
        "ordered_events": [{"choices": [{"delta": {"content": partial}, "finish_reason": None}]}],
        "terminal_response": None,
        "terminal_error": {"canonical_type": "connection_lost"},
        "fallback_attempts": 0,
        "parsed_retry_after_seconds": None,
    }
    return {
        "case": "partial_stream_disconnect",
        "scenario": "partial_stream",
        "record": record,
        "broad_twin": _broad_twin(record, outcome="incomplete", finish_class="connection_lost"),
    }


CASES = (
    case_anthropic_success,
    case_anthropic_tools,
    case_anthropic_retry_then_success,
    case_anthropic_instream_error_after_200,
    case_anthropic_empty_output,
    case_anthropic_structured_output,
    case_openrouter_success,
    case_openrouter_truncation,
    case_openrouter_fallback_midstream_error,
    case_openrouter_http_error,
    case_ollama_success,
    case_ollama_error_body,
    case_timeout,
    case_cancellation,
    case_partial_stream_disconnect,
)


def main() -> int:
    interactions = FIXTURE_ROOT / "interactions"
    interactions.mkdir(parents=True, exist_ok=True)

    (FIXTURE_ROOT / "canaries.json").write_text(
        json.dumps(
            {"content": CONTENT_CANARIES, "forbidden": FORBIDDEN_CANARIES},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (FIXTURE_ROOT / "classification.json").write_text(
        json.dumps(CLASSIFICATION, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for build in CASES:
        fixture = build()
        path = interactions / f"{fixture['case']}.json"
        path.write_text(
            json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {path.relative_to(FIXTURE_ROOT.parents[1])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
