"""Logical instrument manifest with bounded attribute sets (plan §8).

Frozen declarations only — real instruments are created from this manifest
by `telemetry.py` when broad OTel is enabled. Freezing names/attributes here
lets tests catch accidental renames and cardinality growth before they ship
(plan §8: "Freeze their final names in `metrics.py` before use").

Every attribute set is finite and documented; labels never carry IDs, model
names under dynamic routing, timestamps, URLs, content, exception bodies, or
concrete paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class InstrumentKind(StrEnum):
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"


@dataclass(frozen=True, slots=True)
class InstrumentSpec:
    name: str
    kind: InstrumentKind
    unit: str
    #: attribute name -> allowed values (None = documented finite set kept
    #: bounded by an enum in `model.py`, e.g. provider/operation/outcome).
    attributes: dict[str, tuple[str, ...] | None] = field(default_factory=dict)


_STAGES = ("vad", "stt", "persist_user", "context", "llm", "persist_assistant", "tts", "output")
_CAPTURE_STATUS = ("journaled", "exported", "incomplete", "failed")
_LLM_RESULT = ("ok", "empty", "truncated", "language_mismatch", "invalid_structured")

INSTRUMENTS: tuple[InstrumentSpec, ...] = (
    # --- voice/service ---
    InstrumentSpec("therapy_voice_connections_total", InstrumentKind.COUNTER, "1",
                   {"outcome": None, "reason": None}),
    InstrumentSpec("therapy_voice_active_connections", InstrumentKind.GAUGE, "1"),
    InstrumentSpec("therapy_voice_pipeline_transitions_total", InstrumentKind.COUNTER,
                   "1", {"transition": None, "outcome": None}),
    InstrumentSpec("therapy_voice_pipeline_shutdown_seconds", InstrumentKind.HISTOGRAM,
                   "s", {"outcome": None}),
    InstrumentSpec("therapy_conversation_turns_total", InstrumentKind.COUNTER, "1",
                   {"modality": None, "language_group": None, "outcome": None}),
    InstrumentSpec("therapy_turn_stage_duration_seconds", InstrumentKind.HISTOGRAM,
                   "s", {"stage": _STAGES, "outcome": None}),
    InstrumentSpec("therapy_turn_ttfa_seconds", InstrumentKind.HISTOGRAM, "s",
                   {"provider": None, "mode": ("cold", "warm")}),
    InstrumentSpec("therapy_llm_time_to_first_token_seconds", InstrumentKind.HISTOGRAM,
                   "s", {"provider": None, "operation": None, "outcome": None}),
    InstrumentSpec("therapy_stt_realtime_factor", InstrumentKind.HISTOGRAM, "1",
                   {"outcome": None}),
    InstrumentSpec("therapy_stt_redecode_total", InstrumentKind.COUNTER, "1",
                   {"reason": None}),
    InstrumentSpec("therapy_stt_empty_total", InstrumentKind.COUNTER, "1",
                   {"reason": None}),
    InstrumentSpec("therapy_stt_model_load_seconds", InstrumentKind.HISTOGRAM, "s",
                   {"outcome": None}),
    InstrumentSpec("therapy_llm_requests_total", InstrumentKind.COUNTER, "1",
                   {"provider": None, "operation": None, "outcome": None}),
    InstrumentSpec("therapy_llm_input_tokens_total", InstrumentKind.COUNTER, "1",
                   {"provider": None, "operation": None}),
    InstrumentSpec("therapy_llm_output_tokens_total", InstrumentKind.COUNTER, "1",
                   {"provider": None, "operation": None}),
    InstrumentSpec("therapy_llm_retries_total", InstrumentKind.COUNTER, "1",
                   {"provider": None, "operation": None, "reason": None}),
    InstrumentSpec("therapy_llm_rate_limits_total", InstrumentKind.COUNTER, "1",
                   {"provider": None, "operation": None}),
    InstrumentSpec("therapy_llm_output_total", InstrumentKind.COUNTER, "1",
                   {"provider": None, "operation": None, "result": _LLM_RESULT}),
    InstrumentSpec("therapy_tts_requests_total", InstrumentKind.COUNTER, "1",
                   {"language_group": None, "outcome": None}),
    InstrumentSpec("therapy_tts_characters_total", InstrumentKind.COUNTER, "1",
                   {"language_group": None}),
    InstrumentSpec("therapy_tts_synthesis_seconds", InstrumentKind.HISTOGRAM, "s",
                   {"language_group": None, "outcome": None}),
    InstrumentSpec("therapy_tts_time_to_first_audio_seconds", InstrumentKind.HISTOGRAM,
                   "s", {"language_group": None, "outcome": None}),
    InstrumentSpec("therapy_tts_realtime_factor", InstrumentKind.HISTOGRAM, "1",
                   {"language_group": None, "outcome": None}),
    InstrumentSpec("therapy_barge_ins_total", InstrumentKind.COUNTER, "1",
                   {"outcome": None}),
    InstrumentSpec("therapy_barge_in_stop_seconds", InstrumentKind.HISTOGRAM, "s",
                   {"outcome": None}),
    InstrumentSpec("therapy_webrtc_data_channel_open_seconds", InstrumentKind.HISTOGRAM,
                   "s", {"outcome": None}),
    InstrumentSpec("therapy_webrtc_rtt_seconds", InstrumentKind.HISTOGRAM, "s",
                   {"candidate_type": ("relay", "host", "srflx")}),
    InstrumentSpec("therapy_webrtc_jitter_seconds", InstrumentKind.HISTOGRAM, "s",
                   {"candidate_type": ("relay", "host", "srflx")}),
    InstrumentSpec("therapy_webrtc_packet_loss_ratio", InstrumentKind.HISTOGRAM, "1",
                   {"candidate_type": ("relay", "host", "srflx")}),
    InstrumentSpec("therapy_webrtc_connection_total", InstrumentKind.COUNTER, "1",
                   {"candidate_type": ("relay", "host", "srflx"), "outcome": None}),
    # --- persistence/longitudinal ---
    InstrumentSpec("therapy_storage_operations_total", InstrumentKind.COUNTER, "1",
                   {"component": None, "operation": None, "outcome": None}),
    InstrumentSpec("therapy_storage_operation_seconds", InstrumentKind.HISTOGRAM, "s",
                   {"component": None, "operation": None, "outcome": None}),
    InstrumentSpec("therapy_sqlite_busy_total", InstrumentKind.COUNTER, "1",
                   {"component": None}),
    InstrumentSpec("therapy_sqlite_checkpoint_last_success_unixtime",
                   InstrumentKind.GAUGE, "s", {"component": None}),
    InstrumentSpec("therapy_sqlite_integrity_last_success_unixtime",
                   InstrumentKind.GAUGE, "s", {"component": None}),
    InstrumentSpec("therapy_sqlite_wal_bytes", InstrumentKind.GAUGE, "By",
                   {"component": None}),
    InstrumentSpec("therapy_schema_version", InstrumentKind.GAUGE, "1",
                   {"component": None}),
    InstrumentSpec("therapy_data_bytes", InstrumentKind.GAUGE, "By",
                   {"kind": ("db", "wal", "audio", "research", "model_cache", "backups")}),
    InstrumentSpec("therapy_session_finalizations_total", InstrumentKind.COUNTER, "1",
                   {"artifact": None, "outcome": None}),
    InstrumentSpec("therapy_session_finalization_seconds", InstrumentKind.HISTOGRAM,
                   "s", {"outcome": None}),
    InstrumentSpec("therapy_session_finalizers_pending", InstrumentKind.GAUGE, "1"),
    InstrumentSpec("therapy_context_assembly_seconds", InstrumentKind.HISTOGRAM, "s",
                   {"outcome": None}),
    InstrumentSpec("therapy_distillation_runs_total", InstrumentKind.COUNTER, "1",
                   {"outcome": None, "idempotent": ("true", "false")}),
    # --- research/proactivity ---
    InstrumentSpec("therapy_research_ingests_total", InstrumentKind.COUNTER, "1",
                   {"format": None, "outcome": None, "deduplicated": ("true", "false")}),
    InstrumentSpec("therapy_proactivity_scheduler_last_tick_unixtime",
                   InstrumentKind.GAUGE, "s"),
    # --- observer health ---
    InstrumentSpec("therapy_llm_capture_records_total", InstrumentKind.COUNTER, "1",
                   {"operation": None, "status": _CAPTURE_STATUS}),
    InstrumentSpec("therapy_llm_capture_append_seconds", InstrumentKind.HISTOGRAM, "s",
                   {"outcome": None}),
    InstrumentSpec("therapy_llm_capture_export_seconds", InstrumentKind.HISTOGRAM, "s",
                   {"outcome": None}),
    InstrumentSpec("therapy_llm_capture_oldest_unexported_unixtime",
                   InstrumentKind.GAUGE, "s"),
    InstrumentSpec("therapy_llm_capture_last_export_success_unixtime",
                   InstrumentKind.GAUGE, "s"),
    InstrumentSpec("therapy_event_loop_lag_seconds", InstrumentKind.HISTOGRAM, "s"),
    InstrumentSpec("therapy_executor_queue_wait_seconds", InstrumentKind.HISTOGRAM,
                   "s", {"workload_class": None}),
    InstrumentSpec("therapy_broad_span_drops_total", InstrumentKind.COUNTER, "1",
                   {"reason": ("unknown_scope", "forbidden_attribute", "queue_full")}),
    InstrumentSpec("therapy_client_events_total", InstrumentKind.COUNTER, "1",
                   {"name": None, "outcome": None}),
)

#: Fast lookup by name; duplicate names are a manifest bug caught by tests.
INSTRUMENT_INDEX: dict[str, InstrumentSpec] = {
    spec.name: spec for spec in INSTRUMENTS
}
