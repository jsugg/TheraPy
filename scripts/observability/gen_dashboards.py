"""Generate the six required Grafana dashboards (plan §9) deterministically.

Regenerates `deploy/observability/dashboards/*.json` byte-identically so a
reviewer can prove they came from code. Panels use the LGTM image's default
datasource UIDs (`prometheus`, `tempo`, `loki`). Buckets/thresholds start
from the O0 baseline and are tuned in O4 dogfood.

Usage: .venv/bin/python scripts/observability/gen_dashboards.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "deploy/observability/dashboards"

PROM = {"type": "prometheus", "uid": "prometheus"}


def timeseries(
    title: str,
    expr: str,
    *,
    unit: str = "s",
    grid: dict,
    description: str | None = None,
) -> dict:
    panel = {
        "type": "timeseries",
        "title": title,
        "datasource": PROM,
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
        "gridPos": grid,
        "targets": [{"expr": expr, "refId": "A", "datasource": PROM}],
    }
    if description is not None:
        panel["description"] = description
    return panel


def stat(
    title: str,
    expr: str,
    *,
    unit: str = "short",
    grid: dict,
    description: str | None = None,
) -> dict:
    panel = {
        "type": "stat",
        "title": title,
        "datasource": PROM,
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
        "gridPos": grid,
        "targets": [{"expr": expr, "refId": "A", "datasource": PROM}],
    }
    if description is not None:
        panel["description"] = description
    return panel


def row(title: str, *, grid: dict) -> dict:
    return {
        "type": "row",
        "title": title,
        "collapsed": False,
        "panels": [],
        "gridPos": grid,
    }


def dashboard(uid: str, title: str, panels: list[dict]) -> dict:
    return {
        "uid": uid,
        "title": title,
        "tags": ["therapy", "observability"],
        "timezone": "browser",
        "schemaVersion": 39,
        "refresh": "30s",
        "time": {"from": "now-6h", "to": "now"},
        "panels": panels,
        "templating": {"list": []},  # no product-ID variables, ever (plan §1)
        "annotations": {"list": []},
    }


def g(x: int, y: int, w: int = 12, h: int = 8) -> dict:
    return {"x": x, "y": y, "w": w, "h": h}


DASHBOARDS: dict[str, dict] = {
    "conversation-latency": dashboard(
        "therapy-conversation",
        "1 - Conversation path and latency waterfall",
        [
            timeseries(
                "Turn TTFA p50/p95 by provider",
                'histogram_quantile(0.95, sum(rate(therapy_turn_ttfa_seconds_bucket[5m])) by (le, provider))',
                grid=g(0, 0),
            ),
            timeseries(
                "Turn stage duration p95 by stage",
                'histogram_quantile(0.95, sum(rate(therapy_turn_stage_duration_seconds_bucket[5m])) by (le, stage))',
                grid=g(12, 0),
            ),
            timeseries(
                "LLM time-to-first-token p95",
                'histogram_quantile(0.95, sum(rate(therapy_llm_time_to_first_token_seconds_bucket[5m])) by (le, provider))',
                grid=g(0, 8),
            ),
            timeseries(
                "Turns by outcome",
                "sum(rate(therapy_conversation_turns_total[5m])) by (modality, outcome)",
                unit="ops",
                grid=g(12, 8),
            ),
            timeseries(
                "Barge-ins and stop latency p95",
                'histogram_quantile(0.95, sum(rate(therapy_barge_in_stop_seconds_bucket[5m])) by (le))',
                grid=g(0, 16),
            ),
            timeseries(
                "TTS synthesis / first-audio p95",
                'histogram_quantile(0.95, sum(rate(therapy_tts_time_to_first_audio_seconds_bucket[5m])) by (le, language_group))',
                grid=g(12, 16),
            ),
        ],
    ),
    "persistence-memory": dashboard(
        "therapy-persistence",
        "2 - Persistence, memory, finalization, context backlog",
        [
            timeseries(
                "Storage operation p95 by component",
                'histogram_quantile(0.95, sum(rate(therapy_storage_operation_seconds_bucket[5m])) by (le, component))',
                grid=g(0, 0),
            ),
            timeseries(
                "SQLite busy events",
                "sum(rate(therapy_sqlite_busy_total[5m])) by (component)",
                unit="ops",
                grid=g(12, 0),
            ),
            stat(
                "Pending session finalizers",
                "therapy_session_finalizers_pending",
                grid=g(0, 8, 6),
            ),
            stat(
                "Last journal checkpoint age (s)",
                'time() - max(therapy_sqlite_checkpoint_last_success_unixtime_seconds)',
                unit="s",
                grid=g(6, 8, 6),
            ),
            timeseries(
                "Finalizations by artifact/outcome",
                "sum(rate(therapy_session_finalizations_total[15m])) by (artifact, outcome)",
                unit="ops",
                grid=g(12, 8),
            ),
            timeseries(
                "Context assembly p95",
                'histogram_quantile(0.95, sum(rate(therapy_context_assembly_seconds_bucket[5m])) by (le))',
                grid=g(0, 16),
            ),
            timeseries(
                "Data bytes by kind",
                "therapy_data_bytes",
                unit="bytes",
                grid=g(12, 16),
            ),
        ],
    ),
    "research-pipeline": dashboard(
        "therapy-research",
        "3 - Research ingest, OCR, embedding, index, query",
        [
            timeseries(
                "Ingests by format/outcome",
                "sum(rate(therapy_research_ingests_total[15m])) by (format, outcome)",
                unit="ops",
                grid=g(0, 0),
            ),
            timeseries(
                "Storage ops p95 (research component)",
                'histogram_quantile(0.95, sum(rate(therapy_storage_operation_seconds_bucket{component="research"}[5m])) by (le, operation))',
                grid=g(12, 0),
            ),
            timeseries(
                "Research stage duration p95",
                "histogram_quantile(0.95, sum(rate(therapy_research_stage_seconds_bucket[5m])) by (le, stage, outcome))",
                grid=g(0, 8),
            ),
            timeseries(
                "Research queries by outcome",
                "sum(rate(therapy_research_queries_total[15m])) by (outcome)",
                unit="ops",
                grid=g(12, 8),
            ),
            timeseries(
                "Index rebuilds by outcome",
                "sum(rate(therapy_research_reindex_total[15m])) by (outcome)",
                unit="ops",
                grid=g(0, 16),
            ),
            timeseries(
                "Research divergence by kind",
                "sum(rate(therapy_research_divergence_total[15m])) by (kind)",
                unit="ops",
                grid=g(12, 16),
            ),
            timeseries(
                "OCR runs and duration p95",
                "histogram_quantile(0.95, sum(rate(therapy_ocr_seconds_bucket[15m])) by (le, outcome))",
                grid=g(0, 24),
            ),
            timeseries(
                "OCR runs by outcome",
                "sum(rate(therapy_ocr_runs_total[15m])) by (outcome)",
                unit="ops",
                grid=g(12, 24),
            ),
            timeseries(
                "Embedding batches by cache/outcome",
                "sum(rate(therapy_embedding_batches_total[15m])) by (cache, outcome)",
                unit="ops",
                grid=g(0, 32),
            ),
            timeseries(
                "Embedding duration p95 by cache",
                "histogram_quantile(0.95, sum(rate(therapy_embedding_seconds_bucket[15m])) by (le, cache))",
                grid=g(12, 32),
            ),
        ],
    ),
    "proactivity-push": dashboard(
        "therapy-proactivity",
        "4 - Proactivity and push (no content, no engagement)",
        [
            stat(
                "Scheduler last tick age (s)",
                "time() - therapy_proactivity_scheduler_last_tick_unixtime_seconds",
                unit="s",
                grid=g(0, 0, 6),
            ),
            timeseries(
                "Storage ops (proactivity component)",
                'sum(rate(therapy_storage_operations_total{component="proactivity"}[15m])) by (operation, outcome)',
                unit="ops",
                grid=g(6, 0, 18),
            ),
            timeseries(
                "Scheduler ticks by outcome",
                "sum(rate(therapy_proactivity_ticks_total[15m])) by (outcome)",
                unit="ops",
                grid=g(0, 8),
            ),
            timeseries(
                "Scheduler tick duration p95",
                "histogram_quantile(0.95, sum(rate(therapy_proactivity_tick_seconds_bucket[15m])) by (le, outcome))",
                grid=g(12, 8),
            ),
            timeseries(
                "Proactivity jobs by stage/channel",
                "sum(rate(therapy_proactivity_jobs_total[15m])) by (stage, channel)",
                unit="ops",
                grid=g(0, 16),
            ),
            stat(
                "Oldest due enabled job age",
                "therapy_proactivity_oldest_due_age_seconds",
                unit="s",
                grid=g(12, 16),
            ),
            timeseries(
                "Web Push deliveries by status class",
                "sum(rate(therapy_webpush_deliveries_total[15m])) by (status_class)",
                unit="ops",
                grid=g(0, 24),
            ),
            timeseries(
                "Web Push subscription events",
                "sum(rate(therapy_webpush_subscription_events_total[15m])) by (event)",
                unit="ops",
                grid=g(12, 24),
            ),
        ],
    ),
    "reliability": dashboard(
        "therapy-reliability",
        "5 - Reliability: liveness, readiness, relay, resources, supervisors",
        [
            timeseries(
                "Process CPU utilization",
                'process_cpu_utilization{service_name="therapy"}',
                unit="percentunit",
                grid=g(0, 0),
            ),
            timeseries(
                "Process RSS",
                'process_memory_usage{service_name="therapy"}',
                unit="bytes",
                grid=g(12, 0),
            ),
            timeseries(
                "Voice connections by outcome",
                "sum(rate(therapy_voice_connections_total[15m])) by (outcome, reason)",
                unit="ops",
                grid=g(0, 8),
            ),
            stat(
                "Active voice connections",
                "therapy_voice_active_connections",
                grid=g(12, 8, 6),
            ),
            timeseries(
                "Event-loop lag p95",
                'histogram_quantile(0.95, sum(rate(therapy_event_loop_lag_seconds_bucket[5m])) by (le))',
                grid=g(0, 16),
            ),
            timeseries(
                "HTTP server duration p95 by route",
                'histogram_quantile(0.95, sum(rate(http_server_duration_milliseconds_bucket{service_name="therapy"}[5m])) by (le, http_route))',
                unit="ms",
                grid=g(12, 16),
            ),
            # Relay: the collector scrapes coturn's internal 9641; the STUN
            # healthcheck doubles as a synthetic binding probe, so these
            # series carry data even with no active call. turn_* allocation
            # counters appear lazily on the first real relay allocation.
            stat(
                "TURN relay up",
                'up{job="turn"}',
                grid=g(0, 24, 6),
            ),
            timeseries(
                "STUN bindings (incl. synthetic healthcheck probe)",
                'sum by (__name__) ({__name__=~"stun_binding_(request|response)_total", job="turn"})',
                unit="ops",
                grid=g(6, 24, 9),
            ),
            timeseries(
                "TURN allocations/traffic (populates on first relay use)",
                'sum by (__name__) ({__name__=~"turn_.+", job="turn"})',
                unit="ops",
                grid=g(15, 24, 9),
            ),
            # Supervision: restarts show as uptime sawtooth for both the app
            # process and the relay (watchdog/hostwatch verification records
            # are file-based and have no metrics path yet — named open item).
            timeseries(
                "Process uptime (restarts visible as resets)",
                "time() - process_start_time_seconds",
                unit="s",
                grid=g(0, 32, 24),
            ),
            row("§9 service-level indicators", grid=g(0, 40, 24, 1)),
            stat(
                "SLI: external liveness/readiness success ratio",
                'sum(rate(http_server_duration_milliseconds_count{service_name="therapy", http_route=~"/health|/ready", http_status_code=~"2.."}[30m])) / clamp_min(sum(rate(http_server_duration_milliseconds_count{service_name="therapy", http_route=~"/health|/ready"}[30m])), 0.000001)',
                unit="percentunit",
                grid=g(0, 41),
                description=(
                    "Definition: successful external /health and /ready probes divided "
                    "by all such probes, evaluated only inside the owner-declared service "
                    "window; deliberate host sleep/shutdown is excluded. Target: numeric "
                    "objective unset until hostwatch service-window evidence exists."
                ),
            ),
            stat(
                "SLI: voice connection success ratio",
                'sum(rate(therapy_webrtc_connection_total{outcome="connected"}[30m])) / clamp_min(sum(rate(therapy_offers_total{outcome!="rejected"}[30m])), 0.000001)',
                unit="percentunit",
                grid=g(12, 41),
                description=(
                    "Definition: client-confirmed connected states after successful SDP/"
                    "pipeline start divided by all syntactically valid offers; rejected "
                    "offers are excluded. Target: numeric objective unset pending baseline."
                ),
            ),
            stat(
                "SLI: turn completion ratio",
                '((sum(rate(therapy_persist_total{artifact="assistant", outcome="success"}[30m])) or vector(0)) + (sum(rate(therapy_conversation_turns_total{outcome="classified_failure"}[30m])) or vector(0))) / clamp_min(sum(rate(therapy_persist_total{artifact="user", outcome="success"}[30m])), 0.000001)',
                unit="percentunit",
                grid=g(0, 49),
                description=(
                    "Definition: persisted assistant responses plus explicit classified "
                    "failures divided by accepted, durably persisted user turns. Target: "
                    "numeric objective unset pending baseline; no accepted turn may vanish."
                ),
            ),
            timeseries(
                "SLI: TTFA p95 / same-provider rolling baseline",
                "histogram_quantile(0.95, sum(rate(therapy_turn_ttfa_seconds_bucket[5m])) by (le, provider, mode)) / clamp_min(histogram_quantile(0.95, sum(rate(therapy_turn_ttfa_seconds_bucket[1h])) by (le, provider, mode)), 0.001)",
                unit="short",
                grid=g(12, 49),
                description=(
                    "Definition: five-minute TTFA p95 divided by the one-hour p95 for the "
                    "same provider and cold/warm mode; stage panels provide STT/context/LLM/"
                    "TTS/network decomposition. Target: <= 2x rolling baseline; no absolute "
                    "latency objective until baseline evidence supports one."
                ),
            ),
            stat(
                "SLI: acknowledged persistence success ratio",
                'sum(rate(therapy_persist_total{outcome="success"}[30m])) / clamp_min(sum(rate(therapy_persist_total[30m])), 0.000001)',
                unit="percentunit",
                grid=g(0, 57),
                description=(
                    "Definition: acknowledged persisted artifacts committed successfully "
                    "divided by all acknowledged persistence attempts. Target: 100%; "
                    "integrity, backup, and rollback failures must remain zero."
                ),
            ),
            stat(
                "SLI: integrity/backup/rollback failures",
                '(sum(rate(therapy_schema_migrations_total{outcome="error"}[30m])) or vector(0)) + (sum(rate(therapy_backups_total{outcome="error"}[30m])) or vector(0)) + (sum(rate(therapy_sovereignty_rollbacks_total{outcome="error"}[30m])) or vector(0))',
                unit="ops",
                grid=g(12, 57),
                description=(
                    "Definition: migration, backup, and sovereignty rollback error rate; "
                    "periodic isolated export/restore verification is companion evidence. "
                    "Target: exactly zero."
                ),
            ),
            stat(
                "SLI: capture journaled-vs-failed ratio",
                'sum(rate(therapy_llm_capture_records_total{status="journaled"}[30m])) / clamp_min(sum(rate(therapy_llm_capture_records_total{status=~"journaled|failed"}[30m])), 0.000001)',
                unit="percentunit",
                grid=g(0, 65),
                description=(
                    "Definition: initiated LLM attempts durably journaled before dispatch "
                    "divided by journaled plus pre-dispatch failed attempts. Target: 100%."
                ),
            ),
            stat(
                "SLI: evaluation evidence completeness ratio",
                '1 - (sum(rate(therapy_llm_capture_records_total{status="incomplete"}[30m])) / clamp_min(sum(rate(therapy_llm_capture_records_total{status="journaled"}[30m])), 0.000001))',
                unit="percentunit",
                grid=g(12, 65),
                description=(
                    "Definition: journaled LLM attempts without explicit incomplete evidence "
                    "divided by all journaled attempts; terminal completions and exact provider "
                    "errors are complete. Target: 100%, with zero incomplete accepted evaluations."
                ),
            ),
            stat(
                "SLI: restricted export backlog age",
                "time() - therapy_llm_capture_oldest_unexported_unixtime_seconds",
                unit="s",
                grid=g(0, 73),
                description=(
                    "Definition: current time minus the oldest unexported restricted-record "
                    "timestamp. Target: < 3600 seconds; the journal remains authoritative."
                ),
            ),
            stat(
                "SLI: proactivity terminal-state ratio",
                'sum(rate(therapy_proactivity_jobs_total{stage=~"delivered|retry|finalized"}[30m])) / clamp_min(sum(rate(therapy_proactivity_jobs_total{stage="due"}[30m])), 0.000001)',
                unit="percentunit",
                grid=g(12, 73),
                description=(
                    "Definition: enabled, unsuppressed due jobs reaching delivered, retry, or "
                    "terminal finalized state divided by due jobs. Target: 100% inside the "
                    "owner-agreed tolerance; the numeric tolerance is not set yet."
                ),
            ),
            stat(
                "SLI: broad-plane routing violations",
                'sum(rate(therapy_broad_span_drops_total{reason=~"forbidden_attribute|unknown_scope"}[30m]))',
                unit="ops",
                grid=g(0, 81, 24),
                description=(
                    "Definition: forbidden content/infrastructure canary or unknown-scope "
                    "attempts reaching the broad-plane guard; restricted-canary presence is "
                    "verified separately. Target: exactly zero; this is a hard release gate."
                ),
            ),
        ],
    ),
    "telemetry-health": dashboard(
        "therapy-telemetry",
        "6 - Telemetry journal, exporter, collector, cardinality",
        [
            timeseries(
                "Capture records by status",
                "sum(rate(therapy_llm_capture_records_total[15m])) by (operation, status)",
                unit="ops",
                grid=g(0, 0),
            ),
            stat(
                "Oldest unexported interaction age (s)",
                "time() - therapy_llm_capture_oldest_unexported_unixtime_seconds",
                unit="s",
                grid=g(12, 0, 6),
            ),
            stat(
                "Last successful export age (s)",
                "time() - therapy_llm_capture_last_export_success_unixtime_seconds",
                unit="s",
                grid=g(18, 0, 6),
            ),
            timeseries(
                "Broad span drops by reason (routing violations)",
                "sum(rate(therapy_broad_span_drops_total[5m])) by (reason)",
                unit="ops",
                grid=g(0, 8),
            ),
            timeseries(
                "Journal append/export p95",
                'histogram_quantile(0.95, sum(rate(therapy_llm_capture_append_seconds_bucket[5m])) by (le))',
                grid=g(12, 8),
            ),
            timeseries(
                "Active series (cardinality watch)",
                'count({__name__=~"therapy_.+"}) by (__name__)',
                unit="short",
                grid=g(0, 16, 24),
            ),
        ],
    ),
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, payload in DASHBOARDS.items():
        path = OUT / f"{name}.json"
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
