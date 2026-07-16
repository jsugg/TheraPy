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


def timeseries(title: str, expr: str, *, unit: str = "s", grid: dict) -> dict:
    return {
        "type": "timeseries",
        "title": title,
        "datasource": PROM,
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
        "gridPos": grid,
        "targets": [{"expr": expr, "refId": "A", "datasource": PROM}],
    }


def stat(title: str, expr: str, *, unit: str = "short", grid: dict) -> dict:
    return {
        "type": "stat",
        "title": title,
        "datasource": PROM,
        "fieldConfig": {"defaults": {"unit": unit}, "overrides": []},
        "gridPos": grid,
        "targets": [{"expr": expr, "refId": "A", "datasource": PROM}],
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
                'sum(rate({__name__=~"stun_binding_(request|response)_total", job="turn"}[5m])) by (__name__)',
                unit="ops",
                grid=g(6, 24, 9),
            ),
            timeseries(
                "TURN allocations/traffic (populates on first relay use)",
                'sum(rate({__name__=~"turn_.+", job="turn"}[5m])) by (__name__)',
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
