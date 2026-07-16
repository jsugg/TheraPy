"""Langfuse Cloud leg of the O0.3 backend acceptance spike (audit F-04).

Runs the identical checked-in fixture corpus against Langfuse Cloud using an
owner-scoped test key (LANGFUSE_{PUBLIC,SECRET}_KEY / LANGFUSE_BASE_URL from
`.env`). Synthetic data only; the remote-egress review is the owner's
explicit key provision for exactly this measurement.

Reuses the pinned span construction from `backend_spike.py` so all three
candidates see byte-identical OpenInference spans, exports over OTLP HTTP
with Basic auth, then measures: export latency, queryability, canonical
envelope round-trip through the public API, duplicate-resend behavior,
canary presence via API reads (remote storage cannot be scanned directly —
recorded as an explicit measurement limit), and query latency.

Usage (run from the spike venv that has the OTel pins):
    .local/obs-spike/phoenix-venv/bin/python \
        scripts/observability/backend_spike_langfuse.py \
        --output .local/obs-spike/results/langfuse.json
"""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()
)
sys.path.insert(0, str(REPO_ROOT))

from scripts.observability.backend_spike import (  # noqa: E402
    FIXTURE_ROOT,
    _canonical_json,
    _export,
    _fixture_hash,
    build_spans,
    load_fixtures,
)


def _env() -> dict[str, str]:
    values: dict[str, str] = {}
    env_path = REPO_ROOT / ".env"
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, raw = line.partition("=")
            values[key.strip()] = raw.strip().strip('"').strip("'")
    return values


def _api(base: str, auth: str, path: str, params: dict | None = None) -> dict:
    """One public-API GET with bounded 429 backoff (Hobby-tier rate limits)."""
    url = f"{base}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(
        url, headers={"Authorization": f"Basic {auth}"}
    )
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempt == 5:
                raise
            retry_after = error.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else 10.0 * (attempt + 1)
            time.sleep(min(delay, 60.0))
    raise RuntimeError("unreachable")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    env = _env()
    base = env.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com").rstrip("/")
    public_key = env["LANGFUSE_PUBLIC_KEY"]
    secret_key = env["LANGFUSE_SECRET_KEY"]
    auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    otlp_endpoint = f"{base}/api/public/otel/v1/traces"
    headers = {"Authorization": f"Basic {auth}"}

    fixtures = load_fixtures()
    spans = build_spans(fixtures)
    canaries = json.loads((FIXTURE_ROOT / "canaries.json").read_text())
    content_canaries: dict[str, str] = canaries["content"]
    forbidden_canaries: dict[str, str] = canaries["forbidden"]

    result: dict[str, object] = {
        "backend": "langfuse-cloud",
        "endpoint": otlp_endpoint.replace(base, "<LANGFUSE_BASE_URL>"),
        "fixture_count": len(fixtures),
        "fixture_sha256": _fixture_hash(),
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tier_facts": {
            "plan": "Hobby (verified 2026-07-15)",
            "included_units_per_month": 50000,
            "data_access_days": 30,
        },
    }

    first_result, first_ms = _export(spans, otlp_endpoint, headers)
    result["measurements"] = {"first_export_wall_ms": first_ms}
    result["checks"] = {
        "first_export": {"passed": first_result == "SUCCESS", "actual": first_result}
    }
    checks: dict[str, dict[str, object]] = result["checks"]  # type: ignore[assignment]
    measurements: dict[str, object] = result["measurements"]  # type: ignore[assignment]

    # ingestion is asynchronous; poll until every trace is queryable
    expected_trace_ids = {
        f"{span.context.trace_id:032x}" for span in spans
    }
    found: set[str] = set()
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline and found != expected_trace_ids:
        page = _api(base, auth, "/api/public/traces", {"limit": 50})
        found = {
            trace["id"] for trace in page.get("data", [])
        } & expected_trace_ids
        if found != expected_trace_ids:
            time.sleep(5)
    checks["all_records_queryable"] = {
        "passed": found == expected_trace_ids,
        "missing": sorted(expected_trace_ids - found),
    }

    # canonical envelope round-trip + API-level canary scan
    latencies: list[float] = []
    envelope_losses: list[str] = []
    api_text_parts: list[str] = []
    for span in spans:
        trace_id = f"{span.context.trace_id:032x}"
        time.sleep(1.5)  # stay under Hobby-tier API rate limits
        started = time.perf_counter()
        try:
            detail = _api(base, auth, f"/api/public/traces/{trace_id}")
        except Exception as exc:
            envelope_losses.append(f"{trace_id}: fetch {type(exc).__name__}")
            continue
        latencies.append((time.perf_counter() - started) * 1000)
        text = json.dumps(detail, ensure_ascii=False)
        api_text_parts.append(text)
        expected = span.attributes.get("therapy.canonical_record")
        if isinstance(expected, str):
            observations = detail.get("observations", [])
            stored = None
            for obs in observations:
                meta = obs.get("metadata") or {}
                attrs = (
                    meta.get("attributes") if isinstance(meta, dict) else None
                ) or {}
                if "therapy.canonical_record" in attrs:
                    stored = attrs["therapy.canonical_record"]
                    break
            if stored is None and "therapy.canonical_record" in text:
                # present somewhere in the trace payload; verify exact bytes
                stored = expected if expected in text else None
            if stored is None:
                envelope_losses.append(f"{trace_id}: canonical envelope missing")
            elif isinstance(stored, str) and stored != expected:
                if _canonical_json(json.loads(stored)) != expected:
                    envelope_losses.append(f"{trace_id}: envelope transformed")
    checks["canonical_envelope_round_trip"] = {
        "passed": not envelope_losses,
        "losses": envelope_losses[:20],
    }
    if latencies:
        ordered = sorted(latencies)
        measurements["query_p50_ms"] = statistics.median(ordered)
        measurements["query_p95_ms"] = ordered[
            min(len(ordered) - 1, int(len(ordered) * 0.95))
        ]

    api_text = "\n".join(api_text_parts)
    content_counts = {
        name: api_text.count(value) for name, value in content_canaries.items()
    }
    forbidden_counts = {
        name: api_text.count(value) for name, value in forbidden_canaries.items()
    }
    checks["content_canaries_present"] = {
        "passed": all(count > 0 for count in content_counts.values()),
        "counts": content_counts,
    }
    checks["forbidden_canaries_absent"] = {
        "passed": all(count == 0 for count in forbidden_counts.values()),
        "counts": forbidden_counts,
        "note": "API-level scan only; remote storage cannot be inspected",
    }

    # duplicate resend: identical spans again; count traces afterwards
    duplicate_result, duplicate_ms = _export(spans, otlp_endpoint, headers)
    time.sleep(10)
    page = _api(base, auth, "/api/public/traces", {"limit": 50})
    after = {
        trace["id"] for trace in page.get("data", [])
    } & expected_trace_ids
    measurements["duplicate_export_wall_ms"] = duplicate_ms
    checks["duplicate_probe_observed"] = {
        "passed": duplicate_result == "SUCCESS" and after == expected_trace_ids,
        "behavior": "accepted_no_new_trace_ids"
        if after == expected_trace_ids
        else "trace_set_changed",
    }

    result["overall_pass"] = all(
        bool(check["passed"]) for check in checks.values()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "backend": "langfuse-cloud",
                "overall_pass": result["overall_pass"],
                "output": str(args.output),
            }
        )
    )
    return 0 if result["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
