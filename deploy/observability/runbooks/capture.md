# Runbook — capture alerts (`therapy-capture` group)

## Impact

The restricted interaction plane is the evidence-of-record for every LLM
attempt. Journal pre-dispatch failures mean attempts run with visible gaps;
export backlog means Phoenix lags the journal (queries stale, nothing lost);
a routing violation means content or a forbidden key tried to reach the
broad plane — that is a release-gate bug, not an ops nuisance.

## Safe queries (content-free)

- `sum(rate(therapy_llm_capture_records_total[15m])) by (status)`
- `time() - therapy_llm_capture_oldest_unexported_unixtime`
- `sum(rate(therapy_broad_span_drops_total[5m])) by (reason)`
- Dashboard 6 (telemetry-health).

## First three commands

```bash
docker compose ps
docker compose logs --since 30m therapy | grep -E 'capture_degraded|capture_recovered'
docker compose exec -T therapy uv run python - <<'PY'
from therapy.observability.journal import JournalStore
from therapy.observability.config import ObservabilityConfig
s = JournalStore(ObservabilityConfig.from_env().journal_path)
print(s.health()); print("integrity:", s.integrity_check()); s.close()
PY
```

## Restart/rollback boundary

- Journal failures: check disk space and file permissions on the data
  volume FIRST. Restarting `therapy` re-runs recovery (nonterminal rows
  become explicit `incomplete`; nothing is invented).
- Export backlog: restart the `phoenix` service; the worker retries with
  backoff and records stay in the journal until ACK. Backend outage NEVER
  justifies disabling capture (rollback is journal-only, not capture-off).
- Routing violation: this is a code bug. Capture evidence (the span drop
  counter labels + recent broad logs), then disable broad export
  (`THERAPY_OTEL_ENABLED=0`) until the routing test that should have caught
  it is fixed. Do NOT disable the journal.

## State preservation warning

Never delete `interaction-journal.sqlite3*`, the collector WAL volume, or
the Phoenix volume before capturing diagnostics and verifying a consistent
snapshot (`sqlite3 .backup`, integrity check). Unacknowledged journal rows
ignore retention by design.

## Verification

- `therapy_llm_capture_records_total{status="failed"}` rate returns to 0.
- Oldest-unexported age falls below 5 minutes.
- `capture_recovered` event observed in broad logs.
