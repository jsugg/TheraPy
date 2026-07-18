# Runbook — capture alerts (`therapy-capture` group)

## Symptom

- Pre-dispatch journal failures, explicit incomplete evidence, or export age over
  one hour.
- The broad-plane guard drops a forbidden attribute or unknown scope.

## Meaning and impact

The restricted journal is the evidence-of-record for every LLM attempt. A
journal failure or incomplete row invalidates evaluation evidence. Export lag
only makes the backend stale while the journal stays authoritative. A routing
violation is a hard release-gate failure.

## Safe queries (content-free)

- `sum(increase(therapy_llm_capture_records_total[15m])) by (status)`
- `time() - therapy_llm_capture_oldest_unexported_unixtime_seconds`
- `sum(increase(therapy_broad_span_drops_total[5m])) by (reason)`
- Dashboard 6 (telemetry health) and dashboard 5 (§9 SLI row).

## First three commands

```bash
docker compose ps
docker compose logs --since 30m therapy | grep -E 'capture_degraded|capture_recovered|broad_span_dropped'
docker compose exec -T therapy uv run python - <<'PY'
from therapy.observability.config import ObservabilityConfig
from therapy.observability.journal import JournalStore

store = JournalStore(ObservabilityConfig.from_env().journal_path)
print(store.health())
print("integrity:", store.integrity_check())
store.close()
PY
```

## Remediation

- Journal failure: check free space and owner-only permissions, then run the
  journal integrity check before resuming provider dispatch.
- Incomplete evidence: preserve the row and correlate only by safe trace/span
  identifiers in the restricted plane; rerun evaluation only after repair.
- Export lag: restore backend reachability; the worker retries acknowledged rows.
- Routing violation: preserve guard counters, disable broad export, and fix the
  routing test. Never disable the restricted journal.

## Restart and rollback boundary

Restarting marks nonterminal rows explicitly incomplete; it never invents a
completion. Roll back exporter changes independently of capture. For a routing
bug, broad export may remain disabled until the canary gate passes.

## State preservation warning

Never delete journal, collector, or backend volumes. Take a consistent snapshot
and secure failure evidence before any persistent rollback.

## Verification

- Failed/incomplete rates return to zero and the journal integrity check passes.
- Oldest-unexported age falls below five minutes.
- Broad routing violations remain zero through a synthetic canary check.
