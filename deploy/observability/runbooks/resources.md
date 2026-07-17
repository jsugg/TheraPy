# Runbook — resource alerts (`therapy-resources` group)

## Symptom

- Event-loop lag p95 exceeds 500 ms for ten minutes.
- The process restarts twice in fifteen minutes.
- Projected one-hour RSS exceeds 4.5 GB under the 5 GB dev-stack limit.

## Meaning and impact

Loop lag degrades voice latency first. Repeated restarts can indicate OOM or a
supervision loop. Sustained RSS growth risks a protective container restart.

## Safe queries (content-free)

- `histogram_quantile(0.95, sum(rate(therapy_event_loop_lag_seconds_bucket[5m])) by (le))`
- `changes(process_start_time_seconds{service_name="therapy"}[15m])`
- `predict_linear(process_memory_usage{service_name="therapy"}[30m], 3600)`
- Dashboard 5 (reliability).

## First three commands

```bash
docker stats --no-stream
docker compose ps
docker compose logs --since 30m therapy | grep -E 'recovery\.|watchdog\.|resource\.'
```

## Remediation

Bound or defer the classified workload causing lag. For memory growth, capture a
content-free process profile and allocation totals before reducing concurrency.
Confirm the restart reason before changing the memory limit.

## Restart and rollback boundary

Restart only after evidence capture and owner-idle confirmation. Roll back the
smallest workload/runtime change. Raising the 5 GB limit is not remediation
without a measured capacity decision.

## State preservation warning

A restart uses bounded finalizer drain. Never delete volumes or pending records
to relieve memory pressure.

## Verification

- Lag remains below 500 ms for thirty minutes.
- No process-start change occurs for thirty minutes.
- RSS projection stays below 4.5 GB under a representative synthetic workload.

