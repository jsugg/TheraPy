# Runbook — resource alerts (`therapy-resources` group)

## Impact

Event-loop lag degrades voice latency (TTFA) before anything else visibly
fails. Memory growth ends in the compose OOM-kill + restart (by design, to
protect the Docker VM).

## Safe queries (content-free)

- `histogram_quantile(0.95, sum(rate(therapy_event_loop_lag_seconds_bucket[5m])) by (le))`
- `process_memory_usage{service_name="therapy"}`
- `process_cpu_utilization{service_name="therapy"}`
- Dashboard 5 (reliability).

## First three commands

```bash
docker stats --no-stream
docker compose logs --since 30m therapy | grep -E 'WARNING|ERROR'
docker compose exec -T therapy ps aux --sort=-rss | head
```

## Restart/rollback boundary

- Sustained lag during STT/TTS load on this Intel host is expected to a
  degree (see the O0 baseline); alert thresholds assume steady state.
- If a leak is suspected, capture `py-spy dump` evidence BEFORE restarting
  (on demand only — never continuous, plan §1 non-goals).
- OOM-kill loops: lower workload before raising `mem_limit`; the cap
  protects the whole Docker VM (hypervisor stalls observed 2026-07-10).

## State preservation warning

A restart drains finalizers with a bounded timeout; anything undelivered
recovers from the journal/job tables. Never delete volumes to free memory.

## Verification

- Lag p95 back under threshold for 30 minutes; no OOM restarts in
  `docker compose ps`/`docker inspect`.
