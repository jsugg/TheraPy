# Runbook — availability alerts (`therapy-availability` group)

## Impact

Voice negotiation failures block the product's primary path. A stale
scheduler heartbeat delays proactivity jobs (no data loss: jobs persist and
deliver on recovery).

## Safe queries (content-free)

- `sum(rate(therapy_voice_connections_total[15m])) by (outcome, reason)`
- `time() - therapy_proactivity_scheduler_last_tick_unixtime`
- Dashboard 5 (reliability).

## First three commands

```bash
docker compose ps
curl -s http://localhost:8000/health
docker compose logs --since 30m therapy | grep -E 'voice\.|scheduler\.'
```

## Restart/rollback boundary

- Negotiation failures with healthy `/health`: check the `turn` service and
  host Ollama first (`docker compose logs turn`, `curl :11434/api/tags`).
  Restarting `therapy` preempts any live session — confirm the owner is not
  mid-conversation before restarting.
- Scheduler: restart recovers cleanly; `processing` jobs return to `retry`
  on startup by design.

## State preservation warning

Do not clear the proactivity job tables to "unstick" delivery — jobs are
the durable owner-visible record; the scheduler self-heals.

## Verification

- New `voice.session_resumed`/turn events after reconnect.
- Scheduler tick age under 2x interval.
