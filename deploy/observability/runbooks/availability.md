# Runbook — availability alerts (`therapy-availability` group)

## Symptom

- Three voice negotiations or pipeline starts fail inside five minutes.
- The scheduler heartbeat exceeds twice its interval, or three ticks fail.
- The TURN target is down for two consecutive evaluations.
- Active pipeline and connection-owner gauges disagree for five minutes.

## Meaning and impact

Voice failures block the primary path. Scheduler failures delay durable jobs;
they do not delete them. TURN failure prevents relay-required clients from
connecting while direct candidates may still work.

## Safe queries (content-free)

- `sum(increase(therapy_voice_connections_total[5m])) by (outcome, reason)`
- `sum(increase(therapy_voice_pipeline_transitions_total[5m])) by (transition, outcome)`
- `time() - therapy_proactivity_scheduler_last_tick_unixtime_seconds`
- `sum(increase(therapy_proactivity_ticks_total[5m])) by (outcome)`
- `up{job="turn"}`
- `abs(therapy_voice_pipeline_active - therapy_voice_active_connections)`
- Dashboard 5 (reliability).

## First three commands

```bash
docker compose ps
curl -fsS http://localhost:8000/health && curl -fsS http://localhost:8000/ready
docker compose logs --since 30m therapy turn | grep -E 'voice\.|scheduler\.|relay\.'
```

## Remediation

- Healthy HTTP with voice failures: check TURN reachability and provider health,
  then test one synthetic offer without owner data.
- Scheduler: correct the classified dependency failure; persisted jobs resume on
  the next successful tick.
- TURN: verify scrape health and the content-free STUN synthetic before changing
  the relay service.

## Restart and rollback boundary

Restarting `therapy` preempts a live session; confirm the owner is idle first.
Scheduler startup returns interrupted jobs to retry. Roll back only the most
recent runtime/config change after capturing bounded diagnostics. Do not restart
or recreate TURN merely to clear an alert without confirming relay failure.

## State preservation warning

Never clear scheduler/job tables or volumes. Secure failure evidence before any
rollback that changes persistent state.

## Verification

- A synthetic offer connects and pipeline-start failures stop increasing.
- Scheduler age is below twice its interval and error increments stop.
- `up{job="turn"}` remains `1` for two evaluations and STUN bindings advance.
- Active pipeline and owner gauges agree after synthetic connect/disconnect.
