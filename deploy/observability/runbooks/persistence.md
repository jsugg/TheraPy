# Runbook — persistence alerts (`therapy-persistence` group)

## Symptom

An instrumented storage operation fails, SQLite busy events exceed twenty in
ten minutes, a ten-minute checkpoint is more than twenty minutes old, or the
hourly integrity success is more than two hours old.

## Meaning and impact

An accepted mutation may lack durable state, or write contention may threaten
turn/finalizer deadlines. The alert is operational only and exposes no record
content, identifiers, or concrete paths.

## Safe queries (content-free)

- `sum(increase(therapy_storage_operations_total[10m])) by (component, operation, outcome)`
- `sum(increase(therapy_sqlite_busy_total[10m])) by (component)`
- `time() - therapy_sqlite_integrity_last_success_unixtime_seconds`
- `time() - therapy_sqlite_checkpoint_last_success_unixtime_seconds`
- Dashboard 2 (persistence and memory).

## First three commands

```bash
docker compose ps
docker compose logs --since 30m therapy | grep -E 'storage\.|sqlite\.|finalizer\.'
docker compose exec -T therapy sh -c 'df -h; df -i'
```

## Remediation

Stop the failing synthetic operation, check free space and permissions, and run
integrity checks. For contention, identify the bounded component/operation and
remove overlapping maintenance before changing SQLite timeouts.

## Restart and rollback boundary

Before restore/delete/migration rollback, secure failure evidence and take one
consistent DB/file snapshot. Restart only after owner-idle confirmation; do not
retry a non-idempotent mutation without its durable state check.

## State preservation warning

Never remove WAL/database files, finalizer rows, or backup evidence to clear an
alert. A snapshot must include the consistent DB and file set.

## Verification

- A disposable acknowledged mutation commits and reads back successfully.
- Integrity succeeds and no new storage error increments occur.
- Busy increments remain below threshold for thirty minutes.
