# Runbook — provider alerts (`therapy-providers` group)

## Symptom

A provider records three errors/timeouts or rate limits in ten minutes, or TTFA
p95 stays above twice its same-provider/mode one-hour baseline.

## Meaning and impact

The active provider is repeatedly unavailable, throttling, or regressing
relative to its own recent behavior. No absolute latency objective is inferred
before baseline evidence exists.

## Safe queries (content-free)

- `sum(increase(therapy_llm_requests_total[10m])) by (provider, operation, outcome)`
- `sum(increase(therapy_llm_rate_limits_total[10m])) by (provider, operation)`
- `histogram_quantile(0.95, sum(rate(therapy_turn_ttfa_seconds_bucket[5m])) by (le, provider, mode))`
- Dashboard 1 (latency) and dashboard 5 (§9 SLI row).

## First three commands

```bash
docker compose ps
docker compose logs --since 30m therapy | grep -E 'provider\.|llm\.|rate_limit'
curl -fsS http://localhost:8000/health
```

## Remediation

Confirm local network/model health, honor bounded retry/backoff, and stop retries
that amplify throttling. Compare cold/warm and provider/mode series before
attributing TTFA to the provider.

## Restart and rollback boundary

Do not restart for a remote rate limit. Roll back a provider/config change only
after preserving content-free counters and restricted evidence. A provider
switch changes behavior and requires the normal evaluation gate.

## State preservation warning

Never delete interaction journal rows or exact provider-error evidence. Keep all
payload inspection inside the restricted plane.

## Verification

- Error/rate-limit increments stop through the full alert window.
- TTFA returns below twice the same provider/mode rolling baseline.
- A synthetic attempt has complete journal evidence.
