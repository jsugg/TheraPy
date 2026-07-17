# §9 alert firing/recovery drill

**Status:** BLOCKED before injection; no pending/firing/recovered transition was
claimed.

## Selected alert and synthetic condition

- Alert: `scheduler-heartbeat-stale` — Proactivity scheduler heartbeat older
  than 2x interval.
- Intended injection: an OTLP gauge named
  `therapy_proactivity_scheduler_last_tick_unixtime`, with a fixed synthetic
  service/resource identity and value `time() - 3600`, followed by a current
  timestamp using the same identity for recovery.
- Safety: the payload contained only bounded synthetic labels and a timestamp;
  it did not read or mutate product data.

## Attempt and blocker

At `2026-07-17T04:44:53Z`, Grafana was healthy on
`http://localhost:3000`, but the documented broad OTLP endpoint was not
published on the host:

```text
curl: (7) Failed to connect to localhost port 4318 after 0 ms: Couldn't connect to server
collector_http=000 error=Failed to connect to localhost port 4318 after 0 ms: Couldn't connect to server
```

The pinned `.venv` OpenTelemetry SDK attempted
`http://localhost:4318/v1/metrics`; all retries ended in connection refused.
`docker port therapy-collector-1` returned no mapping, and
`docker port therapy-lgtm-1` exposed only `127.0.0.1:3000`. A subsequent
Prometheus query for the fixed synthetic service returned an empty vector, so
no injected series reached the stack.

The running Grafana also still contained only the six rules loaded at container
startup. Its rules API reported `scheduler-heartbeat-stale` as `NoData`; two
rules with data were `Alerting (Error)` because the startup-loaded definitions
used range results directly in thresholds. The updated worktree provisioning now
marks every Prometheus query `instant: true` and `range: false`, but applying
that file to the already-running Grafana would require a provisioning reload or
container restart. Both Grafana mutation and container lifecycle changes were
outside this drill's read-only constraints.

## State timestamps

| State | Timestamp | Evidence |
|---|---|---|
| pending | not reached | OTLP endpoint refused the synthetic batch |
| firing | not reached | no synthetic series existed; live rule remained `NoData` |
| recovered | not applicable | no firing state or accepted series existed |

## Cleanup

No telemetry batch was accepted, no synthetic series existed to clear, no
product data was touched, and no container was restarted, recreated, or
stopped. Re-run this drill after the collector's host `4318` mapping and the
updated alert provisioning are active; continuously export the stale gauge
through the five-minute pending window, then export the current timestamp until
the rules API records recovery.
