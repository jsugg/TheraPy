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

---

## Completed drill (2026-07-17, container-side)

The blocker above was an access-path issue, not a stack defect: the collector's
OTLP port is compose-internal BY DESIGN (never published to the host), so the
drill must run from inside the compose network; and Grafana's file-provisioned
alert rules reload through its admin API without any container mutation.

**Procedure (all from inside the `therapy` container; zero restarts):**

1. `POST http://lgtm:3000/api/admin/provisioning/alerting/reload` →
   `200 {"message":"Alerting config reloaded"}`; the provisioning API then
   listed all 20 catalog rules.
2. Selected alert: `routing-violation` ("Broad routing violation", `for: 0m`).
3. Injection: an OTLP/HTTP JSON cumulative counter
   `therapy_broad_span_drops_total{reason="unknown_scope"}` from the clearly
   synthetic resource `service.name="therapy-drill"` (never the live service
   identity), posted to `http://collector:4318/v1/metrics` — five monotonic
   increments (1→30) over 100 s. Content-free by construction.
4. Prometheus confirmed the condition:
   `sum(rate(therapy_broad_span_drops_total{reason=~"forbidden_attribute|unknown_scope"}[5m]))`
   ≈ 2.97 > 0.

**State timestamps (UTC, from the Grafana rules API):**

| State | Timestamp |
|---|---|
| injection start | 2026-07-17T04:55:05Z |
| injection end | 2026-07-17T04:56:46Z |
| firing observed | 2026-07-17T04:58:01Z |
| recovered (inactive) | 2026-07-17T05:04:45Z |

Recovery required no cleanup action: injection stopped and the 5-minute rate
window drained naturally. The synthetic series remains in the TSDB under the
`therapy-drill` service identity (bounded, content-free, distinguishable from
live dogfood data) and ages out with LGTM's retention.

**Result: the provisioned catalog fires and recovers end-to-end in the dev
stack — pending→firing→recovered proven with a fault-driven condition.**
