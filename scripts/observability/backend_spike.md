# O0.3 backend acceptance spike runbook

Run from the repository root on the Intel Mac. The commands use only loopback,
create disposable environments under `.local/obs-spike/`, retain results/logs,
and never require credentials.

## 1. Prepare pinned environments

```bash
set -euo pipefail

ROOT="$PWD/.local/obs-spike"
mkdir -p "$ROOT"/{logs,results,storage/phoenix,storage/mlflow/artifacts}

uv venv --python .venv/bin/python "$ROOT/phoenix-venv"
uv pip install --python "$ROOT/phoenix-venv/bin/python" \
  arize-phoenix==18.0.0 \
  opentelemetry-sdk==1.43.0 \
  opentelemetry-exporter-otlp-proto-http==1.43.0 \
  openinference-semantic-conventions==0.1.30

uv venv --python .venv/bin/python "$ROOT/mlflow-venv"
uv pip install --python "$ROOT/mlflow-venv/bin/python" \
  mlflow==3.14.0 \
  opentelemetry-sdk==1.43.0 \
  opentelemetry-exporter-otlp-proto-http==1.43.0 \
  openinference-semantic-conventions==0.1.30

.venv/bin/python scripts/observability/fixture_hash.py
```

Expected corpus SHA-256:
`8198524ecf0027f87415ed95c0e39dff2828a09363c66db86adcda9e44c06479`.
(An earlier run used `683b2279…` with two colliding `(trace_id, span_id)`
pairs; the generator was fixed so every case is a distinct attempt.)

## 2. Confirm the two high ports are free

```bash
.venv/bin/python - <<'PY'
import socket

for port in (62006, 62007):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", port))
print("ports 62006 and 62007 are free")
PY
```

If either bind fails, substitute two free ports consistently in the commands
and in `--endpoint`/`--server-command`.

## 3. Phoenix 18.0.0

Phoenix 18.0.0's HTTP server honors `--host`, but its built-in gRPC collector
binds `[::]`. The `serve-phoenix` shim invokes the pinned Phoenix server after
disabling that unused gRPC listener, so the spike exposes only HTTP on
`127.0.0.1`.

```bash
rm -rf "$ROOT/storage/phoenix"
mkdir -p "$ROOT/storage/phoenix"

PHOENIX_COMMAND="$ROOT/phoenix-venv/bin/python scripts/observability/backend_spike.py serve-phoenix --host 127.0.0.1 --port 62006"

env \
  PHOENIX_TELEMETRY_ENABLED=false \
  PHOENIX_ALLOW_EXTERNAL_RESOURCES=false \
  PHOENIX_DISABLE_AGENT_ASSISTANT=true \
  PHOENIX_AGENTS_DISABLE_WEB_ACCESS=true \
  PHOENIX_HOST=127.0.0.1 \
  PHOENIX_PORT=62006 \
  PHOENIX_WORKING_DIR="$ROOT/storage/phoenix" \
  PHOENIX_SQL_DATABASE_URL="sqlite:////Users/jsugg/dev/github/TheraPy/.local/obs-spike/storage/phoenix/phoenix.db" \
  $PHOENIX_COMMAND >"$ROOT/logs/phoenix.log" 2>&1 &
PHOENIX_PID=$!
echo "$PHOENIX_PID" >"$ROOT/phoenix.pid"

.venv/bin/python - <<'PY'
from time import sleep
from urllib.request import urlopen

for _ in range(240):
    try:
        with urlopen(
            "http://127.0.0.1:62006/arize_phoenix_version", timeout=0.5
        ) as response:
            assert response.read().decode() == "18.0.0"
            break
    except Exception:
        sleep(0.25)
else:
    raise SystemExit("Phoenix did not become ready")
PY

lsof -nP -iTCP:62006 -sTCP:LISTEN

env \
  PHOENIX_TELEMETRY_ENABLED=false \
  PHOENIX_ALLOW_EXTERNAL_RESOURCES=false \
  PHOENIX_DISABLE_AGENT_ASSISTANT=true \
  PHOENIX_AGENTS_DISABLE_WEB_ACCESS=true \
  PHOENIX_HOST=127.0.0.1 \
  PHOENIX_PORT=62006 \
  PHOENIX_WORKING_DIR="$ROOT/storage/phoenix" \
  PHOENIX_SQL_DATABASE_URL="sqlite:////Users/jsugg/dev/github/TheraPy/.local/obs-spike/storage/phoenix/phoenix.db" \
  "$ROOT/phoenix-venv/bin/python" scripts/observability/backend_spike.py measure \
    --backend phoenix \
    --endpoint http://127.0.0.1:62006/v1/traces \
    --storage-dir "$ROOT/storage/phoenix" \
    --server-pid "$PHOENIX_PID" \
    --server-command "$PHOENIX_COMMAND" \
    --output "$ROOT/results/phoenix.json" \
    --outage-probe >"$ROOT/logs/phoenix-measure.log" 2>&1

# --outage-probe normally stopped it; this is an idempotent fallback.
kill "$PHOENIX_PID" 2>/dev/null || true
wait "$PHOENIX_PID" 2>/dev/null || true
```

## 4. MLflow 3.14.0

Job execution remains at MLflow's documented default (`true`) so the RSS
snapshot includes the server configuration actually launched below.

```bash
rm -rf "$ROOT/storage/mlflow"
mkdir -p "$ROOT/storage/mlflow/artifacts"

MLFLOW_COMMAND="$ROOT/mlflow-venv/bin/mlflow server --host 127.0.0.1 --port 62007 --backend-store-uri sqlite:////Users/jsugg/dev/github/TheraPy/.local/obs-spike/storage/mlflow/mlflow.db --serve-artifacts --default-artifact-root mlflow-artifacts:/ --artifacts-destination /Users/jsugg/dev/github/TheraPy/.local/obs-spike/storage/mlflow/artifacts --workers 1"

env MLFLOW_DISABLE_TELEMETRY=true DO_NOT_TRACK=true \
  $MLFLOW_COMMAND >"$ROOT/logs/mlflow.log" 2>&1 &
MLFLOW_PID=$!
echo "$MLFLOW_PID" >"$ROOT/mlflow.pid"

.venv/bin/python - <<'PY'
from time import sleep
from urllib.request import urlopen

for _ in range(240):
    try:
        with urlopen("http://127.0.0.1:62007/health", timeout=0.5) as response:
            assert response.read().decode() == "OK"
            break
    except Exception:
        sleep(0.25)
else:
    raise SystemExit("MLflow did not become ready")
PY

lsof -nP -iTCP:62007 -sTCP:LISTEN

env \
  MLFLOW_DISABLE_TELEMETRY=true \
  DO_NOT_TRACK=true \
  MLFLOW_TRACKING_URI=http://127.0.0.1:62007 \
  "$ROOT/mlflow-venv/bin/python" scripts/observability/backend_spike.py measure \
    --backend mlflow \
    --endpoint http://127.0.0.1:62007/v1/traces \
    --storage-dir "$ROOT/storage/mlflow" \
    --server-pid "$MLFLOW_PID" \
    --server-command "$MLFLOW_COMMAND" \
    --output "$ROOT/results/mlflow.json" \
    --outage-probe >"$ROOT/logs/mlflow-measure.log" 2>&1

kill "$MLFLOW_PID" 2>/dev/null || true
wait "$MLFLOW_PID" 2>/dev/null || true
```

The script creates or reuses the `therapy-observability-spike` MLflow
experiment and supplies its required `x-mlflow-experiment-id` OTLP header.

## 5. Inspect retained evidence and teardown

```bash
python3 -m json.tool "$ROOT/results/phoenix.json" >/dev/null
python3 -m json.tool "$ROOT/results/mlflow.json" >/dev/null

lsof -nP -iTCP:62006 -sTCP:LISTEN || true
lsof -nP -iTCP:62007 -sTCP:LISTEN || true

# Keep results and logs. The venvs are optional after review:
# rm -rf "$ROOT/phoenix-venv" "$ROOT/mlflow-venv"
```

The outage probe sends `SIGTERM` to the server tree immediately before a
one-span export and records the OTLP exporter's result. It is a deterministic
post-kill availability test, not a guaranteed in-flight TCP interruption.
