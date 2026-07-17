# TheraPy dev tasks. Run `make` (or `make help`) to list targets.
#
# Source (incl. the PWA static assets) and tests are bind-mounted into the
# container, so UI/code/test edits are live — no image rebuild needed except
# when dependencies change (then `make rebuild`).
#
#   UI edit (JS/CSS/HTML) → just reload the browser
#   Python edit           → `make restart`
#   Test edit             → `make test` / `make test-e2e`  (pytest reads it directly)
#
# Tests live under tests/suites/{unit,integration,e2e} and are auto-marked by
# folder. Select any subset with ARGS, e.g.:
#   make test ARGS="-k memory -x"  make test-e2e ARGS="-k hold"  make test-unit

COMPOSE := docker compose
SVC     := therapy
EXEC    := $(COMPOSE) exec -T $(SVC)
RUN     := $(EXEC) uv run
VENV    := .venv/bin
ARGS    ?=

.DEFAULT_GOAL := help

.PHONY: help
help: ## List available targets
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-30s\033[0m %s\n", $$1, $$2}'

# ---- Container lifecycle ----------------------------------------------------

.PHONY: up
up: ## Build if needed + (re)start the stack in the background
	$(COMPOSE) up -d --build $(SVC)

.PHONY: rebuild
rebuild: ## Force a clean image rebuild (use when dependencies change)
	$(COMPOSE) build --no-cache $(SVC)
	$(COMPOSE) up -d $(SVC)

.PHONY: restart
restart: ## Restart the server to pick up Python edits (no rebuild)
	$(COMPOSE) restart $(SVC)

.PHONY: down
down: ## Stop the stack
	$(COMPOSE) down

.PHONY: status
status: ## Show container status + health
	$(COMPOSE) ps

.PHONY: logs
logs: ## Follow the server logs
	$(COMPOSE) logs -f $(SVC)

.PHONY: shell
shell: ## Open an interactive shell in the running container
	$(COMPOSE) exec $(SVC) bash

# ---- Tests & quality --------------------------------------------------------
# ARGS passes extra pytest flags: `make test ARGS="-k memory"`, `make test-e2e ARGS="-k hold"`.

.PHONY: test
test: ## Unit + integration in the container (the real test bed)
	$(RUN) pytest $(ARGS)

.PHONY: test-unit unit
test-unit: ## Just the unit suite
	$(RUN) pytest -m unit $(ARGS)

unit: test-unit

.PHONY: test-integration integration
test-integration: ## Just the integration suite
	$(RUN) pytest -m integration $(ARGS)

integration: test-integration

.PHONY: test-e2e e2e
test-e2e: ## ALL browser e2e (auto-installs Chromium if missing)
	$(RUN) playwright install --with-deps chromium firefox >/dev/null # WebKit deps are unavailable in the pinned image.
	$(RUN) pytest -m e2e $(ARGS)

e2e: test-e2e

.PHONY: test-fast
test-fast: ## Quick framework-free run on the host slim venv
	$(VENV)/python -m pytest $(ARGS)

.PHONY: lint
lint: ## Ruff lint (host)
	$(VENV)/ruff check .

.PHONY: verify-longitudinal-loop
verify-longitudinal-loop: ## Verify longitudinal self-knowledge loop (host, isolated data)
	$(VENV)/python scripts/verify_longitudinal_loop.py

.PHONY: check
check: lint test-fast ## Fast pre-push gate: lint + framework-free tests (host)

# --- observability gate tooling (obs plan §11) -------------------------------

.PHONY: obs-canary-scan
obs-canary-scan: ## Routing/secret canary gate over the fixture corpus
	$(VENV)/python scripts/observability/canary_scan.py fixtures

.PHONY: obs-fixture-hash
obs-fixture-hash: ## Reproducible identity of the observability fixture corpus
	$(VENV)/python scripts/observability/fixture_hash.py

.PHONY: obs-baseline
obs-baseline: ## Telemetry-off/on workload baseline against a running instance
	$(VENV)/python scripts/observability/baseline.py --label off

.PHONY: obs-dashboards
obs-dashboards: ## Regenerate the six Grafana dashboards deterministically
	$(VENV)/python scripts/observability/gen_dashboards.py

.PHONY: obs-fixtures
obs-fixtures: ## Regenerate golden interaction fixtures + canaries
	$(VENV)/python scripts/observability/gen_interaction_fixtures.py
