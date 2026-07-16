"""Server launcher: observability bootstrap BEFORE the app imports (plan O1.1).

`python -m therapy.server` configures JSON stdout logging, the third-party
logger policy, and the owned OTel provider first, then imports/serves the
FastAPI app. Pipecat's `setup_tracing()` is never called — it obtains
tracers from the provider installed here (plan §2).

Uvicorn's access log is off: it prints concrete paths/queries (leak audit
item 12); broad FastAPI spans replace it in O2.
"""

from __future__ import annotations

import logging
import os


def main() -> int:
    from therapy import __version__
    from therapy.observability.config import ObservabilityConfig
    from therapy.observability.logging import configure_stdout_json_logging, emit_event
    from therapy.observability.telemetry import initialize as initialize_telemetry

    config = ObservabilityConfig.from_env()

    pipecat_version = "not-installed"
    try:
        import importlib.metadata as metadata

        pipecat_version = metadata.version("pipecat-ai")
    except Exception:
        pass

    configure_stdout_json_logging(
        level=config.log_level,
        service_version=__version__,
        environment=config.environment,
        resource={
            "pipecat.version": pipecat_version,
            "capture.mode": config.capture_mode.value,
            "capture.backend": config.interaction_backend,
            "schema.genai": "none",  # pinned when O2 adds the semconv dep
            "config.fingerprint": config.fingerprint(),
        },
    )
    otel_on = initialize_telemetry(config, service_version=__version__)
    emit_event(
        "app.starting",
        severity=logging.INFO,
        component="server",
        operation="bootstrap",
        outcome="success" if otel_on or not config.otel_enabled else "error",
    )

    import uvicorn

    uvicorn.run(
        "therapy.server.app:app",
        host=os.environ.get("THERAPY_HOST", "0.0.0.0"),
        port=int(os.environ.get("THERAPY_PORT", "8000")),
        access_log=False,
        log_config=None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
