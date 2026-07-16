"""Strict, frozen observability configuration (plan §4).

Values are parsed and validated exactly once at startup. Invalid values
fail fast with the offending variable NAME only — never its value, so a
mistyped secret cannot leak through an error message. `repr()` and the
fingerprint expose no secrets either (this object stores none; endpoint
URLs are validated to contain no embedded credentials or query).
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from urllib.parse import urlsplit

from therapy.observability.model import CaptureMode

#: Backends the runtime may export interactions to. `journal` is the
#: always-on local source of truth; `phoenix` is the single backend selected
#: by the O0.3 ADR (docs/evidence/observability-backend-spike.md). No third
#: runtime SDK path ships (plan §4).
SUPPORTED_BACKENDS: tuple[str, ...] = ("journal", "phoenix")

SUPPORTED_ENVIRONMENTS: tuple[str, ...] = ("development", "test", "dogfood", "vps-test")

SUPPORTED_LOG_LEVELS: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


class ConfigError(ValueError):
    """Invalid observability configuration; message carries names only."""


def _flag(name: str, default: str) -> bool:
    raw = os.environ.get(name, default).strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off", ""}:
        return False
    raise ConfigError(f"{name} must be a boolean flag")


def _positive_int(name: str, default: int, *, maximum: int = 1_000_000) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc
    if not 0 < value <= maximum:
        raise ConfigError(f"{name} out of range")
    return value


def _positive_float(name: str, default: float, *, maximum: float = 3600.0) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number") from exc
    if not 0 < value <= maximum:
        raise ConfigError(f"{name} out of range")
    return value


def _choice(name: str, default: str, allowed: tuple[str, ...]) -> str:
    raw = os.environ.get(name, default).strip()
    value = raw.lower() if raw.lower() in allowed else raw.upper()
    if value not in allowed:
        raise ConfigError(f"{name} must be one of {sorted(allowed)}")
    return value


def _otlp_endpoint(name: str, default: str | None) -> str | None:
    """OTLP HTTP endpoint: scheme/host only, no credentials, no query."""
    raw = os.environ.get(name, "").strip() or default
    if raw is None or raw == "":
        return None
    parts = urlsplit(raw)
    if parts.scheme not in {"http", "https"}:
        raise ConfigError(f"{name} must be an http(s) URL")
    if not parts.hostname:
        raise ConfigError(f"{name} must include a host")
    if parts.username or parts.password:
        raise ConfigError(f"{name} must not embed credentials")
    if parts.query or parts.fragment:
        raise ConfigError(f"{name} must not carry a query or fragment")
    return raw.rstrip("/")


@dataclass(frozen=True, slots=True)
class ObservabilityConfig:
    """Validated once at startup; safe to log via `fingerprint()`."""

    capture_mode: CaptureMode
    journal_path: Path
    interaction_backend: str
    remote_export_enabled: bool
    retention_days: int
    queue_size: int
    group_commit_ms: int
    otel_enabled: bool
    otlp_broad_endpoint: str | None
    otlp_restricted_endpoint: str | None
    otel_export_timeout_secs: float
    environment: str
    log_level: str
    client_telemetry_enabled: bool

    @classmethod
    def from_env(cls) -> ObservabilityConfig:
        data_dir = Path(os.environ.get("THERAPY_DATA_DIR", "data"))
        journal_default = data_dir / "interaction-journal.sqlite3"
        journal_raw = os.environ.get("THERAPY_INTERACTION_JOURNAL", "").strip()
        journal_path = Path(journal_raw) if journal_raw else journal_default

        # The journal must never share a file with the product database.
        # `therapy.db` is THE product DB (MemoryStore/UserModel/research all
        # open it); compare resolved paths so relative spellings can't slip
        # past, and keep the name-level guard for other product sqlite files.
        product_db_names = {"therapy.db", "memory.sqlite3", "user_model.sqlite3",
                            "research.sqlite3"}
        if journal_path.name in product_db_names:
            raise ConfigError("THERAPY_INTERACTION_JOURNAL must not be a product DB")
        try:
            if journal_path.resolve() == (data_dir / "therapy.db").resolve():
                raise ConfigError(
                    "THERAPY_INTERACTION_JOURNAL must not be the product DB"
                )
        except OSError:
            pass

        raw_mode = os.environ.get("THERAPY_CAPTURE_MODE", "runtime").strip().lower()
        try:
            capture_mode = CaptureMode(raw_mode)
        except ValueError as exc:
            raise ConfigError(
                "THERAPY_CAPTURE_MODE must be disabled|runtime|evaluation"
            ) from exc

        backend = (
            os.environ.get("THERAPY_INTERACTION_BACKEND", "journal").strip().lower()
        )
        if backend not in SUPPORTED_BACKENDS:
            raise ConfigError(
                "THERAPY_INTERACTION_BACKEND must be one of "
                f"{sorted(SUPPORTED_BACKENDS)}"
            )
        if backend != "journal" and not os.environ.get(
            "THERAPY_OTLP_RESTRICTED_ENDPOINT", ""
        ).strip():
            raise ConfigError(
                "THERAPY_INTERACTION_BACKEND requires "
                "THERAPY_OTLP_RESTRICTED_ENDPOINT (never inferred from broad)"
            )

        return cls(
            capture_mode=capture_mode,
            journal_path=journal_path,
            interaction_backend=backend,
            remote_export_enabled=_flag("THERAPY_INTERACTION_REMOTE_EXPORT", "0"),
            retention_days=_positive_int(
                "THERAPY_INTERACTION_RETENTION_DAYS", 30, maximum=3650
            ),
            queue_size=_positive_int(
                "THERAPY_INTERACTION_QUEUE_SIZE", 256, maximum=65536
            ),
            group_commit_ms=_positive_int(
                "THERAPY_INTERACTION_GROUP_COMMIT_MS", 50, maximum=5000
            ),
            otel_enabled=_flag("THERAPY_OTEL_ENABLED", "0"),
            otlp_broad_endpoint=_otlp_endpoint(
                "THERAPY_OTLP_BROAD_ENDPOINT", "http://localhost:4318"
            ),
            otlp_restricted_endpoint=_otlp_endpoint(
                "THERAPY_OTLP_RESTRICTED_ENDPOINT", None
            ),
            otel_export_timeout_secs=_positive_float(
                "THERAPY_OTEL_EXPORT_TIMEOUT_SECS", 3.0, maximum=60.0
            ),
            environment=_choice(
                "THERAPY_ENVIRONMENT", "development", SUPPORTED_ENVIRONMENTS
            ),
            log_level=_choice("THERAPY_LOG_LEVEL", "INFO", SUPPORTED_LOG_LEVELS),
            client_telemetry_enabled=_flag("THERAPY_CLIENT_TELEMETRY", "0"),
        )

    def fingerprint(self) -> str:
        """Deterministic, secret-free digest for the startup record."""
        payload = {
            f.name: str(getattr(self, f.name)) for f in fields(self)
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        return digest[:16]
