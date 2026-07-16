"""ObservabilityConfig contract (plan §4, O1 test list)."""

import pytest

from therapy.observability.config import (
    SUPPORTED_BACKENDS,
    ConfigError,
    ObservabilityConfig,
)
from therapy.observability.model import CaptureMode


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in (
        "THERAPY_CAPTURE_MODE",
        "THERAPY_INTERACTION_JOURNAL",
        "THERAPY_INTERACTION_BACKEND",
        "THERAPY_INTERACTION_REMOTE_EXPORT",
        "THERAPY_INTERACTION_RETENTION_DAYS",
        "THERAPY_INTERACTION_QUEUE_SIZE",
        "THERAPY_INTERACTION_GROUP_COMMIT_MS",
        "THERAPY_OTEL_ENABLED",
        "THERAPY_OTLP_BROAD_ENDPOINT",
        "THERAPY_OTLP_RESTRICTED_ENDPOINT",
        "THERAPY_OTEL_EXPORT_TIMEOUT_SECS",
        "THERAPY_ENVIRONMENT",
        "THERAPY_LOG_LEVEL",
        "THERAPY_CLIENT_TELEMETRY",
        "THERAPY_DATA_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_defaults_match_the_contract_table(clean_env) -> None:
    config = ObservabilityConfig.from_env()
    assert config.capture_mode is CaptureMode.RUNTIME
    assert config.journal_path.name == "interaction-journal.sqlite3"
    assert config.interaction_backend == "journal"
    assert config.remote_export_enabled is False
    assert config.retention_days == 30
    assert config.queue_size == 256
    assert config.group_commit_ms == 50
    assert config.otel_enabled is False
    assert config.otlp_broad_endpoint == "http://localhost:4318"
    assert config.otlp_restricted_endpoint is None
    assert config.otel_export_timeout_secs == 3.0
    assert config.environment == "development"
    assert config.log_level == "INFO"
    assert config.client_telemetry_enabled is False


def test_journal_defaults_under_data_dir(clean_env) -> None:
    clean_env.setenv("THERAPY_DATA_DIR", "/data")
    config = ObservabilityConfig.from_env()
    assert str(config.journal_path) == "/data/interaction-journal.sqlite3"


def test_journal_must_not_be_a_product_db(clean_env) -> None:
    clean_env.setenv("THERAPY_INTERACTION_JOURNAL", "/data/memory.sqlite3")
    with pytest.raises(ConfigError):
        ObservabilityConfig.from_env()


def test_unknown_backend_rejected(clean_env) -> None:
    clean_env.setenv("THERAPY_INTERACTION_BACKEND", "kafka")
    with pytest.raises(ConfigError):
        ObservabilityConfig.from_env()
    assert SUPPORTED_BACKENDS == ("journal",)  # extended only by the O0 ADR


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("THERAPY_CAPTURE_MODE", "yolo"),
        ("THERAPY_INTERACTION_RETENTION_DAYS", "0"),
        ("THERAPY_INTERACTION_RETENTION_DAYS", "-3"),
        ("THERAPY_INTERACTION_QUEUE_SIZE", "0"),
        ("THERAPY_INTERACTION_QUEUE_SIZE", "abc"),
        ("THERAPY_INTERACTION_GROUP_COMMIT_MS", "999999"),
        ("THERAPY_OTEL_ENABLED", "maybe"),
        ("THERAPY_OTEL_EXPORT_TIMEOUT_SECS", "-1"),
        ("THERAPY_ENVIRONMENT", "production"),
        ("THERAPY_LOG_LEVEL", "TRACE"),
    ],
)
def test_invalid_values_fail_fast(clean_env, name: str, value: str) -> None:
    clean_env.setenv(name, value)
    with pytest.raises(ConfigError):
        ObservabilityConfig.from_env()


@pytest.mark.parametrize(
    "endpoint",
    [
        "ftp://collector:4318",
        "http://user:secret@collector:4318",
        "http://collector:4318/v1?token=abc",
        "http:///nohost",
    ],
)
def test_otlp_endpoint_rejects_credentials_query_and_bad_schemes(
    clean_env, endpoint: str
) -> None:
    clean_env.setenv("THERAPY_OTLP_BROAD_ENDPOINT", endpoint)
    with pytest.raises(ConfigError):
        ObservabilityConfig.from_env()


def test_error_messages_never_echo_values(clean_env) -> None:
    secret_ish = "http://user:sekrit-token@collector:4318"
    clean_env.setenv("THERAPY_OTLP_BROAD_ENDPOINT", secret_ish)
    with pytest.raises(ConfigError) as excinfo:
        ObservabilityConfig.from_env()
    assert "sekrit" not in str(excinfo.value)


def test_fingerprint_is_deterministic_and_short(clean_env) -> None:
    first = ObservabilityConfig.from_env().fingerprint()
    second = ObservabilityConfig.from_env().fingerprint()
    assert first == second
    assert len(first) == 16
    clean_env.setenv("THERAPY_ENVIRONMENT", "test")
    assert ObservabilityConfig.from_env().fingerprint() != first


def test_restricted_endpoint_never_inferred_from_broad(clean_env) -> None:
    clean_env.setenv("THERAPY_OTLP_BROAD_ENDPOINT", "http://collector:4318")
    config = ObservabilityConfig.from_env()
    assert config.otlp_restricted_endpoint is None
