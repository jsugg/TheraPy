"""Upsert O3.4 golden datasets into the selected local Phoenix 18.0.0.

Run this module with the existing spike environment, not the project venv::

    .local/obs-spike/phoenix-venv/bin/python \
        scripts/observability/phoenix_datasets.py

Phoenix 18's ``create_dataset`` has same-name update semantics. Stable example
IDs make re-runs idempotent: the upload is diffed against the latest dataset
version, changes create a new version, and a duplicate named dataset is not
created. This script intentionally uses that observed versioning behavior.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import urlsplit

REPO_ROOT = next(
    path
    for path in Path(__file__).resolve().parents
    if (path / "pyproject.toml").exists()
)
INTERACTIONS_ROOT = REPO_ROOT / "tests/fixtures/observability/interactions"
BEHAVIOR_FIXTURE_PATH = REPO_ROOT / "tests/fixtures/observability/behavior/cases.json"
PHOENIX_VERSION = "18.0.0"
GOLDEN_DATASET_NAME = "therapy-golden-interactions"
BEHAVIOR_DATASET_NAME = "therapy-behavior-cases"
DEFAULT_ENDPOINT = "http://localhost:6006"

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | Sequence["JsonValue"] | Mapping[str, "JsonValue"]
type JsonObject = dict[str, JsonValue]


class DatasetResult(Protocol):
    """Phoenix dataset result fields printed by this script."""

    name: str
    version_id: str
    example_count: int


class DatasetsClient(Protocol):
    """Narrow Phoenix dataset client surface used by this script."""

    def create_dataset(
        self,
        *,
        name: str,
        examples: Iterable[JsonObject],
        dataset_description: str,
        timeout: int,
    ) -> DatasetResult: ...


class PhoenixClient(Protocol):
    """Narrow Phoenix client surface used by this script."""

    @property
    def datasets(self) -> DatasetsClient: ...


class _ClientFactory(Protocol):
    def __call__(self, *, base_url: str) -> object: ...


def _json_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be a JSON object")
    return cast(dict[str, object], value)


def _json_value(value: object, label: str) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_value(item, f"{label}[]") for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        mapping = cast(dict[str, object], value)
        return {
            key: _json_value(item, f"{label}.{key}") for key, item in mapping.items()
        }
    raise ValueError(f"{label} contains a non-JSON value")


def _required_string(payload: dict[str, object], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value


def _required_object(
    payload: dict[str, object], key: str, label: str
) -> dict[str, object]:
    return _json_object(payload.get(key), f"{label}.{key}")


def _load_object(path: Path) -> dict[str, object]:
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    return _json_object(raw, str(path))


def build_golden_interaction_examples() -> list[JsonObject]:
    """Build Phoenix examples from the canonical request and terminal slices."""
    examples: list[JsonObject] = []
    paths = sorted(INTERACTIONS_ROOT.glob("*.json"))
    if not paths:
        raise ValueError(f"no interaction fixtures found under {INTERACTIONS_ROOT}")
    for path in paths:
        fixture = _load_object(path)
        case_id = _required_string(fixture, "case", str(path))
        scenario = _required_string(fixture, "scenario", str(path))
        record = _required_object(fixture, "record", str(path))
        request = _required_object(record, "request", f"{path}.record")
        provider = _required_string(record, "provider", f"{path}.record")
        status = _required_string(record, "status", f"{path}.record")
        if "response" not in record or "error" not in record:
            raise ValueError(f"{path}.record must contain response and error fields")
        terminal: JsonObject = {
            "status": status,
            "response": _json_value(record["response"], f"{path}.record.response"),
            "error": _json_value(record["error"], f"{path}.record.error"),
        }
        examples.append(
            {
                "id": case_id,
                "input": cast(
                    JsonObject, _json_value(request, f"{path}.record.request")
                ),
                "output": terminal,
                "metadata": {"scenario": scenario, "provider": provider},
            }
        )
    return examples


def build_behavior_examples() -> list[JsonObject]:
    """Build Phoenix examples from behavioral fixture expectations."""
    payload = _load_object(BEHAVIOR_FIXTURE_PATH)
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(f"{BEHAVIOR_FIXTURE_PATH}: cases must be a non-empty list")

    examples: list[JsonObject] = []
    for index, raw_case in enumerate(raw_cases):
        label = f"{BEHAVIOR_FIXTURE_PATH}: cases[{index}]"
        case = _json_object(raw_case, label)
        case_id = _required_string(case, "id", label)
        user_input = _required_string(case, "user_input", label)
        dimension = _required_string(case, "dimension", label)
        expected = case.get("expected_behavior")
        high_risk = case.get("high_risk")
        if not isinstance(expected, list) or not all(
            isinstance(item, str) for item in expected
        ):
            raise ValueError(f"{label}.expected_behavior must be a list of strings")
        if not isinstance(high_risk, bool):
            raise ValueError(f"{label}.high_risk must be a boolean")
        examples.append(
            {
                "id": case_id,
                "input": {"user_input": user_input},
                "output": {"expected": cast(list[str], expected)},
                "metadata": {"dimension": dimension, "high_risk": high_risk},
            }
        )
    return examples


def _phoenix_client(endpoint: str) -> PhoenixClient:
    try:
        installed_version = version("arize-phoenix")
    except PackageNotFoundError as error:
        raise RuntimeError(
            "arize-phoenix is unavailable; run this script with "
            ".local/obs-spike/phoenix-venv/bin/python"
        ) from error
    if installed_version != PHOENIX_VERSION:
        raise RuntimeError(
            f"arize-phoenix=={PHOENIX_VERSION} is required; found {installed_version}"
        )
    try:
        phoenix_client = import_module("phoenix.client")
    except ImportError as error:
        raise RuntimeError(
            "Phoenix client import failed; use the existing obs-spike Phoenix venv"
        ) from error

    raw_client_factory = getattr(phoenix_client, "Client", None)
    if not callable(raw_client_factory):
        raise RuntimeError("Phoenix 18 does not expose the expected Client API")
    client_factory = cast(_ClientFactory, raw_client_factory)
    client = client_factory(base_url=endpoint)
    datasets = getattr(client, "datasets", None)
    if datasets is None or not callable(getattr(datasets, "create_dataset", None)):
        raise RuntimeError(
            "Phoenix 18 client does not expose the expected datasets API"
        )
    return cast(PhoenixClient, client)


def _validate_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("--endpoint must be an absolute HTTP(S) URL")
    return endpoint.rstrip("/")


def upsert_datasets(endpoint: str, timeout: int) -> list[DatasetResult]:
    """Upsert both versioned datasets at a Phoenix endpoint."""
    client = _phoenix_client(_validate_endpoint(endpoint))
    golden = client.datasets.create_dataset(
        name=GOLDEN_DATASET_NAME,
        examples=build_golden_interaction_examples(),
        dataset_description=(
            "Versioned canonical requests and terminal outcomes from synthetic "
            "golden interaction fixtures."
        ),
        timeout=timeout,
    )
    behavior = client.datasets.create_dataset(
        name=BEHAVIOR_DATASET_NAME,
        examples=build_behavior_examples(),
        dataset_description=(
            "Versioned synthetic behavioral expectations for regression experiments."
        ),
        timeout=timeout,
    )
    return [golden, behavior]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Upsert TheraPy evaluation fixtures into local Phoenix 18."
    )
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument(
        "--timeout",
        type=int,
        default=15,
        help="per-dataset Phoenix request timeout in seconds (default: 15)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the Phoenix upload CLI with clear dependency and endpoint errors."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")
    try:
        datasets = upsert_datasets(args.endpoint, args.timeout)
    except Exception as error:
        # HTTP/client exceptions can embed concrete endpoints, query strings,
        # or response text — surface only the bounded exception class (O3
        # audit privacy finding).
        parser.error(
            "Phoenix dataset upload failed "
            f"(error_type={type(error).__name__}); inspect locally with a "
            "debugger for details"
        )
    for dataset in datasets:
        print(
            json.dumps(
                {
                    "dataset": dataset.name,
                    "version_id": dataset.version_id,
                    "example_count": dataset.example_count,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
