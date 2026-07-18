"""Verify and summarize one restricted interaction journal replay on stdout.

The command accepts no output path and performs no network operations. Exact
captured content remains in the journal; the CLI emits only bounded metadata
and deterministic verification results.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = next(
    path
    for path in Path(__file__).resolve().parents
    if (path / "pyproject.toml").exists()
)
sys.path.insert(0, str(REPO_ROOT))

from therapy.observability.interactions import JsonValue  # noqa: E402
from therapy.observability.replay import (  # noqa: E402
    RenderedReplay,
    ReplayEnvelope,
    ReplayError,
    reconstruct,
    replay_render,
)


def _event_count(value: JsonValue | None) -> int:
    return len(value) if isinstance(value, list) else 0


def replay_summary(
    envelope: ReplayEnvelope, rendered: RenderedReplay
) -> dict[str, JsonValue]:
    """Build a content-free verification summary for stdout.

    Args:
        envelope: Reconstructed restricted interaction evidence.
        rendered: Pure request render and verification result.

    Returns:
        Bounded identifiers, lifecycle metadata, counts, and verification flags.
    """
    record = envelope.record
    provider_native = record.get("provider_native")
    native_events: JsonValue | None = None
    if isinstance(provider_native, dict):
        native_events = provider_native.get("ordered_events")
    return {
        "interaction_id": envelope.interaction_id,
        "operation": record.get("operation")
        if isinstance(record.get("operation"), str)
        else "unknown",
        "provider": record.get("provider")
        if isinstance(record.get("provider"), str)
        else "unknown",
        "status": record.get("status")
        if isinstance(record.get("status"), str)
        else "unknown",
        "stream_event_count": _event_count(record.get("stream")),
        "provider_event_count": _event_count(native_events),
        "has_terminal": isinstance(record.get("terminal"), dict),
        "checksum_verified": rendered.checksum_verified,
        "exact_render_match": rendered.exact_match,
        "verified": rendered.verified,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct and verify one restricted journal interaction."
    )
    parser.add_argument("--journal", required=True, type=Path)
    parser.add_argument("--interaction-id", required=True)
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the bounded summary as JSON on stdout",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the stdout-only replay verification CLI."""
    args = _parser().parse_args(argv)
    try:
        envelope = reconstruct(args.journal, args.interaction_id)
        rendered = replay_render(envelope)
    except (OSError, ReplayError, ValueError) as error:
        print(f"replay failed: {error}", file=sys.stderr)
        return 2

    summary = replay_summary(envelope, rendered)
    if args.json:
        print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    else:
        for key, value in summary.items():
            print(f"{key}: {value}")
    return 0 if rendered.verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
