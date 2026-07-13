"""Personal-data export/delete CLI for local-first memory (SPEC §8).

Exports include transcripts, summaries, and user-model facts. Raw audio remains
as local files in the data directory; snapshots only include relative paths.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from therapy.memory import MemoryStore


def _parser() -> argparse.ArgumentParser:
    """Build the memory maintenance CLI parser."""
    parser = argparse.ArgumentParser(prog="python -m therapy.memory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser(
        "export",
        help="write a JSON snapshot of all personal data",
    )
    export_parser.add_argument(
        "--output",
        type=Path,
        help="write the JSON snapshot to PATH instead of stdout",
    )

    delete_parser = subparsers.add_parser(
        "delete",
        help="delete all stored personal data",
    )
    delete_parser.add_argument(
        "--yes",
        action="store_true",
        help="confirm destructive deletion",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the memory export/delete CLI and return a process exit code."""
    args = _parser().parse_args(argv)

    if args.command == "export":
        payload = json.dumps(MemoryStore().export_all(), indent=2, ensure_ascii=False)
        if args.output is None:
            sys.stdout.write(payload)
        else:
            args.output.write_text(payload, encoding="utf-8")
        return 0

    if args.command == "delete":
        if not args.yes:
            print(
                "Refusing to delete personal data without --yes.",
                file=sys.stderr,
            )
            return 2
        MemoryStore().delete_all()
        print("Deleted all TheraPy memory data.", file=sys.stderr)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
