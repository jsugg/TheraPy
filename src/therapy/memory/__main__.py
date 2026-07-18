"""Owner-data and local research maintenance CLI."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import cast

from therapy.data import DataSovereignty
from therapy.knowledge.research import ResearchKB


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m therapy.memory")
    commands = parser.add_subparsers(dest="command", required=True)

    export = commands.add_parser("export", help="export all owner data and corpus")
    export.add_argument("--output", type=Path)

    restore = commands.add_parser("restore", help="restore a complete owner export")
    restore.add_argument("--input", type=Path, required=True)
    restore.add_argument("--yes", action="store_true")

    delete = commands.add_parser("delete", help="delete all owner data and corpus")
    delete.add_argument("--yes", action="store_true")

    ingest = commands.add_parser("research-ingest", help="ingest/OCR one local source")
    ingest.add_argument("path", type=Path)
    ingest.add_argument("--title")
    ingest.add_argument("--ref")
    ingest.add_argument("--force", action="store_true")

    commands.add_parser("research-list", help="list local research sources")
    show = commands.add_parser("research-show", help="show extraction/OCR preview")
    show.add_argument("document_id", type=int)
    correct = commands.add_parser("research-correct", help="correct one OCR block")
    correct.add_argument("document_id", type=int)
    correct.add_argument("anchor")
    correct.add_argument("--text", required=True)
    reindex = commands.add_parser("research-reindex", help="rebuild semantic index")
    reindex.add_argument("document_id", type=int, nargs="?")
    remove = commands.add_parser("research-delete", help="delete one local source")
    remove.add_argument("document_id", type=int)
    remove.add_argument("--yes", action="store_true")
    return parser


def _private_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)


def _print_json(value: object) -> None:
    sys.stdout.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Run one owner maintenance command and return a process exit code."""
    args = _parser().parse_args(argv)
    try:
        if args.command == "export":
            payload = DataSovereignty().export_json()
            if args.output is None:
                sys.stdout.buffer.write(payload)
            else:
                _private_write(args.output, payload)
            return 0
        if args.command == "restore":
            if not args.yes:
                print("Refusing destructive restore without --yes.", file=sys.stderr)
                return 2
            snapshot_value: object = json.loads(args.input.read_text(encoding="utf-8"))
            if not isinstance(snapshot_value, dict):
                raise ValueError("restore snapshot root must be an object")
            raw_snapshot = cast(dict[object, object], snapshot_value)
            if not all(isinstance(key, str) for key in raw_snapshot):
                raise ValueError("restore snapshot field names must be strings")
            snapshot = cast(dict[str, object], snapshot_value)
            _print_json(DataSovereignty().restore_snapshot(snapshot))
            return 0
        if args.command == "delete":
            if not args.yes:
                print("Refusing to delete personal data without --yes.", file=sys.stderr)
                return 2
            DataSovereignty().delete_all()
            print("Deleted all TheraPy owner data and corpus.", file=sys.stderr)
            return 0

        research = ResearchKB()
        if args.command == "research-ingest":
            payload = args.path.read_bytes()
            result = research.ingest_bytes(
                payload,
                args.path.name,
                None,
                source_title=args.title,
                source_ref=args.ref,
                force=args.force,
            )
            _print_json(result)
            return 0
        if args.command == "research-list":
            _print_json({"documents": research.documents()})
            return 0
        if args.command == "research-show":
            document = research.document(args.document_id)
            if document is None:
                raise ValueError("research document not found")
            _print_json({"document": document})
            return 0
        if args.command == "research-correct":
            if not research.correct_block(args.document_id, args.anchor, args.text):
                raise ValueError("research block not found")
            _print_json({"document": research.document(args.document_id)})
            return 0
        if args.command == "research-reindex":
            _print_json({"chunks_indexed": research.reindex(args.document_id)})
            return 0
        if args.command == "research-delete":
            if not args.yes:
                print("Refusing research deletion without --yes.", file=sys.stderr)
                return 2
            if not research.delete_document(args.document_id):
                raise ValueError("research document not found")
            _print_json({"deleted": args.document_id})
            return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {type(exc).__name__}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
