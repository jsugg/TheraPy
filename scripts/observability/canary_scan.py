"""Routing/secret canary scanner (plan O0.2, §11).

Reusable gate logic: content canaries must appear in restricted evidence
only; forbidden (credential/infrastructure) canaries must appear nowhere.

Two entry points:

- `scan_fixture_tree()` — the O0 gate over the committed fixture corpus,
  where each interaction fixture separates its `record` (restricted plane)
  from its `broad_twin` (broad plane).
- `scan_artifacts()` — later-phase scans over arbitrary produced artifacts
  (stdout logs, OTLP dumps, dashboards, backend exports): every hit of any
  content or forbidden canary in a *broad/artifact* surface is a failure.

Exit code 0 = gate passed; 1 = violations (printed one per line).

Usage:
    .venv/bin/python scripts/observability/canary_scan.py fixtures
    .venv/bin/python scripts/observability/canary_scan.py artifacts PATH...
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()
)
FIXTURE_ROOT = REPO_ROOT / "tests/fixtures/observability"


def load_canaries(root: Path = FIXTURE_ROOT) -> tuple[dict[str, str], dict[str, str]]:
    data = json.loads((root / "canaries.json").read_text(encoding="utf-8"))
    return data["content"], data["forbidden"]


@dataclass
class ScanReport:
    """Violations plus the coverage ledger the O0 gate needs."""

    violations: list[str] = field(default_factory=list)
    restricted_hits: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.violations


def _hits(text: str, canaries: dict[str, str]) -> list[str]:
    return [name for name, value in canaries.items() if value in text]


def scan_fixture_tree(root: Path = FIXTURE_ROOT) -> ScanReport:
    """O0 gate: content canaries exactly restricted; forbidden nowhere."""
    content, forbidden = load_canaries(root)
    report = ScanReport(restricted_hits=dict.fromkeys(content, 0))

    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "canaries.json":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = path.relative_to(root).as_posix()

        for name in _hits(text, forbidden):
            report.violations.append(f"{rel}: forbidden canary '{name}' present")

        if path.parent.name == "interactions" and path.suffix == ".json":
            fixture = json.loads(text)
            record_text = json.dumps(fixture.get("record", {}), sort_keys=True)
            broad_text = json.dumps(fixture.get("broad_twin", {}), sort_keys=True)
            for name in _hits(record_text, content):
                report.restricted_hits[name] += 1
            for name in _hits(broad_text, content):
                report.violations.append(
                    f"{rel}: content canary '{name}' leaked into broad_twin"
                )
        else:
            # Non-interaction fixture surfaces are all broad-safe surfaces:
            # exact content canaries may not appear there either.
            for name in _hits(text, content):
                report.violations.append(
                    f"{rel}: content canary '{name}' outside restricted records"
                )

    for name, count in report.restricted_hits.items():
        if count == 0:
            report.violations.append(
                f"content canary '{name}' missing from every restricted record"
            )
    return report


def scan_artifacts(paths: list[Path], root: Path = FIXTURE_ROOT) -> ScanReport:
    """Broad-artifact scan: any canary hit at all is a violation."""
    content, forbidden = load_canaries(root)
    merged = {**content, **forbidden}
    report = ScanReport()
    for base in paths:
        candidates = [base] if base.is_file() else sorted(base.rglob("*"))
        for path in candidates:
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                report.violations.append(f"{path}: unreadable ({type(exc).__name__})")
                continue
            for name in _hits(text, merged):
                report.violations.append(f"{path}: canary '{name}' present")
    return report


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in {"fixtures", "artifacts"}:
        print(__doc__, file=sys.stderr)
        return 2
    if argv[0] == "fixtures":
        report = scan_fixture_tree()
    else:
        report = scan_artifacts([Path(arg) for arg in argv[1:]])
    for violation in report.violations:
        print(f"CANARY-VIOLATION {violation}")
    if report.ok:
        summary = {
            "result": "pass",
            "restricted_hits": report.restricted_hits,
        }
        print(json.dumps(summary, sort_keys=True))
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
