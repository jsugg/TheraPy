"""Single reproducible hash over the committed observability fixture corpus.

The O0.3 backend spike must run *identical* fixtures through every candidate
(plan O0.3); this digest is the identity recorded in the decision record.

Usage: .venv/bin/python scripts/observability/fixture_hash.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()
)
FIXTURE_ROOT = REPO_ROOT / "tests/fixtures/observability"


def fixture_hash(root: Path = FIXTURE_ROOT) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\x00")
        digest.update(path.read_bytes())
        digest.update(b"\x01")
    return digest.hexdigest()


def main() -> int:
    print(json.dumps({"fixture_root": str(FIXTURE_ROOT.relative_to(REPO_ROOT)), "sha256": fixture_hash()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
