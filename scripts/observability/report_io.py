"""Restricted-plane report writing shared by the evaluation CLIs.

Evaluation reports carry exact fixture and response content (restricted
plane), so by default they may only land under the repo's ignored `.local`
directory, are created owner-read/write only, and the CLIs print a
repo-relative label instead of an arbitrary concrete path (O3 audit privacy
finding).
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path


def write_restricted_report(
    report: Mapping[str, object],
    output: Path,
    *,
    repo_root: Path,
    allow_unrestricted: bool = False,
) -> str:
    """Write ``report`` as 0600 JSON and return a printable destination label.

    Raises:
        ValueError: when the destination is outside the restricted `.local`
            directory and the caller did not deliberately override.
    """
    resolved = output.resolve()
    restricted_root = (repo_root / ".local").resolve()
    inside_restricted = resolved.is_relative_to(restricted_root)
    if not inside_restricted and not allow_unrestricted:
        raise ValueError(
            "refusing to write a restricted evaluation report outside "
            f"{restricted_root}; pass --unrestricted-output to override "
            "deliberately"
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd = os.open(resolved, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(payload)
    # os.open's mode is masked by umask and ignored for existing files.
    resolved.chmod(0o600)
    if inside_restricted:
        return str(resolved.relative_to(repo_root.resolve()))
    return "external destination (explicitly unrestricted)"
