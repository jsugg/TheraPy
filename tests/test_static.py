"""Regression tests for static shell assets."""

from pathlib import Path
import re


STATIC_DIR = Path(__file__).parents[1] / "src/therapy/server/static"


def test_hidden_views_do_not_participate_in_layout() -> None:
    css = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")

    assert re.search(r"\[hidden\]\s*\{\s*display:\s*none\s*!important\s*;", css), (
        "hidden chat panes must not flex and squeeze the history transcript"
    )


def test_shell_cache_is_bumped_after_static_shell_fix() -> None:
    service_worker = (STATIC_DIR / "sw.js").read_text(encoding="utf-8")

    assert "therapy-shell-v2" not in service_worker
