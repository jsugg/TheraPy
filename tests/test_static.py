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


def test_client_loads_resumed_transcript_over_http() -> None:
    # The resumed transcript must render from an HTTP fetch on connect, not
    # only the data-channel `session` replay — pipecat drops that replay when
    # the channel is slow to open on mobile over the TURN relay, leaving the
    # pane empty (field test 2026-07-10).
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert "loadConnectHistory" in app_js
    assert re.search(r"fetch\(`/api/sessions/\$\{[^}]+\}`\)", app_js), (
        "connect must fetch the resumed session transcript over HTTP"
    )
