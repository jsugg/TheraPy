"""Regression tests for static shell assets."""

import json
import re
import struct
from pathlib import Path

import pytest


@pytest.fixture
def static_dir(repo_root: Path) -> Path:
    return repo_root / "src/therapy/server/static"


def _png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name} is not a PNG"
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def test_hidden_views_do_not_participate_in_layout(static_dir: Path) -> None:
    css = (static_dir / "styles.css").read_text(encoding="utf-8")

    assert re.search(r"\[hidden\]\s*\{\s*display:\s*none\s*!important\s*;", css), (
        "hidden chat panes must not flex and squeeze the history transcript"
    )


def test_shell_cache_is_bumped_after_static_shell_fix(static_dir: Path) -> None:
    service_worker = (static_dir / "sw.js").read_text(encoding="utf-8")

    assert "therapy-shell-v2" not in service_worker


def test_manifest_declares_installable_png_icons(static_dir: Path) -> None:
    # Chrome only offers "Install app" with raster 192px and 512px icons; the
    # SVG-only manifest gave no install option on the phone (field test
    # 2026-07-11). A maskable icon keeps Android from framing it on white.
    manifest = json.loads(
        (static_dir / "manifest.webmanifest").read_text(encoding="utf-8")
    )
    icons = manifest.get("icons", [])
    png = {i["sizes"]: i for i in icons if i.get("type") == "image/png"}

    assert "192x192" in png, "need a 192px PNG icon"
    assert "512x512" in png, "need a 512px PNG icon"
    assert any(i.get("purpose") == "maskable" for i in icons), "need a maskable icon"

    # Every declared PNG must exist at its true size — the service worker's
    # addAll fails the whole install if any shell asset 404s, and a mislabeled
    # size defeats the installability check.
    for sizes, icon in png.items():
        path = static_dir / icon["src"].lstrip("/")
        assert path.exists(), f"missing icon {icon['src']}"
        want = tuple(int(n) for n in sizes.split("x"))
        assert _png_size(path) == want, f"{icon['src']} is not {sizes}"


def test_focus_mode_markup_and_presence_wiring_present(static_dir: Path) -> None:
    # Phase C: the fullscreen focus overlay the companion JS drives must exist
    # in the shell (opened from the portrait), and the server-pushed-presence
    # consumer must be wired. Fast-gate insurance for the opt-in browser E2E.
    html = (static_dir / "index.html").read_text(encoding="utf-8")
    companion = (static_dir / "companion.js").read_text(encoding="utf-8")
    app = (static_dir / "app.js").read_text(encoding="utf-8")

    for needed in ('id="focus-mode"', 'id="focus-exit"', 'id="focus-avatar"',
                   'id="focus-presence"', 'id="focus-avatar-frame"'):
        assert needed in html, f"index.html missing {needed}"
    # The portrait is the (keyboard-reachable) opener.
    assert re.search(r'id="avatar-frame"[^>]*role="button"', html), (
        "the portrait must be an accessible focus-mode opener"
    )
    assert "setupFocusMode" in companion, "focus-mode wiring missing"
    # Server-pushed presence must be consumed by both the companion and the app.
    assert "setServerPresence" in companion, "companion must consume server presence"
    assert "setServerPresence" in app, "app must consume server presence"


def test_client_renders_resumed_transcript_from_offer_answer(static_dir: Path) -> None:
    # The resumed transcript rides on the /api/offer answer and is rendered
    # synchronously on connect. It must not depend on the data-channel replay
    # (pipecat drops it when the channel is slow to open on mobile), nor on an
    # async fetch (that raced reconnects and live turns — review 2026-07-11).
    app_js = (static_dir / "app.js").read_text(encoding="utf-8")

    assert "renderHistoryOnce(answer.turns" in app_js, (
        "connect must render the resumed transcript from the offer answer"
    )
    assert "loadConnectHistory" not in app_js, (
        "the racy async history fetch must be gone"
    )
