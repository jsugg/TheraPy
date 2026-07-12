"""Regression tests for static shell assets."""

import json
from pathlib import Path
import re
import struct


STATIC_DIR = Path(__file__).parents[1] / "src/therapy/server/static"


def _png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name} is not a PNG"
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def test_hidden_views_do_not_participate_in_layout() -> None:
    css = (STATIC_DIR / "styles.css").read_text(encoding="utf-8")

    assert re.search(r"\[hidden\]\s*\{\s*display:\s*none\s*!important\s*;", css), (
        "hidden chat panes must not flex and squeeze the history transcript"
    )


def test_shell_cache_is_bumped_after_static_shell_fix() -> None:
    service_worker = (STATIC_DIR / "sw.js").read_text(encoding="utf-8")

    assert "therapy-shell-v2" not in service_worker


def test_manifest_declares_installable_png_icons() -> None:
    # Chrome only offers "Install app" with raster 192px and 512px icons; the
    # SVG-only manifest gave no install option on the phone (field test
    # 2026-07-11). A maskable icon keeps Android from framing it on white.
    manifest = json.loads(
        (STATIC_DIR / "manifest.webmanifest").read_text(encoding="utf-8")
    )
    icons = manifest.get("icons", [])
    png = {i["sizes"]: i for i in icons if i.get("type") == "image/png"}

    assert "192x192" in png and "512x512" in png, "need 192 and 512 PNG icons"
    assert any(i.get("purpose") == "maskable" for i in icons), "need a maskable icon"

    # Every declared PNG must exist at its true size — the service worker's
    # addAll fails the whole install if any shell asset 404s, and a mislabeled
    # size defeats the installability check.
    for sizes, icon in png.items():
        path = STATIC_DIR / icon["src"].lstrip("/")
        assert path.exists(), f"missing icon {icon['src']}"
        want = tuple(int(n) for n in sizes.split("x"))
        assert _png_size(path) == want, f"{icon['src']} is not {sizes}"


def test_focus_mode_markup_and_presence_wiring_present() -> None:
    # Phase C: the fullscreen focus overlay the companion JS drives must exist
    # in the shell (opened from the portrait), and the server-pushed-presence
    # consumer must be wired. Fast-gate insurance for the opt-in browser E2E.
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    companion = (STATIC_DIR / "companion.js").read_text(encoding="utf-8")
    app = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    for needed in ('id="focus-mode"', 'id="focus-exit"', 'id="focus-avatar"',
                   'id="focus-presence"', 'id="focus-avatar-frame"'):
        assert needed in html, f"index.html missing {needed}"
    # The portrait is the (keyboard-reachable) opener.
    assert re.search(r'id="avatar-frame"[^>]*role="button"', html), (
        "the portrait must be an accessible focus-mode opener"
    )
    assert "setupFocusMode" in companion, "focus-mode wiring missing"
    assert "setServerPresence" in companion and "setServerPresence" in app, (
        "server-pushed presence must be consumed by the client"
    )


def test_client_renders_resumed_transcript_from_offer_answer() -> None:
    # The resumed transcript rides on the /api/offer answer and is rendered
    # synchronously on connect. It must not depend on the data-channel replay
    # (pipecat drops it when the channel is slow to open on mobile), nor on an
    # async fetch (that raced reconnects and live turns — review 2026-07-11).
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")

    assert "renderHistoryOnce(answer.turns" in app_js, (
        "connect must render the resumed transcript from the offer answer"
    )
    assert "loadConnectHistory" not in app_js, (
        "the racy async history fetch must be gone"
    )
