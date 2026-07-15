"""Full PWA browser end-to-end (Playwright, headless Chromium).

Covers what only a real browser can: PWA installability (the field-tested
gap — the manifest, service worker and icons Chrome checks before offering
"Install app") and the live UI flow (connect over WebRTC with a fake mic, a
typed turn, its transcript rendering, and the resume-label logic).

Run: `docker compose exec therapy uv run pytest -m e2e`
(a one-time `uv run playwright install chromium` populates the cache volume).
"""

import re
from typing import TYPE_CHECKING

import pytest

# Playwright ships only in the container test bed. Where it (and the realtime
# stack these E2E exercise) isn't installed, keep the module importable and skip
# each test at runtime — so `pytest -m e2e` still *collects* these items and
# exits 0 (all skipped) rather than returning "no tests collected". The fallback
# stays behind `TYPE_CHECKING` so the type checker only ever sees the real
# Playwright types.
if TYPE_CHECKING:
    from playwright.sync_api import Page, expect

    _HAS_PLAYWRIGHT = True
else:
    try:
        from playwright.sync_api import Page, expect

        _HAS_PLAYWRIGHT = True
    except ImportError:
        _HAS_PLAYWRIGHT = False
        Page = object

        def expect(*args, **kwargs):
            raise RuntimeError("playwright is not installed")


# The `e2e` marker is applied automatically by folder (tests/conftest.py); only
# the runtime skip when Playwright is absent needs declaring here.
pytestmark = pytest.mark.skipif(not _HAS_PLAYWRIGHT, reason="playwright not installed")


def test_chrome_reports_pwa_installable(page: Page, e2e_server: str) -> None:
    # Capture the install signal if the headless build fires it (it often
    # suppresses the event, so this is reported, not asserted).
    page.add_init_script(
        "window.__installable = false;"
        "window.addEventListener('beforeinstallprompt', () => { window.__installable = true; });"
    )
    page.goto(f"{e2e_server}/")

    # A registered, active service worker is one of Chrome's hard install
    # requirements; wait for it rather than assume.
    page.wait_for_function(
        "navigator.serviceWorker.ready.then(r => !!r.active)", timeout=20_000
    )
    # The worker must actually CONTROL the page (skipWaiting + clients.claim);
    # a worker that never takes over is a known way to lose the install option.
    page.wait_for_function("!!navigator.serviceWorker.controller", timeout=10_000)

    # Chrome's own verdict — the check the earlier structural test skipped, so
    # it wrongly reported the app installable. Empty means Chrome would offer
    # install (the headless build just never fires the banner event).
    cdp = page.context.new_cdp_session(page)
    errors = cdp.send("Page.getInstallabilityErrors")["installabilityErrors"]
    assert errors == [], f"Chrome reports installability blockers: {errors}"

    manifest = page.evaluate(
        """async () => {
            const link = document.querySelector('link[rel=manifest]');
            const res = await fetch(link.href);
            return { ok: res.ok, type: res.headers.get('content-type'),
                     body: res.ok ? await res.json() : null };
        }"""
    )
    assert manifest["ok"], "manifest did not load"
    assert "application/manifest+json" in (manifest["type"] or "")

    icons = manifest["body"]["icons"]
    png = {i["sizes"]: i for i in icons if i.get("type") == "image/png"}
    assert "192x192" in png, "Chrome needs a 192px PNG"
    assert "512x512" in png, "Chrome needs a 512px PNG"
    assert any(i.get("purpose") == "maskable" for i in icons), "need a maskable icon"

    # Every declared PNG must actually decode in the browser at its true size —
    # a 404 or a mislabeled size defeats the install check even if the JSON
    # looks right.
    for sizes, icon in png.items():
        want = [int(n) for n in sizes.split("x")]
        got = page.evaluate(
            """async (src) => {
                const img = new Image();
                const p = new Promise((res, rej) => {
                    img.onload = () => res([img.naturalWidth, img.naturalHeight]);
                    img.onerror = () => rej(new Error('icon failed to load'));
                });
                img.src = src;
                return await p;
            }""",
            icon["src"],
        )
        assert got == want, f"{icon['src']} decoded as {got}, expected {want}"


def test_live_connection_renders_transcript_presence_and_resume_flow(
    page: Page, e2e_server: str
) -> None:
    # Prove presence is *server-pushed*, not merely inferred (phase C): tap the
    # chat data channel and record every message the client receives on it.
    page.add_init_script(
        """
        window.__dcMessages = [];
        const origCreate = RTCPeerConnection.prototype.createDataChannel;
        RTCPeerConnection.prototype.createDataChannel = function (...args) {
          const channel = origCreate.apply(this, args);
          channel.addEventListener('message', (event) => {
            try { window.__dcMessages.push(JSON.parse(event.data)); } catch (e) {}
          });
          return channel;
        };
        """
    )
    # Fresh isolated store → nothing to resume yet.
    page.goto(f"{e2e_server}/")
    expect(page.locator("#connect")).to_have_text("Start conversation", timeout=15_000)

    page.locator("#connect").click()
    # WebRTC establishes (status flips to "listening"); the connect button hides.
    expect(page.locator("#status")).to_have_text("listening", timeout=60_000)
    expect(page.locator("#controls")).to_be_visible()

    page.locator("#text").fill("Hello from the browser test.")
    page.locator("#send").click()

    # The user's typed turn renders immediately, client-side.
    expect(page.locator("#chat .msg.user").last).to_contain_text(
        "Hello from the browser test."
    )
    # The assistant reply renders from the data-channel transcript — the whole
    # point of the render path. Generous timeout: the isolated server pays a
    # cold model load and gemma runs on CPU.
    expect(page.locator("#chat .msg.assistant").last).to_be_visible(timeout=120_000)

    # Companion presence layer is live and driven off the machine status: the
    # pill left "offline" once connected, and the assistant bubble is decorated
    # with the avatar's name (companion.js observing #status and #chat).
    expect(page.locator("#presence")).to_have_attribute(
        "data-presence", re.compile(r"listening|speaking|thinking")
    )
    expect(page.locator("#chat .msg.assistant .assistant-name").last).to_have_text(
        "Rowan"
    )

    # The pipeline actually pushed authoritative presence over the data channel:
    # the typed turn drives thinking→listening server-side, so at least one
    # presence message with a server-witnessable state must have arrived. This
    # is what distinguishes the pushed path from the client-side inference.
    pushed_states = page.evaluate(
        "() => (window.__dcMessages || [])"
        ".filter((m) => m && m.type === 'presence').map((m) => m.state)"
    )
    assert pushed_states, "no server-pushed presence message arrived"
    assert set(pushed_states) <= {"listening", "thinking", "speaking"}, pushed_states

    # Push-to-talk is an opt-in mic mode: toggling it reveals the Hold button;
    # deep hold/long-press behavior has focused coverage elsewhere.
    expect(page.locator("#talk")).to_be_hidden()
    page.locator("#mic-mode").click()
    expect(page.locator("#talk")).to_be_visible()
    page.locator("#mic-mode").click()
    expect(page.locator("#talk")).to_be_hidden()

    # A session with real turns now exists, so reconnecting would resume it:
    # the landing button must say so (the empty-probe guard, Hardening 9).
    page.goto(f"{e2e_server}/")
    expect(page.locator("#connect")).to_have_text("Resume conversation", timeout=15_000)

    # Resuming renders the prior transcript synchronously from the offer answer
    # (the raceless path that replaced the async fetch — review 2026-07-11).
    page.locator("#connect").click()
    expect(page.locator("#chat .msg.user").first).to_contain_text(
        "Hello from the browser test.", timeout=60_000
    )


def test_avatar_picker_swaps_and_persists_selected_skin(
    page: Page, e2e_server: str
) -> None:
    # No WebRTC needed — the companion presence is header chrome, always on.
    page.goto(f"{e2e_server}/")

    # The default portrait actually decodes (a 404/mislabel would show 0).
    loaded = page.evaluate(
        """() => {
            const img = document.getElementById('avatar');
            return { w: img.naturalWidth, src: img.currentSrc || img.src };
        }"""
    )
    assert loaded["w"] > 0, "avatar portrait did not decode"
    assert "/avatars/rowan/" in loaded["src"], loaded["src"]

    # The picker lists both shipped skins; choosing one swaps the portrait and
    # persists the choice (localStorage, same pattern as the reply-language pin).
    page.locator("#avatar-pick").click()
    expect(page.locator("#avatar-picker")).to_be_visible()
    expect(page.locator("#avatar-picker .avatar-option")).to_have_count(2)

    page.locator('#avatar-picker .avatar-option[data-avatar-id="luna"]').click()
    page.wait_for_function(
        """() => {
            const img = document.getElementById('avatar');
            return (img.currentSrc || img.src).includes('/avatars/luna/');
        }""",
        timeout=10_000,
    )
    assert page.evaluate("() => localStorage.getItem('avatar')") == "luna"


def test_focus_mode_opens_scroll_locks_and_closes(
    page: Page, e2e_server: str
) -> None:
    # Tapping the portrait enters the immersive, voice-first fullscreen view
    # (phase C). No WebRTC needed — it's presentation over the always-on
    # companion; barge-in still works by speaking, so there is no interrupt
    # control to assert. Here we prove the overlay opens, decodes, scroll-locks
    # the page behind, and closes on Escape.
    page.goto(f"{e2e_server}/")

    overlay = page.locator("#focus-mode")
    expect(overlay).to_be_hidden()

    page.locator("#avatar-frame").click()
    expect(overlay).to_be_visible()
    expect(page.locator("#focus-exit")).to_be_visible()

    # The big portrait actually decodes and mirrors the current avatar.
    focus_avatar = page.evaluate(
        """() => {
            const i = document.getElementById('focus-avatar');
            return { w: i.naturalWidth, src: i.currentSrc || i.src };
        }"""
    )
    assert focus_avatar["w"] > 0, "focus portrait did not decode"
    assert "/avatars/rowan/" in focus_avatar["src"], focus_avatar["src"]

    label = page.evaluate(
        "() => document.getElementById('focus-presence').textContent.trim()"
    )
    assert label, "focus presence label is empty"

    # The page behind is scroll-locked while focus mode is open.
    assert page.evaluate(
        "() => document.documentElement.classList.contains('focus-active')"
    )

    # Escape leaves focus mode and releases the scroll lock.
    page.keyboard.press("Escape")
    expect(overlay).to_be_hidden()
    assert not page.evaluate(
        "() => document.documentElement.classList.contains('focus-active')"
    )


def test_history_view_hides_composer_start_cta(page: Page, e2e_server: str) -> None:
    # Landing shows the composer CTA; opening history must not leave a second
    # "start" button alongside its own "New conversation" (field report
    # 2026-07-11 — two buttons doing the same thing).
    page.goto(f"{e2e_server}/")
    expect(page.locator("#connect")).to_be_visible()

    page.locator("#history").click()
    expect(page.locator("#session-new")).to_be_visible()
    expect(page.locator("#composer")).to_be_hidden()
    expect(page.locator("#connect")).to_be_hidden()

    # Closing history restores the single landing CTA.
    page.locator("#history").click()
    expect(page.locator("#composer")).to_be_visible()
    expect(page.locator("#connect")).to_be_visible()


def test_hold_to_talk_handles_longpress_and_pointer_lifecycle(
    page: Page, e2e_server: str
) -> None:
    # The field bug: on a phone, press-and-hold pops the native selection menu
    # ("Copy / Select all / Share"), which cancels the touch → pointercancel →
    # endHold, so "listening" stops the instant the menu appears. A browser can't
    # render that OS menu, but it CAN lock in the DOM/CSS contract that prevents
    # it, plus the hold state machine. No WebRTC/model needed — the hold logic is
    # independent of the pipeline, so we reveal the controls and drive it directly.
    page.goto(f"{e2e_server}/")
    expect(page.locator("#mic-mode")).to_have_text("Open mic", timeout=15_000)
    # The controls row is hidden until a live connection; reveal it to exercise
    # the button without connecting.
    page.evaluate("() => { document.getElementById('controls').hidden = false; }")
    page.locator("#mic-mode").click()  # → push mode reveals the Hold button
    talk = page.locator("#talk")
    expect(talk).to_be_visible()

    # 1) Gesture-claiming opt-out: default `touch-action: manipulation` (inherited
    # from button) lets a slight drift become a scroll → pointercancel. `none`
    # keeps the pointer with the button for the whole hold.
    assert talk.evaluate("el => getComputedStyle(el).touchAction") == "none"
    # 2) Text is not selectable (the long-press selection that pops the menu).
    assert talk.evaluate("el => getComputedStyle(el).userSelect") in (
        "none",
        "-webkit-none",
    )
    # 3) The menu triggers via contextmenu (Android) / selectstart — both cancelled.
    assert talk.evaluate(
        "el => !el.dispatchEvent("
        "new MouseEvent('contextmenu', {bubbles: true, cancelable: true}))"
    ), "contextmenu must be prevented on the hold button"
    assert talk.evaluate(
        "el => !el.dispatchEvent("
        "new Event('selectstart', {bubbles: true, cancelable: true}))"
    ), "selectstart must be prevented on the hold button"

    primary = {"pointerId": 1, "isPrimary": True, "button": 0}

    # Hold state machine: press holds, release releases.
    talk.dispatch_event("pointerdown", primary)
    expect(talk).to_have_attribute("aria-pressed", "true")
    assert "is-holding" in (talk.get_attribute("class") or "")
    talk.dispatch_event("pointerup", {"pointerId": 1})
    expect(talk).to_have_attribute("aria-pressed", "false")

    # Multi-touch: a SECOND finger's release must not end the first finger's hold.
    talk.dispatch_event("pointerdown", primary)
    expect(talk).to_have_attribute("aria-pressed", "true")
    talk.dispatch_event("pointerdown", {"pointerId": 2, "isPrimary": False, "button": 0})
    talk.dispatch_event("pointerup", {"pointerId": 2})
    expect(talk).to_have_attribute("aria-pressed", "true")  # still held
    talk.dispatch_event("pointerup", {"pointerId": 1})
    expect(talk).to_have_attribute("aria-pressed", "false")

    # A genuinely cancelled gesture must fail safe by ending the hold (mute).
    talk.dispatch_event("pointerdown", primary)
    expect(talk).to_have_attribute("aria-pressed", "true")
    talk.dispatch_event("pointercancel", {"pointerId": 1})
    expect(talk).to_have_attribute("aria-pressed", "false")

    # A non-primary contact / secondary mouse button must never start a hold.
    talk.dispatch_event("pointerdown", {"pointerId": 3, "isPrimary": True, "button": 2})
    expect(talk).to_have_attribute("aria-pressed", "false")


def test_composer_stays_sticky_at_viewport_bottom(
    page: Page, e2e_server: str
) -> None:
    # The composer (buttons + text input) must stay pinned to the bottom, mirroring
    # the sticky header — field report: the footer scrolled away.
    page.goto(f"{e2e_server}/")
    composer = page.locator("#composer")
    expect(composer).to_be_visible()
    assert composer.evaluate("el => getComputedStyle(el).position") == "sticky"
    assert composer.evaluate("el => getComputedStyle(el).bottom") == "0px"
