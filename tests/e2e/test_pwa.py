"""Full PWA browser end-to-end (Playwright, headless Chromium).

Covers what only a real browser can: PWA installability (the field-tested
gap — the manifest, service worker and icons Chrome checks before offering
"Install app") and the live UI flow (connect over WebRTC with a fake mic, a
typed turn, its transcript rendering, and the resume-label logic).

Run: `docker compose exec therapy uv run pytest -m e2e`
(a one-time `uv run playwright install chromium` populates the cache volume).
"""

import re

import pytest
from playwright.sync_api import Page, expect

pytestmark = pytest.mark.e2e


def test_1_pwa_is_installable(page: Page, e2e_server: str) -> None:
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
    assert "192x192" in png and "512x512" in png, "Chrome needs 192 and 512 PNGs"
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


def test_2_connect_typed_turn_transcript_and_resume_label(
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

    # Push-to-talk is an opt-in mic mode: toggling it reveals the Hold button
    # (the mic-track gating itself is unit-covered; here we prove the UI wiring).
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


def test_3_companion_avatar_renders_and_swaps(page: Page, e2e_server: str) -> None:
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


def test_5_focus_mode_opens_decodes_and_closes(page: Page, e2e_server: str) -> None:
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


def test_4_history_view_hides_the_start_cta(page: Page, e2e_server: str) -> None:
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
