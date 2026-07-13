# UX spec — companion presence & swappable avatars

**Status:** draft v1 (2026-07-10), from the owner's concept exploration (four
concept boards: eight input-mode layouts + avatar art, working name "Rowan").
Scoped as the next UI iteration after phases 1–2; no pipeline changes required.

## 1. Direction

The PWA gains a **companion presence**: a persistent avatar with a live status
("Listening", "Ready to listen", "Thinking…") replacing the bare text status.
This makes the assistant feel *present* without pretending to be a person —
consistent with the persona rule (SPEC §5): one stable character, adaptive
register. The avatar is a **skin**: it never changes the dialogue policy,
prompts, or the assistant's self-description. `dialogue/policy.py` is
untouched by this spec.

## 2. Avatar system (swappable)

- An avatar is a static asset pack under `server/static/avatars/<id>/`:

  ```
  avatars/rowan/
  ├── manifest.json    # {"id": "rowan", "name": "Rowan", "palette": {...}}
  ├── portrait.webp    # 512×512, the concept art
  └── portrait-sm.webp # 96×96 header chip / message avatar
  ```

- `manifest.json` may override the CSS accent palette (the concepts' deep
  forest green maps cleanly onto the existing `--accent`/`--accent-soft`
  variables — theming is palette swap, not a restyle).
- Selection persists client-side (`localStorage.avatar`, same pattern as
  `replyLanguage`); an avatar picker lives behind the header chip. Server
  needs nothing beyond serving the static files; `GET /api/avatars` (listing
  manifest files) is optional sugar for the picker.
- **Presence states** drive one visual: `offline · connecting · listening ·
  thinking · speaking`. All are already knowable client-side: connection
  state, mic activity, transcript-pending (user transcript arrived, assistant
  reply not yet), and bot-audio playing. v1 renders them as the concepts'
  status pill + a subtle ring pulse on the portrait; no server changes.
  (A server-pushed `state` data-channel message can refine this later.)
- The avatar's display name is UI-only in v1. Whether the assistant *answers*
  to the avatar name (persona naming) is a product decision deferred — it
  would touch the prompt layer, which this spec deliberately avoids.

## 3. Input-mode layout — decision

Evaluated the eight concepts against the SPEC §2 rule that voice and text are
**equal citizens in one conversation** (no mode to enter or leave):

| Concept | Verdict |
|---|---|
| 1 Minimal Toggle | ✅ **Base layout.** Closest to the current UI; text field + mic always visible; no modality wall. |
| 4 Hold to Talk | ✅ Optional *mic behavior* setting (open-mic vs push-to-talk) on the same layout — useful in noisy places, good for barge-in confidence. |
| 5 Voice-First Fullscreen | ✅ Later, as a "focus mode" you enter explicitly (tap the portrait); "tap to interrupt" maps to existing barge-in. |
| 2 Mode Switch Tabs / 7 Radial selector | ❌ Rejected: Write/Speak as *modes* contradicts mixing modalities mid-conversation. |
| 3 Slide-Up Speak / 6 Floating button / 8 Dual shortcuts | ❌ Rejected for v1: extra ceremony before speaking; revisit pieces if push-to-talk proves popular. |

Voice-message bubbles with play buttons (shown in every concept) map directly
onto phase-2 data: user utterance audio is already archived per turn — the
history view can offer playback of one's own turns without new capture work.

## 4. Phasing

- **A (small):** avatar packs + picker; presence pill + portrait in header
  (Minimal Toggle restyle); palette from manifest. Pure static/client work.
- **B:** push-to-talk option; voice-bubble playback in the transcript
  browser (serve archived WAVs via an authenticated-by-tailnet endpoint).
- **C:** fullscreen focus mode; server-pushed presence states; avatar
  reactions to register shifts (ser integration era, phase 3+).

Accessibility throughout: status changes announced via the existing
`aria-live` region; ring pulse honors `prefers-reduced-motion`; hit targets
≥44 px (the concepts' footer controls already comply).
