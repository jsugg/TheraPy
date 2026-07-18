"""Voice/text loop verification (SPEC §9): drives the live server end-to-end.

A scripted WebRTC client (aiortc — already a pipecat dependency) holds one
continuous conversation against a running TheraPy server:

1. speaks one utterance in each of es/en/pt over the SAME connection,
   checking per-utterance language detection and voice switching;
2. reply-language auto mode (SPEC §7): a code-switched phrase whose
   minority-language words must NOT flip the reply language, then a phrase
   whose dominance flips mid-phrase and MUST flip it (the two normative
   SPEC §7 examples, spoken — the flip phrase is stitched from a Spanish
   voice and an English voice into one utterance);
3. sends a typed turn and verifies it gets a text reply (whether audio
   accompanies it is the user's speaker-toggle choice, so it is not gated);
4. speaks again while the assistant reply is still playing (barge-in) and
   checks the reply audio stops;
5. pinned mode (SPEC §7): pins replies to pt over the data channel, speaks
   Spanish, and checks the transcript stays es (STT auto) while the reply
   comes back pt; unpins and checks auto resumes.

Per spoken turn it reports client-side TTFA (end of user speech → first
voiced audio frame); the server logs its own TTFA per turn (risk R1).
User utterances are synthesized with the same Kokoro weights the server
uses, with distinct "user" voices, so the run needs no microphone.

Run inside the container (models must be cached; see README):

    docker compose exec therapy uv run --no-dev python scripts/verify_voice_text_loop.py
"""

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from fractions import Fraction
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

import httpx
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamTrack
from av import AudioFrame
from kokoro_onnx import Kokoro
from pipecat.services.kokoro.tts import KOKORO_CACHE_DIR

from therapy.observability.interactions import JsonValue, require_json_object

SERVER = "http://localhost:8000"
RATE = 48_000
FRAME_SAMPLES = RATE // 50  # 20 ms
VOICED_RMS = 200.0  # int16 RMS above which a frame counts as bot speech
SCRIPT_NAME = "verify_voice_text_loop"
SCENARIOS = frozenset({"voice-text-loop"})

type VerificationResult = Literal["pass", "fail"]


def _text(value: JsonValue | object) -> str:
    """Return a protocol value only when it is text."""
    return value if isinstance(value, str) else ""


def build_verification_record(
    *, scenario: str, duration_s: float, result: VerificationResult
) -> dict[str, str | float]:
    """Build the final bounded verification record."""
    if scenario not in SCENARIOS:
        raise ValueError("unsupported verification scenario")
    try:
        build = version("therapy")
    except PackageNotFoundError:
        build = "unknown"
    return {
        "record": "verification",
        "script": SCRIPT_NAME,
        "build": build,
        "scenario": scenario,
        "duration_s": duration_s,
        "result": result,
        "environment": "test",
    }

# Distinct male "user" voices so the harness never hears itself in the
# assistant's (female) voices.
UTTERANCES = [
    ("es", "em_alex", "es",
     "Hola. Hoy fue un día bastante largo en el trabajo, pero por fin terminé el proyecto."),
    ("en", "am_adam", "en-us",
     "Hi there. Today was a long day at work, but I finally finished the project I mentioned."),
    ("pt", "pm_alex", "pt-br",
     "Oi. Hoje foi um dia bem longo no trabalho, mas finalmente terminei o projeto."),
]
TYPED_TURN = "¿Puedes resumir en una frase lo que te conté hoy?"
BARGE_IN = ("en", "am_adam", "en-us", "Sorry to interrupt, can you keep it shorter please?")

# SPEC §7 normative reply-language phrases, spoken (segments are stitched
# into ONE utterance; each segment uses the voice of its own language).
CODE_SWITCH_KEEP = [("em_alex", "es", "Todo bien, me estoy sintiendo ok.")]
CODE_SWITCH_FLIP = [
    ("em_alex", "es", "Sí, todo bien…"),
    ("am_adam", "en-us", "though… no… actually things have been hard lately."),
]
PIN_UTTERANCE = ("em_alex", "es", "Hoy me costó mucho concentrarme en la oficina.")
UNPIN_UTTERANCE = ("em_alex", "es", "Gracias, sigamos en español entonces.")


class UserAudioTrack(MediaStreamTrack):
    """Outbound mic replacement: paced 20 ms frames — utterances, else silence."""

    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self._pending = np.empty(0, dtype=np.int16)
        self._pts = 0
        self._t0: float | None = None
        self.speech_ended_at: float | None = None

    def feed(self, samples: np.ndarray) -> None:
        self._pending = np.concatenate([self._pending, samples])
        self.speech_ended_at = None

    @property
    def speaking(self) -> bool:
        return self._pending.size > 0

    async def recv(self) -> AudioFrame:
        if self._t0 is None:
            self._t0 = time.monotonic()
        target = self._t0 + self._pts / RATE
        delay = target - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)

        if self._pending.size:
            chunk = self._pending[:FRAME_SAMPLES]
            self._pending = self._pending[FRAME_SAMPLES:]
            if chunk.size < FRAME_SAMPLES:
                chunk = np.pad(chunk, (0, FRAME_SAMPLES - chunk.size))
            if not self._pending.size:
                self.speech_ended_at = time.monotonic()
        else:
            chunk = np.zeros(FRAME_SAMPLES, dtype=np.int16)

        frame = AudioFrame.from_ndarray(chunk.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = RATE
        frame.pts = self._pts
        frame.time_base = Fraction(1, RATE)
        self._pts += FRAME_SAMPLES
        return frame


@dataclass
class Observations:
    """Everything the harness hears and reads back from the server."""

    transcripts: list[dict[str, JsonValue]] = field(
        default_factory=list[dict[str, JsonValue]]
    )
    voiced_at: list[float] = field(default_factory=list[float])
    transcript_event: asyncio.Event = field(default_factory=asyncio.Event)

    def first_voiced_after(self, t: float) -> float | None:
        return next((v for v in self.voiced_at if v >= t), None)

    def last_voiced(self) -> float | None:
        return self.voiced_at[-1] if self.voiced_at else None

    async def wait_transcript(
        self, role: str, after: int, timeout: float = 120.0
    ) -> dict[str, JsonValue]:
        deadline = time.monotonic() + timeout
        while True:
            for msg in self.transcripts[after:]:
                if msg.get("role") == role:
                    return msg
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"No {role!r} transcript within {timeout}s")
            self.transcript_event.clear()
            try:
                await asyncio.wait_for(self.transcript_event.wait(), remaining)
            except TimeoutError:
                pass


async def watch_bot_audio(track: MediaStreamTrack, obs: Observations) -> None:
    while True:
        try:
            frame = await track.recv()
        except Exception:
            return
        if not isinstance(frame, AudioFrame):
            continue
        samples = frame.to_ndarray().astype(np.float32)
        if np.sqrt(np.mean(samples**2)) > VOICED_RMS:
            obs.voiced_at.append(time.monotonic())


def load_user_voice() -> Kokoro:
    model = KOKORO_CACHE_DIR / "kokoro-v1.0.onnx"
    voices = KOKORO_CACHE_DIR / "voices-v1.0.bin"
    if not (model.exists() and voices.exists()):
        sys.exit("Kokoro weights not cached yet — start a conversation once, or pre-warm.")
    return Kokoro(str(model), str(voices))


def synthesize(kokoro: Kokoro, text: str, voice: str, lang: str) -> np.ndarray:
    samples, sr = kokoro.create(text, voice=voice, speed=1.0, lang=lang)
    # Linear-interp resample to the WebRTC clock; fine for STT purposes.
    n = int(len(samples) * RATE / sr)
    resampled = np.interp(
        np.linspace(0, len(samples), n, endpoint=False), np.arange(len(samples)), samples
    )
    return (np.clip(resampled, -1, 1) * 32767).astype(np.int16)


def _trim_silence(samples: np.ndarray, threshold: int = 300) -> np.ndarray:
    voiced = np.where(np.abs(samples) > threshold)[0]
    return samples if voiced.size == 0 else samples[voiced[0] : voiced[-1] + 1]


def synthesize_segments(kokoro: Kokoro, segments: list[tuple[str, str, str]]) -> np.ndarray:
    """One code-switched utterance from per-language segments.

    Trailing synthesis silence is trimmed and segments are joined with a
    150 ms gap — well under the server's 0.7 s VAD stop, so the whole
    phrase lands as a single turn.
    """
    gap = np.zeros(int(0.15 * RATE), dtype=np.int16)
    parts: list[np.ndarray] = []
    for voice, lang, text in segments:
        if parts:
            parts.append(gap)
        parts.append(_trim_silence(synthesize(kokoro, text, voice, lang)))
    return np.concatenate(parts)


async def wait_speech_end(mic: UserAudioTrack) -> float:
    while mic.speaking or mic.speech_ended_at is None:
        await asyncio.sleep(0.05)
    return mic.speech_ended_at


async def spoken_turn(
    label: str,
    audio: np.ndarray,
    mic: UserAudioTrack,
    obs: Observations,
    results: list[str],
    *,
    expect_user: str | None = None,
    expect_reply: str | None = None,
) -> None:
    """One spoken turn: feed audio, await transcripts, check languages + TTFA.

    `expect_user` checks STT's detected language; `expect_reply` checks the
    server's chosen reply language (SPEC §7). Either may be None to skip —
    e.g. Whisper's whole-utterance label is unspecified for a code-switched
    phrase, where only the reply language is normative.
    """
    seen = len(obs.transcripts)
    mic.feed(audio)
    ended = await wait_speech_end(mic)

    user_msg = await obs.wait_transcript("user", seen)
    bot_msg = await obs.wait_transcript("assistant", seen)
    first_audio = None
    for _ in range(600):  # up to 60 s for slow CPU synthesis
        first_audio = obs.first_voiced_after(ended)
        if first_audio:
            break
        await asyncio.sleep(0.1)

    ttfa = f"{first_audio - ended:.2f}s" if first_audio else "NO AUDIO"
    checks: list[str] = []
    if expect_user:
        detected = user_msg.get("language")
        checks.append("stt: ok" if detected == expect_user else f"stt: MISMATCH ({detected})")
    if expect_reply:
        reply_lang = bot_msg.get("language")
        checks.append(
            f"reply-lang: ok ({reply_lang})" if reply_lang == expect_reply
            else f"reply-lang: MISMATCH (want {expect_reply}, got {reply_lang})"
        )
    results.append(
        f"[{label}] {' | '.join(checks)} | client TTFA (incl. 0.7s VAD close): {ttfa}\n"
        f"     heard: {_text(user_msg.get('text'))!r}\n"
        f"     reply: {_text(bot_msg.get('text'))[:120]!r}"
    )


async def main() -> None:
    kokoro = load_user_voice()
    mic = UserAudioTrack()
    obs = Observations()
    results: list[str] = []

    pc = RTCPeerConnection()
    pc.addTrack(mic)

    channel = pc.createDataChannel("chat", ordered=True)

    def on_message(raw: str | bytes) -> None:
        if isinstance(raw, str) and '"transcript"' in raw:
            payload: object = json.loads(raw)
            obs.transcripts.append(
                require_json_object(payload, "data-channel transcript")
            )
        obs.transcript_event.set()

    channel.on("message", on_message)

    @pc.on("track")
    def _on_track(track: MediaStreamTrack) -> None:
        if track.kind == "audio":
            asyncio.ensure_future(watch_bot_audio(track, obs))

    _ = _on_track

    await pc.setLocalDescription(await pc.createOffer())
    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.05)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            # new_session=1: scenarios assume a fresh context — never resume
            # whatever session a previous run or field test left behind.
            f"{SERVER}/api/offer?new_session=1",
            json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
            timeout=60,
        )
        response.raise_for_status()
        answer_payload: object = response.json()
        answer = require_json_object(answer_payload, "offer answer")
    answer_sdp = answer.get("sdp")
    answer_type = answer.get("type")
    if not isinstance(answer_sdp, str) or not isinstance(answer_type, str):
        raise TypeError("offer answer must contain string sdp and type")
    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=answer_sdp, type=answer_type)
    )

    # Generous window: on a cold server the first connection also pays
    # whisper/kokoro model loading before the handshake completes.
    for _ in range(1200):
        if channel.readyState == "open":
            break
        await asyncio.sleep(0.05)
    else:
        sys.exit("Data channel never opened")
    print("Connected. Running trilingual spoken turns…", flush=True)

    # Mirror the PWA, which replays its reply-language selector on connect
    # (sendReplyLanguage). Auto (null) must NOT anchor a language before the
    # user has spoken — asserting it injected an English "user is now speaking
    # English" note that answered a Spanish first turn in English (field test
    # 2026-07-11). Sending it here is the end-to-end regression for that fix:
    # the es turn below comes back English without it. The script diverged
    # from the real client by skipping this, which is how the bug reached the
    # field despite a green dry run.
    channel.send(json.dumps({"type": "reply_language", "language": None}))
    await asyncio.sleep(0.5)

    # 1. One spoken turn per language, same connection (language switching).
    for label, voice, lang, text in UTTERANCES:
        await spoken_turn(
            label, synthesize(kokoro, text, voice, lang), mic, obs, results,
            expect_user=label, expect_reply=label,
        )
        print(results[-1], flush=True)
        await asyncio.sleep(1.0)

    # 2. Typed turn → text reply. The reply modality mirrors the input by
    # default (typed → silent), but audio on a typed reply is fine — the user's
    # speaker toggle is the audio control, so it is observed, not gated (owner,
    # 2026-07-11). Gate only on the text reply arriving.
    while (last := obs.last_voiced()) and time.monotonic() - last < 2.0:
        await asyncio.sleep(0.2)
    seen = len(obs.transcripts)
    sent_at = time.monotonic()
    channel.send(json.dumps({"type": "user_text", "text": TYPED_TURN}))
    bot_msg = await obs.wait_transcript("assistant", seen)
    await asyncio.sleep(2.0)
    modality = "silent (mirrored)" if obs.first_voiced_after(sent_at) is None else "with audio"
    bot_text = _text(bot_msg.get("text"))
    status = "text reply ✓" if bot_text else "NO TEXT REPLY"
    results.append(f"[typed] {status} — {modality}\n     reply: {bot_text[:120]!r}")
    print(results[-1], flush=True)

    # 3. Voice turn again (mirroring must restore speech), used for barge-in:
    #    interrupt as soon as the reply audio starts.
    label, voice, lang, text = BARGE_IN
    seen = len(obs.transcripts)
    mic.feed(synthesize(kokoro, "Now tell me, in detail, everything a good weekly routine needs.", "am_adam", "en-us"))
    ended = await wait_speech_end(mic)
    await obs.wait_transcript("assistant", seen, timeout=180)
    for _ in range(600):
        if obs.first_voiced_after(ended):
            break
        await asyncio.sleep(0.1)
    if not obs.first_voiced_after(ended):
        results.append("[barge-in] SKIPPED — voice turn after typed turn produced no audio (mirroring broken?)")
    else:
        seen_interrupt = len(obs.transcripts)
        mic.feed(synthesize(kokoro, text, voice, lang))
        await wait_speech_end(mic)
        await asyncio.sleep(2.5)
        last = obs.last_voiced()
        elapsed = None if last is None else time.monotonic() - last
        if elapsed is None or elapsed > 1.5:
            results.append("[barge-in] reply audio stopped after interruption ✓")
        else:
            results.append(
                f"[barge-in] audio still playing {elapsed:.1f}s ago — "
                "check allow_interruptions"
            )
        # Drain the interruption's own turn — its transcript and reply arrive
        # late and must not bleed into the next scenario's assertions.
        await obs.wait_transcript("user", seen_interrupt)
        await obs.wait_transcript("assistant", seen_interrupt)
        while (last := obs.last_voiced()) and time.monotonic() - last < 3.0:
            await asyncio.sleep(0.2)
    print(results[-1], flush=True)

    # 4. Reply-language auto mode (SPEC §7 normative examples, spoken).
    #    Re-anchor in Spanish first, so the code-switched phrase tests
    #    "minority words don't flip" rather than a plain dominance switch.
    print("Reply-language scenarios…", flush=True)
    await asyncio.sleep(1.0)
    await spoken_turn(
        "anchor-es",
        synthesize(kokoro, "Bueno, volvamos al español un momento.", "em_alex", "es"),
        mic, obs, results, expect_user="es", expect_reply="es",
    )
    print(results[-1], flush=True)
    await asyncio.sleep(1.0)
    # (a) es-dominant phrase with a lone English word — must NOT flip.
    await spoken_turn(
        "code-switch-keep", synthesize_segments(kokoro, CODE_SWITCH_KEEP),
        mic, obs, results, expect_user="es", expect_reply="es",
    )
    print(results[-1], flush=True)
    await asyncio.sleep(1.0)
    # (b) dominance flips to English mid-phrase — reply MUST flip. Whisper's
    # whole-utterance label is unspecified for a mixed phrase; only the
    # reply language is asserted.
    await spoken_turn(
        "code-switch-flip", synthesize_segments(kokoro, CODE_SWITCH_FLIP),
        mic, obs, results, expect_reply="en",
    )
    print(results[-1], flush=True)

    # 5. Pinned mode (SPEC §7): replies come back in the pinned language
    #    while STT keeps auto-detecting; unpin restores auto.
    await asyncio.sleep(1.0)
    channel.send(json.dumps({"type": "reply_language", "language": "pt"}))
    await asyncio.sleep(1.0)  # let the override land before the next turn
    voice, lang, text = PIN_UTTERANCE
    await spoken_turn(
        "pinned-pt", synthesize(kokoro, text, voice, lang),
        mic, obs, results, expect_user="es", expect_reply="pt",
    )
    print(results[-1], flush=True)
    await asyncio.sleep(1.0)
    channel.send(json.dumps({"type": "reply_language", "language": None}))
    await asyncio.sleep(1.0)
    voice, lang, text = UNPIN_UTTERANCE
    await spoken_turn(
        "unpinned-auto", synthesize(kokoro, text, voice, lang),
        mic, obs, results, expect_user="es", expect_reply="es",
    )
    print(results[-1], flush=True)

    print("\n=== Phase-1 dry run summary ===")
    for line in results:
        print(line)
    print("\nServer-side TTFA lines: docker compose logs therapy | grep TTFA")

    await pc.close()

    bad = ("MISMATCH", "NO AUDIO", "NO TEXT REPLY", "SKIPPED", "still playing")
    failures = [line for line in results if any(marker in line for marker in bad)]
    if failures:
        sys.exit(f"\nFAIL — {len(failures)} scenario(s) not green.")
    print("\nPASS — all scenarios green.")


def _run_with_record() -> None:
    """Run the verifier and always leave its machine record last on stdout."""
    started = time.monotonic()
    result: VerificationResult = "fail"
    try:
        asyncio.run(main())
    except SystemExit as exc:
        if exc.code in (None, 0):
            result = "pass"
        raise
    else:
        result = "pass"
    finally:
        record = build_verification_record(
            scenario="voice-text-loop",
            duration_s=time.monotonic() - started,
            result=result,
        )
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    _run_with_record()
