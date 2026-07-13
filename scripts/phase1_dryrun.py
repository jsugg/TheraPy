"""Phase-1 instrumented dry run (SPEC §9): drives the live server end-to-end.

A scripted WebRTC client (aiortc — already a pipecat dependency) holds one
continuous conversation against a running TheraPy server:

1. speaks one utterance in each of es/en/pt over the SAME connection,
   checking per-utterance language detection and voice switching;
2. reply-language auto mode (SPEC §7): a code-switched phrase whose
   minority-language words must NOT flip the reply language, then a phrase
   whose dominance flips mid-phrase and MUST flip it (the two normative
   SPEC §7 examples, spoken — the flip phrase is stitched from a Spanish
   voice and an English voice into one utterance);
3. sends a typed turn and verifies the reply is silent server-side
   (modality mirroring — no TTS audio should arrive);
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

    docker compose exec therapy uv run --no-dev python scripts/phase1_dryrun.py
"""

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from fractions import Fraction

import httpx
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamTrack
from av import AudioFrame
from kokoro_onnx import Kokoro

from pipecat.services.kokoro.tts import KOKORO_CACHE_DIR

SERVER = "http://localhost:8000"
RATE = 48_000
FRAME_SAMPLES = RATE // 50  # 20 ms
VOICED_RMS = 200.0  # int16 RMS above which a frame counts as bot speech

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

    transcripts: list[dict] = field(default_factory=list)
    voiced_at: list[float] = field(default_factory=list)
    transcript_event: asyncio.Event = field(default_factory=asyncio.Event)

    def first_voiced_after(self, t: float) -> float | None:
        return next((v for v in self.voiced_at if v >= t), None)

    def last_voiced(self) -> float | None:
        return self.voiced_at[-1] if self.voiced_at else None

    async def wait_transcript(self, role: str, after: int, timeout: float = 120.0) -> dict:
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
            except asyncio.TimeoutError:
                pass


async def watch_bot_audio(track: MediaStreamTrack, obs: Observations) -> None:
    while True:
        try:
            frame = await track.recv()
        except Exception:
            return
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
    checks = []
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
        f"     heard: {user_msg.get('text', '')!r}\n"
        f"     reply: {bot_msg.get('text', '')[:120]!r}"
    )


async def main() -> None:
    kokoro = load_user_voice()
    mic = UserAudioTrack()
    obs = Observations()
    results: list[str] = []

    pc = RTCPeerConnection()
    pc.addTrack(mic)

    channel = pc.createDataChannel("chat", ordered=True)
    channel.on("message", lambda raw: (
        obs.transcripts.append(json.loads(raw)) if '"transcript"' in raw else None,
        obs.transcript_event.set(),
    ))

    @pc.on("track")
    def on_track(track: MediaStreamTrack) -> None:
        if track.kind == "audio":
            asyncio.ensure_future(watch_bot_audio(track, obs))

    await pc.setLocalDescription(await pc.createOffer())
    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.05)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{SERVER}/api/offer",
            json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
            timeout=60,
        )
        response.raise_for_status()
        answer = response.json()
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))

    # Generous window: on a cold server the first connection also pays
    # whisper/kokoro model loading before the handshake completes.
    for _ in range(1200):
        if channel.readyState == "open":
            break
        await asyncio.sleep(0.05)
    else:
        sys.exit("Data channel never opened")
    print("Connected. Running trilingual spoken turns…", flush=True)

    # 1. One spoken turn per language, same connection (language switching).
    for label, voice, lang, text in UTTERANCES:
        await spoken_turn(
            label, synthesize(kokoro, text, voice, lang), mic, obs, results,
            expect_user=label, expect_reply=label,
        )
        print(results[-1], flush=True)
        await asyncio.sleep(1.0)

    # 2. Typed turn → transcript reply, and NO audio (server-side TTS skip).
    # First let the previous reply's audio finish draining, otherwise its
    # tail registers as a false "leak".
    while (last := obs.last_voiced()) and time.monotonic() - last < 2.0:
        await asyncio.sleep(0.2)
    seen = len(obs.transcripts)
    sent_at = time.monotonic()
    channel.send(json.dumps({"type": "user_text", "text": TYPED_TURN}))
    bot_msg = await obs.wait_transcript("assistant", seen)
    await asyncio.sleep(3.0)  # grace window in which audio must NOT appear
    leaked = obs.first_voiced_after(sent_at)
    silent = "silent ✓" if leaked is None else f"AUDIO LEAKED at +{leaked - sent_at:.2f}s"
    results.append(f"[typed] reply modality: {silent}\n     reply: {bot_msg.get('text', '')[:120]!r}")
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
        stopped = last is None or last < time.monotonic() - 1.5
        results.append(
            "[barge-in] reply audio stopped after interruption ✓" if stopped
            else f"[barge-in] audio still playing {time.monotonic() - last:.1f}s ago — check allow_interruptions"
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

    bad = ("MISMATCH", "NO AUDIO", "LEAKED", "SKIPPED", "still playing")
    failures = [line for line in results if any(marker in line for marker in bad)]
    if failures:
        sys.exit(f"\nFAIL — {len(failures)} scenario(s) not green.")
    print("\nPASS — all scenarios green.")


if __name__ == "__main__":
    asyncio.run(main())
