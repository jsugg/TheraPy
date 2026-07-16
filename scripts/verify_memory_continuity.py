"""Memory-continuity verification (SPEC §9): sessions + data round-trip.

Scripted client against the live server, typed turns only (memory is
modality-agnostic, and text keeps the run fast on CPU):

1. Session A states a distinctive personal fact and disconnects.
2. Waits until the server has summarized and closed session A
   (GET /api/sessions — summarization runs in the background).
3. Session B (a fresh connection) asks about the fact and asserts the
   assistant's reply references it — continuity via distilled summaries +
   user-model facts, never verbatim history (SPEC §8).
4. Export round-trip: `python -m therapy.memory export` contains the fact.
5. Delete round-trip: `... delete --yes` leaves no sessions behind
   (verified via a fresh export AND /api/sessions).

Run inside the container:

    docker compose exec therapy uv run --no-dev python scripts/verify_memory_continuity.py
"""

import asyncio
import json
import subprocess
import sys
import time
from fractions import Fraction
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

import httpx
import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamTrack
from av import AudioFrame

SERVER = "http://localhost:8000"
RATE = 48_000
FRAME_SAMPLES = RATE // 50  # 20 ms
FACT_TOKEN = "Nebulosa"  # distinctive enough to be unambiguous in replies
SCRIPT_NAME = "verify_memory_continuity"
SCENARIOS = frozenset({"memory-continuity"})

type VerificationResult = Literal["pass", "fail"]


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


SESSION_A_TURNS = [
    f"Hola, te quería contar algo: acabo de adoptar una perrita y le puse {FACT_TOKEN}.",
    f"Sí, {FACT_TOKEN} es una cachorra mestiza, la encontré cerca del trabajo.",
]
SESSION_B_QUESTION = "¿Te acuerdas del nombre de mi perrita? ¿Cómo se llama?"


class SilentMic(MediaStreamTrack):
    """Paced silence — the PWA always sends a mic track, so this client does too."""

    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self._pts = 0
        self._t0: float | None = None

    async def recv(self) -> AudioFrame:
        if self._t0 is None:
            self._t0 = time.monotonic()
        delay = self._t0 + self._pts / RATE - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        frame = AudioFrame.from_ndarray(
            np.zeros((1, FRAME_SAMPLES), dtype=np.int16), format="s16", layout="mono"
        )
        frame.sample_rate = RATE
        frame.pts = self._pts
        frame.time_base = Fraction(1, RATE)
        self._pts += FRAME_SAMPLES
        return frame


class TypedClient:
    """Minimal typed-turn client: silent mic, data channel for conversation."""

    def __init__(self) -> None:
        self.pc = RTCPeerConnection()
        self.pc.addTrack(SilentMic())
        self.channel = self.pc.createDataChannel("chat", ordered=True)
        self.transcripts: list[dict] = []
        self.event = asyncio.Event()
        self.channel.on("message", self._on_message)

    def _on_message(self, raw: str) -> None:
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            return
        if message.get("type") == "transcript":
            self.transcripts.append(message)
            self.event.set()
        if message.get("type") == "session":
            self.session_state = message
            self.event.set()

    async def request_session_state(self, timeout: float = 30.0) -> dict:
        """Ask for the server-truth chat state, like the PWA on channel open."""
        self.session_state: dict | None = None
        self.channel.send(json.dumps({"type": "client_ready"}))
        deadline = time.monotonic() + timeout
        while self.session_state is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                sys.exit(f"FAIL: no session state within {timeout}s")
            self.event.clear()
            try:
                await asyncio.wait_for(self.event.wait(), remaining)
            except TimeoutError:
                pass
        return self.session_state

    async def connect(self, *, new_session: bool = True) -> None:
        # recvonly audio keeps the server's pipeline shape identical to the PWA.
        # new_session=True isolates scenarios from reconnect-resume; the
        # resume scenario itself connects with False, like the real PWA.
        self.pc.addTransceiver("audio", direction="recvonly")
        await self.pc.setLocalDescription(await self.pc.createOffer())
        while self.pc.iceGatheringState != "complete":
            await asyncio.sleep(0.05)
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{SERVER}/api/offer" + ("?new_session=1" if new_session else ""),
                json={
                    "sdp": self.pc.localDescription.sdp,
                    "type": self.pc.localDescription.type,
                },
                timeout=60,
            )
            response.raise_for_status()
            answer = response.json()
        await self.pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
        )
        for _ in range(1200):
            if self.channel.readyState == "open":
                return
            await asyncio.sleep(0.05)
        sys.exit("FAIL: data channel never opened")

    async def ask(self, text: str, timeout: float = 180.0) -> str:
        """Send a typed turn; return the assistant's reply text."""
        seen = len(self.transcripts)
        self.channel.send(json.dumps({"type": "user_text", "text": text}))
        deadline = time.monotonic() + timeout
        while True:
            for message in self.transcripts[seen:]:
                if message.get("role") == "assistant":
                    return str(message.get("text", ""))
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                sys.exit(f"FAIL: no assistant reply within {timeout}s")
            self.event.clear()
            try:
                await asyncio.wait_for(self.event.wait(), remaining)
            except TimeoutError:
                pass

    async def close(self) -> None:
        await self.pc.close()


async def list_sessions() -> list[dict]:
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{SERVER}/api/sessions", timeout=30)
        response.raise_for_status()
        return response.json()["sessions"]


async def wait_for_summary(deadline_s: float = 240.0) -> dict:
    """Wait until the NEWEST session is ended and summarized.

    Older summarized sessions (e.g. dry runs) may exist — only the session
    that just disconnected counts, and it is first in the newest-first list.
    """
    deadline = time.monotonic() + deadline_s
    async with httpx.AsyncClient() as client:
        while time.monotonic() < deadline:
            response = await client.get(f"{SERVER}/api/sessions", timeout=30)
            response.raise_for_status()
            sessions = response.json()["sessions"]
            if sessions and sessions[0]["ended_at"] and sessions[0]["summary"]:
                return sessions[0]
            await asyncio.sleep(3.0)
    sys.exit(f"FAIL: newest session not summarized within {deadline_s}s")


def run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "therapy.memory", *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


async def main() -> None:
    results: list[str] = []

    # 1. Session A: state the fact, get an acknowledgement, disconnect.
    a = TypedClient()
    await a.connect()
    for turn in SESSION_A_TURNS:
        reply = await a.ask(turn)
        print(f"[A] user: {turn!r}\n    assistant: {reply[:100]!r}", flush=True)
    await a.close()
    results.append("[session A] fact stated, replies received ✓")

    # 1b. Reconnect-resume: a drop + reconnect within the resume window must
    #     continue session A, not open an amnesiac new one (regression:
    #     field test 2026-07-10). Deterministic check — same newest session
    #     id, no new session row, still answering. This reconnect also
    #     exercises cancelling session A's in-flight finalization.
    before = await list_sessions()
    a2 = TypedClient()
    await a2.connect(new_session=False)
    state = await a2.request_session_state()
    replayed = (
        state.get("resumed") is True
        and state.get("session_id") == before[0]["id"]
        and len(state.get("turns") or []) >= 4  # 2 user + 2 assistant from A
    )
    results.append(
        "[replay] client received the resumed transcript ✓"
        if replayed
        else f"[replay] FAIL — {json.dumps(state)[:160]}"
    )
    reply = await a2.ask("¿Sigues ahí? Se cortó la conexión un momento.")
    print(f"[resume] assistant: {reply[:100]!r}", flush=True)
    await a2.close()
    after = await list_sessions()
    if len(after) == len(before) and after[0]["id"] == before[0]["id"]:
        results.append("[resume] reconnect continued the same session ✓")
    else:
        results.append(
            f"[resume] FAIL — sessions {len(before)}→{len(after)}, "
            f"newest {before[0]['id'][:8]}→{after[0]['id'][:8]}"
        )

    # 2. Background summarization must finish before B connects — continuity
    #    is injected at connect time.
    summarized = await wait_for_summary()
    print(f"[summary] {str(summarized['summary'])[:160]!r}", flush=True)
    ok = FACT_TOKEN.lower() in str(summarized["summary"]).lower()
    results.append(
        f"[summary] session A summarized; fact token "
        f"{'present ✓' if ok else 'MISSING (relying on facts table)'}"
    )

    # 3. Session B: fresh connection, ask about the fact.
    b = TypedClient()
    await b.connect()
    reply = await b.ask(SESSION_B_QUESTION)
    print(f"[B] assistant: {reply!r}", flush=True)
    await b.close()
    if FACT_TOKEN.lower() not in reply.lower():
        results.append(f"[continuity] FAIL — reply does not mention {FACT_TOKEN!r}")
    else:
        results.append(f"[continuity] assistant recalled {FACT_TOKEN!r} across sessions ✓")

    # Session B's own background finalization must land before the delete
    # round-trip, or a late end_session/fact write races the wipe.
    await wait_for_summary()

    # 4. Export round-trip: the fact must be in the JSON snapshot.
    export = run_cli("export")
    if export.returncode != 0:
        results.append(f"[export] FAIL — exit {export.returncode}: {export.stderr[:200]}")
    else:
        snapshot = json.loads(export.stdout)
        n_sessions = len(snapshot["sessions"])
        found = FACT_TOKEN.lower() in export.stdout.lower()
        results.append(
            f"[export] {n_sessions} session(s); fact "
            f"{'present ✓' if found else 'MISSING — FAIL'}"
        )

    # 5. Delete round-trip: refuse without --yes, wipe with it, verify empty.
    refused = run_cli("delete")
    guarded = refused.returncode == 2
    deleted = run_cli("delete", "--yes")
    post = json.loads(run_cli("export").stdout)
    async with httpx.AsyncClient() as client:
        api_sessions = (await client.get(f"{SERVER}/api/sessions", timeout=30)).json()[
            "sessions"
        ]
    empty = deleted.returncode == 0 and not post["sessions"] and not api_sessions
    results.append(
        f"[delete] guard {'✓' if guarded else 'FAIL'} | "
        f"wipe {'verified empty ✓' if empty else 'FAIL — data remains'}"
    )

    print("\n=== Memory-continuity verification summary ===")
    for line in results:
        print(line)
    if any("FAIL" in line for line in results):
        sys.exit("\nFAIL — memory-continuity verification not green.")
    print("\nPASS — continuity, export, and delete all verified.")


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
            scenario="memory-continuity",
            duration_s=time.monotonic() - started,
            result=result,
        )
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    _run_with_record()
