"""WebRTC connectivity check from the client's network position.

The in-container dry run can't see host↔container ICE problems (it connects
from inside the network the server advertises). This probe runs where a real
client runs — the host, or any tailnet machine with the repo — and verifies
the handshake end to end: signaling, ICE, data channel.

    python scripts/verify_relay_connectivity.py [--server URL] [--relay-only]

`--relay-only` strips every non-relay candidate from the client's offer, so
the connection can only succeed through the TURN relay (compose `turn`
service). This simulates the phone-over-Tailscale case, where the server's
container-internal host candidates are unreachable and the relay is the only
viable path — the regression check for "connecting → disconnected" on mobile.
"""

import argparse
import asyncio
import json
import sys
import time
from fractions import Fraction
from importlib.metadata import PackageNotFoundError, version
from typing import Literal

import httpx
import numpy as np
from aiortc import (
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.mediastreams import MediaStreamTrack
from av import AudioFrame

RATE = 48_000
FRAME_SAMPLES = RATE // 50  # 20 ms
SCRIPT_NAME = "verify_relay_connectivity"
SCENARIOS = frozenset({"direct", "relay-only"})

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


class SilentMic(MediaStreamTrack):
    """Paced silence — clients always send a mic track."""

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


def strip_non_relay(sdp: str) -> str:
    """Drop host/srflx candidates so only the TURN relay path remains."""
    kept = [
        line
        for line in sdp.splitlines()
        if not line.startswith("a=candidate") or " typ relay " in f"{line} "
    ]
    return "\r\n".join(kept) + "\r\n"


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--relay-only", action="store_true")
    args = parser.parse_args()

    host = httpx.URL(args.server).host
    async with httpx.AsyncClient() as client:
        ice = (await client.get(f"{args.server}/api/ice-config", timeout=15)).json()
    servers = [
        RTCIceServer(
            urls=[f"turn:{host}:{ice['port']}?transport=udp"],
            username=ice["username"],
            credential=ice["credential"],
        )
    ]

    pc = RTCPeerConnection(RTCConfiguration(iceServers=servers))
    pc.addTrack(SilentMic())
    channel = pc.createDataChannel("chat", ordered=True)

    @pc.on("iceconnectionstatechange")
    def _on_ice() -> None:
        print(f"ICE: {pc.iceConnectionState}", flush=True)

    _ = _on_ice

    await pc.setLocalDescription(await pc.createOffer())
    while pc.iceGatheringState != "complete":
        await asyncio.sleep(0.05)

    sdp = pc.localDescription.sdp
    if args.relay_only:
        relay_count = sum(" typ relay" in line for line in sdp.splitlines())
        if not relay_count:
            sys.exit("FAIL: no relay candidates gathered — TURN unreachable or bad creds")
        sdp = strip_non_relay(sdp)
        print(f"relay-only: offering {relay_count} relay candidate(s)", flush=True)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            # A connectivity probe must never resume (and re-finalize) a
            # real conversation left within the resume window.
            f"{args.server}/api/offer?new_session=1",
            json={"sdp": sdp, "type": pc.localDescription.type},
            timeout=60,
        )
        response.raise_for_status()
        answer = response.json()
    await pc.setRemoteDescription(
        RTCSessionDescription(sdp=answer["sdp"], type=answer["type"])
    )

    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if channel.readyState == "open":
            print(f"PASS: data channel open ({'relay' if args.relay_only else 'any'} path)")
            await pc.close()
            return
        if pc.connectionState in ("failed", "closed"):
            sys.exit(f"FAIL: connection {pc.connectionState}")
        await asyncio.sleep(0.2)
    sys.exit(f"FAIL: timeout (conn={pc.connectionState}, channel={channel.readyState})")


def _run_with_record() -> None:
    """Run the verifier and always leave its machine record last on stdout."""
    started = time.monotonic()
    result: VerificationResult = "fail"
    scenario = "relay-only" if "--relay-only" in sys.argv[1:] else "direct"
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
            scenario=scenario,
            duration_s=time.monotonic() - started,
            result=result,
        )
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    _run_with_record()
