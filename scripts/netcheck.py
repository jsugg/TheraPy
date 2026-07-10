"""WebRTC connectivity check from the client's network position.

The in-container dry run can't see host↔container ICE problems (it connects
from inside the network the server advertises). This probe runs where a real
client runs — the host, or any tailnet machine with the repo — and verifies
the handshake end to end: signaling, ICE, data channel.

    python scripts/netcheck.py [--server URL] [--relay-only]

`--relay-only` strips every non-relay candidate from the client's offer, so
the connection can only succeed through the TURN relay (compose `turn`
service). This simulates the phone-over-Tailscale case, where the server's
container-internal host candidates are unreachable and the relay is the only
viable path — the regression check for "connecting → disconnected" on mobile.
"""

import argparse
import asyncio
import sys
import time
from fractions import Fraction

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
    def on_ice() -> None:
        print(f"ICE: {pc.iceConnectionState}", flush=True)

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


if __name__ == "__main__":
    asyncio.run(main())
