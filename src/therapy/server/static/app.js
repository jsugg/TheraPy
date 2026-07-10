/* TheraPy PWA client — vanilla WebRTC against SmallWebRTCTransport.
 *
 * Voice and text share one conversation: mic audio flows over the peer
 * connection; typed turns and transcripts flow over the "chat" data channel.
 * Reply modality mirrors the input by default — the server skips TTS for
 * typed turns entirely; the 🔊 toggle sends a "voice_replies" override so
 * the decision is server-side (SPEC §5). Local mute is instant feedback.
 */

const $ = (id) => document.getElementById(id);
const chat = $("chat");
const status = $("status");
const botAudio = $("bot-audio");

let pc = null;
let channel = null;
let micTrack = null;
let speakerOverride = null; // null = auto (mirror modality), true/false = user override

function setStatus(text, state = "idle") {
  status.textContent = text;
  status.dataset.state = state;
}

function addMessage(role, text, language) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  if (language) {
    const tag = document.createElement("span");
    tag.className = "lang";
    tag.textContent = language;
    div.appendChild(tag);
  }
  div.appendChild(document.createTextNode(text));
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

function applySpeaker(defaultOn) {
  const on = speakerOverride === null ? defaultOn : speakerOverride;
  botAudio.muted = !on;
  $("speaker").setAttribute("aria-pressed", String(on));
}

function sendSpeakerOverride() {
  if (!channel || channel.readyState !== "open") return;
  channel.send(JSON.stringify({ type: "voice_replies", enabled: speakerOverride }));
}

async function connect() {
  setStatus("connecting…");
  const media = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true },
  });
  micTrack = media.getAudioTracks()[0];

  pc = new RTCPeerConnection();
  pc.addTrack(micTrack, media);
  pc.addTransceiver("audio", { direction: "recvonly" });

  channel = pc.createDataChannel("chat", { ordered: true });
  channel.onmessage = (event) => {
    let msg;
    try { msg = JSON.parse(event.data); } catch { return; }
    if (msg.type === "transcript" && msg.text) {
      addMessage(msg.role, msg.text, msg.language);
    }
  };

  pc.ontrack = (event) => { botAudio.srcObject = event.streams[0]; };
  pc.onconnectionstatechange = () => {
    if (pc.connectionState === "connected") setStatus("listening", "listening");
    if (["failed", "disconnected", "closed"].includes(pc.connectionState)) {
      setStatus("disconnected");
      $("connect").hidden = false;
      $("controls").hidden = true;
    }
  };

  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  // Wait for ICE gathering so a single POST carries all candidates.
  await new Promise((resolve) => {
    if (pc.iceGatheringState === "complete") return resolve();
    pc.addEventListener("icegatheringstatechange", () => {
      if (pc.iceGatheringState === "complete") resolve();
    });
  });

  const response = await fetch("/api/offer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sdp: pc.localDescription.sdp,
      type: pc.localDescription.type,
    }),
  });
  await pc.setRemoteDescription(await response.json());

  $("connect").hidden = true;
  $("controls").hidden = false;
  applySpeaker(true);
}

function sendText() {
  const input = $("text");
  const text = input.value.trim();
  if (!text || !channel || channel.readyState !== "open") return;
  channel.send(JSON.stringify({ type: "user_text", text }));
  addMessage("user", text);
  input.value = "";
  applySpeaker(false); // typed turn → mirror to silent replies unless overridden
}

$("connect").addEventListener("click", () =>
  connect().catch((err) => setStatus(`error: ${err.message}`))
);
$("send").addEventListener("click", sendText);
$("text").addEventListener("keydown", (e) => { if (e.key === "Enter") sendText(); });

$("mic").addEventListener("click", () => {
  if (!micTrack) return;
  micTrack.enabled = !micTrack.enabled;
  $("mic").setAttribute("aria-pressed", String(micTrack.enabled));
  setStatus(micTrack.enabled ? "listening" : "mic off",
            micTrack.enabled ? "listening" : "idle");
  if (micTrack.enabled) applySpeaker(true); // back to voice → voice replies
});

$("speaker").addEventListener("click", () => {
  speakerOverride = botAudio.muted; // flip
  applySpeaker(speakerOverride);
  sendSpeakerOverride();
});

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("/sw.js");
}
