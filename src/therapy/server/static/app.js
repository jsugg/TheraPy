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
const historyButton = $("history");
const historyView = $("history-view");
const sessionList = $("session-list");
const sessionDetail = $("session-detail");
const sessionTurns = $("session-turns");
const botAudio = $("bot-audio");

let pc = null;
let channel = null;
let micTrack = null;
let speakerOverride = null; // null = auto (mirror modality), true/false = user override
let viewingSessionId = null; // session open in the history detail view

// Reply language (SPEC §7): "" = auto, or a pinned es/en/pt. Persists across
// visits and is re-sent on every connect — the server holds the live state.
const langSelect = $("lang");
langSelect.value = localStorage.getItem("replyLanguage") || "";

function sendReplyLanguage() {
  if (!channel || channel.readyState !== "open") return;
  channel.send(JSON.stringify({
    type: "reply_language",
    language: langSelect.value || null,
  }));
}

langSelect.addEventListener("change", () => {
  localStorage.setItem("replyLanguage", langSelect.value);
  sendReplyLanguage();
});

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

function firstLine(text) {
  return (text || "").split("\n")[0];
}

function sessionDate(value) {
  if (!value) return "Unknown date";
  return new Date(value).toLocaleString();
}

function renderSessionList(sessions) {
  sessionList.replaceChildren();
  if (!sessions.length) {
    const empty = document.createElement("div");
    empty.className = "session-row";
    empty.textContent = "No sessions yet.";
    sessionList.appendChild(empty);
    return;
  }

  for (const session of sessions) {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "session-row";

    const title = document.createElement("div");
    title.textContent = sessionDate(session.started_at);
    if (session.ended_at === null) {
      title.appendChild(document.createTextNode(" (active)"));
    }

    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `${session.turn_count || 0} turns`;

    row.appendChild(title);
    row.appendChild(meta);
    if (session.summary) {
      const summary = document.createElement("div");
      summary.textContent = firstLine(session.summary);
      row.appendChild(summary);
    }

    row.addEventListener("click", () => loadSession(session.id));
    sessionList.appendChild(row);
  }
}

function renderSessionTurns(turns) {
  sessionTurns.replaceChildren();
  for (const turn of turns) {
    const div = document.createElement("div");
    div.className = `msg ${turn.role}`;

    const tag = document.createElement("span");
    tag.className = "lang";
    tag.textContent = [turn.language, turn.modality].filter(Boolean).join(" · ");
    div.appendChild(tag);
    div.appendChild(document.createTextNode(turn.text));
    sessionTurns.appendChild(div);
  }
}

async function loadSessions() {
  sessionDetail.hidden = true;
  $("session-list-view").hidden = false;
  sessionList.textContent = "Loading sessions…";
  const response = await fetch("/api/sessions");
  if (!response.ok) throw new Error("Could not load sessions");
  const payload = await response.json();
  renderSessionList(payload.sessions || []);
}

async function loadSession(sessionId) {
  const response = await fetch(`/api/sessions/${encodeURIComponent(sessionId)}`);
  if (!response.ok) throw new Error("Could not load session");
  const payload = await response.json();
  viewingSessionId = sessionId;
  renderSessionTurns(payload.turns || []);
  $("session-list-view").hidden = true;
  sessionDetail.hidden = false;
}

function setHistoryVisible(open) {
  historyView.hidden = !open;
  chat.hidden = open;
  historyButton.setAttribute("aria-pressed", String(open));
  if (open) loadSessions().catch((err) => setStatus(`error: ${err.message}`));
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

async function iceServers() {
  // TURN relay for paths where host candidates can't reach the pipeline
  // (phone over Tailscale). The page's own hostname reaches the relay on
  // every network the PWA is served from; failure just means no relay
  // fallback, so direct paths still connect.
  try {
    const cfg = await (await fetch("/api/ice-config")).json();
    return [{
      urls: [
        `turn:${location.hostname}:${cfg.port}?transport=udp`,
        `turn:${location.hostname}:${cfg.port}?transport=tcp`,
      ],
      username: cfg.username,
      credential: cfg.credential,
    }];
  } catch {
    return [];
  }
}

async function connect(opts = {}) {
  setStatus("connecting…");
  if (pc) {
    try { pc.close(); } catch { /* already dead */ }
    pc = null;
  }
  const media = await navigator.mediaDevices.getUserMedia({
    audio: { echoCancellation: true, noiseSuppression: true },
  });
  micTrack = media.getAudioTracks()[0];

  pc = new RTCPeerConnection({ iceServers: await iceServers() });
  pc.addTrack(micTrack, media);
  pc.addTransceiver("audio", { direction: "recvonly" });

  channel = pc.createDataChannel("chat", { ordered: true });
  channel.onopen = () => {
    sendReplyLanguage(); // replay the persisted pin (SPEC §7)
    // Ask for server-truth chat state — the reply re-renders the resumed
    // transcript, so a drop or page reload doesn't show an empty pane.
    channel.send(JSON.stringify({ type: "client_ready" }));
  };
  channel.onmessage = (event) => {
    let msg;
    try { msg = JSON.parse(event.data); } catch { return; }
    if (msg.type === "transcript" && msg.text) {
      addMessage(msg.role, msg.text, msg.language);
    }
    if (msg.type === "session") {
      chat.replaceChildren();
      for (const turn of msg.turns || []) {
        addMessage(turn.role, turn.text, turn.language);
      }
      if (msg.resumed) setStatus("resumed", "listening");
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
  // Gather candidates for a single POST, but bounded: with a TURN server
  // configured, browsers can take many seconds to declare gathering
  // "complete" (e.g. a slow tcp variant) — 3 s of candidates is enough.
  await new Promise((resolve) => {
    const timer = setTimeout(resolve, 3000);
    const done = () => { clearTimeout(timer); resolve(); };
    if (pc.iceGatheringState === "complete") return done();
    pc.addEventListener("icegatheringstatechange", () => {
      if (pc.iceGatheringState === "complete") done();
    });
  });

  // Default: the server resumes a recently interrupted session. Explicit
  // choices from the history view override it in either direction.
  let offerUrl = "/api/offer";
  if (opts.sessionId) offerUrl += `?session=${encodeURIComponent(opts.sessionId)}`;
  else if (opts.newSession) offerUrl += "?new_session=1";
  const response = await fetch(offerUrl, {
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

function startConversation(opts = {}) {
  // Immediate feedback: the button yields to the status line right away;
  // it only comes back if the connection attempt fails.
  $("connect").hidden = true;
  connect(opts).catch((err) => {
    setStatus(`error: ${err.message}`);
    $("connect").hidden = false;
    $("controls").hidden = true;
  });
}

$("connect").addEventListener("click", () => startConversation());
$("send").addEventListener("click", sendText);
$("text").addEventListener("keydown", (e) => { if (e.key === "Enter") sendText(); });
$("history").addEventListener("click", () => setHistoryVisible(historyView.hidden));
$("history-back").addEventListener("click", () => {
  sessionDetail.hidden = true;
  $("session-list-view").hidden = false;
});

// Session management: start fresh, continue a chosen session, or erase one.
$("session-new").addEventListener("click", () => {
  setHistoryVisible(false);
  startConversation({ newSession: true });
});
$("session-resume").addEventListener("click", () => {
  if (!viewingSessionId) return;
  setHistoryVisible(false);
  startConversation({ sessionId: viewingSessionId });
});
$("session-delete").addEventListener("click", async () => {
  if (!viewingSessionId) return;
  if (!window.confirm("Delete this conversation and its audio for good?")) return;
  const response = await fetch(
    `/api/sessions/${encodeURIComponent(viewingSessionId)}`,
    { method: "DELETE" },
  );
  if (!response.ok) {
    const detail = (await response.json().catch(() => ({}))).detail;
    setStatus(`error: ${detail || "could not delete"}`);
    return;
  }
  viewingSessionId = null;
  loadSessions().catch((err) => setStatus(`error: ${err.message}`));
});

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
