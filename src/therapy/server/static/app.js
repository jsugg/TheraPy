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
const workspace = $("model-workspace");
const workspaceButton = $("model-workspace-button");

const telemetry = (() => {
  const ENDPOINT = "/api/telemetry/client";
  const MAX_EVENTS = 50;
  const BATCH_SIZE = 20;
  const MAX_AGE_MS = 24 * 60 * 60 * 1000;
  const EVENT_NAMES = new Set([
    "media_permission", "signaling_state", "ice_state", "data_channel_state",
    "peer_state", "transcript_echo_timeout", "playback_failure", "disconnect",
    "webrtc_sample", "sw_lifecycle", "shell_fetch", "cache_fallback",
    "cache_recovery", "push_lifecycle",
  ]);
  const OUTCOMES = new Set([
    "success", "error", "timeout", "fallback", "recovered", "denied",
    "granted", "received", "shown", "clicked", "connected", "disconnected",
    "failed", "installed", "activated", "refreshed", "deactivated",
  ]);
  const NUMBER_FIELDS = {
    duration_ms: [0, 600000, false],
    rtt_ms: [0, 60000, false],
    jitter_ms: [0, 10000, false],
    packet_loss_ratio: [0, 1, false],
    bitrate_kbps: [0, 1000000, false],
    bytes_delta: [0, 10000000000, true],
    concealed_samples: [0, 1000000000, true],
    dropped_events: [0, 10000, true],
  };
  const CANDIDATE_TYPES = new Set(["relay", "host", "srflx"]);
  const randomHex = (byteLength) => {
    const bytes = new Uint8Array(byteLength);
    do crypto.getRandomValues(bytes); while (bytes.every((byte) => byte === 0));
    return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
  };
  const traceparent = `00-${randomHex(16)}-${randomHex(8)}-01`;
  let queue = [];
  let droppedEvents = 0;
  let enabled = null;
  let flushing = null;

  // Only bounded schema fields cross this privacy boundary; all other input is ignored.
  function sanitize(raw) {
    if (!raw || typeof raw !== "object" || !EVENT_NAMES.has(raw.name)) return null;
    const event = {
      name: raw.name,
      outcome: OUTCOMES.has(raw.outcome) ? raw.outcome : "success",
    };
    for (const [field, [minimum, maximum, integer]] of Object.entries(NUMBER_FIELDS)) {
      const value = raw[field];
      if (typeof value !== "number" || !Number.isFinite(value)) continue;
      if (value < minimum || value > maximum || (integer && !Number.isInteger(value))) continue;
      event[field] = value;
    }
    if (CANDIDATE_TYPES.has(raw.candidate_type)) {
      event.candidate_type = raw.candidate_type;
    }
    return event;
  }

  function attachDroppedEvents(event) {
    if (!droppedEvents) return;
    const available = 10000 - (event.dropped_events || 0);
    const attached = Math.min(available, droppedEvents);
    if (attached > 0) {
      event.dropped_events = (event.dropped_events || 0) + attached;
      droppedEvents -= attached;
    }
  }

  function enqueue(raw) {
    if (enabled === false) return false;
    const event = sanitize(raw);
    if (!event) return false;
    if (queue.length === MAX_EVENTS) {
      const dropped = queue.shift();
      droppedEvents += 1 + (dropped.event.dropped_events || 0);
    }
    attachDroppedEvents(event);
    queue.push({ event, queuedAt: Date.now() });
    if (queue.length >= BATCH_SIZE) void flush();
    return true;
  }

  function expireOldEvents() {
    const oldest = Date.now() - MAX_AGE_MS;
    queue = queue.filter((entry) => {
      if (entry.queuedAt >= oldest) return true;
      droppedEvents += 1 + (entry.event.dropped_events || 0);
      return false;
    });
  }

  async function flush() {
    if (enabled === false) return null;
    if (flushing) return flushing;
    expireOldEvents();
    if (!queue.length) return null;
    const entries = queue.slice(0, BATCH_SIZE);
    const request = (async () => {
      try {
        const response = await fetch(ENDPOINT, {
          method: "POST",
          headers: { "Content-Type": "application/json", traceparent },
          body: JSON.stringify({
            schema_version: 1,
            events: entries.map((entry) => entry.event),
          }),
          keepalive: true,
        });
        if (response.status === 404) {
          enabled = false;
          queue = [];
          droppedEvents = 0;
          return response.status;
        }
        if (!response.ok) return response.status;
        enabled = true;
        const sent = new Set(entries);
        queue = queue.filter((entry) => !sent.has(entry));
        return response.status;
      } catch {
        return null;
      }
    })();
    flushing = request;
    try {
      return await request;
    } finally {
      flushing = null;
      if (enabled !== false && queue.length >= BATCH_SIZE) void flush();
    }
  }

  return Object.freeze({
    enqueue,
    flush,
    get size() { return queue.length; },
    get traceparent() { return traceparent; },
  });
})();

window.telemetry = telemetry;
botAudio.addEventListener("error", () => {
  telemetry.enqueue({ name: "playback_failure", outcome: "error" });
});
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "hidden") void telemetry.flush();
});
navigator.serviceWorker?.addEventListener("message", (event) => {
  if (event.data?.type === "telemetry") telemetry.enqueue(event.data.event);
});

const NODE_TYPES = [
  "identity_fact", "value", "goal", "pattern", "preference", "thread",
  "person", "strength", "strategy", "thought_record", "boundary",
];
const EDGE_TYPES = [
  "involves", "triggers", "soothes", "works_for", "failed_for", "supports",
  "conflicts_with", "instance_of", "about",
];
let graphPayload = { nodes: [], edges: [], boundaries: [], pending_insights: [] };

let pc = null;
let channel = null;
let micTrack = null;
let peerConnectionId = null;
let speakerOverride = null; // null = auto (mirror modality), true/false = user override
let viewingSessionId = null; // session open in the history detail view
let historyLoaded = false; // initial transcript rendered for the current connection
let pushToTalk = false; // push-to-talk: mic gated until the Hold button is held
const pendingTypedEchoes = [];
const intentionalConnections = new WeakSet();
const disconnectedConnections = new WeakSet();
const statsStates = new WeakMap();

setInterval(() => {
  if (pc?.connectionState === "connected") void telemetry.flush();
}, 30000);

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

// Resuming must not look like starting fresh — label the button by what
// connecting will actually do (the server decides via the resume window).
async function refreshConnectLabel() {
  try {
    const state = await (await fetch("/api/resumable")).json();
    $("connect").textContent = state.session_id
      ? "Resume conversation"
      : "Start conversation";
  } catch { /* server unreachable — leave the current label */ }
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
  return div;
}

function reconcileTypedEcho(text, language) {
  const index = pendingTypedEchoes.findIndex((pending) => pending.text === text);
  if (index < 0) return false;
  const [{ element, timer }] = pendingTypedEchoes.splice(index, 1);
  clearTimeout(timer);
  element.removeAttribute("data-awaiting-echo");
  if (language && !element.querySelector(".lang")) {
    const tag = document.createElement("span");
    tag.className = "lang";
    tag.textContent = language;
    element.prepend(tag);
  }
  return true;
}

// Render the resumed transcript once per connection. The offer answer carries
// it (rendered synchronously on connect, below); the data-channel `session`
// replay is a deduped fallback. `historyLoaded` guards against the fallback
// re-rendering, and rendering before any live turn keeps ordering correct.
function renderHistoryOnce(turns, resumed) {
  if (historyLoaded) return;
  historyLoaded = true;
  for (const turn of turns) addMessage(turn.role, turn.text, turn.language);
  if (resumed) setStatus("resumed", "listening");
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

    // Title first (auto-generated topic, or the date until one exists);
    // the full summary lives in the detail view, not the list.
    const title = document.createElement("div");
    title.className = "title";
    title.textContent = session.title || sessionDate(session.started_at);
    if (session.ended_at === null) {
      title.appendChild(document.createTextNode(" (active)"));
    }

    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent =
      `${sessionDate(session.started_at)} · ${session.turn_count || 0} turns`;

    row.appendChild(title);
    row.appendChild(meta);

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
    // Own voice turns are replayable from the archive (SPEC §8): companion.js
    // renders the control, the server streams the WAV by turn id.
    if (turn.has_audio && window.Companion) {
      Companion.addPlayButton(
        div,
        `/api/sessions/${encodeURIComponent(viewingSessionId)}/turns/${turn.id}/audio`,
      );
    }
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
  const session = payload.session || {};
  $("session-title").textContent =
    session.title || sessionDate(session.started_at);
  renderSessionTurns(payload.turns || []);
  $("session-list-view").hidden = true;
  sessionDetail.hidden = false;
}

function setHistoryVisible(open) {
  if (open) setWorkspaceVisible(false);
  historyView.hidden = !open;
  chat.hidden = open;
  // The history view has its own "New conversation" action, so hide the footer
  // composer/CTA while browsing — no duplicate "start" button on screen.
  $("composer").hidden = open;
  historyButton.setAttribute("aria-pressed", String(open));
  if (open) loadSessions().catch((err) => setStatus(`error: ${err.message}`));
}

function setWorkspaceVisible(open) {
  workspace.hidden = !open;
  if (open) {
    historyView.hidden = true;
    historyButton.setAttribute("aria-pressed", "false");
  }
  chat.hidden = open;
  $("composer").hidden = open;
  workspaceButton.setAttribute("aria-pressed", String(open));
  if (open) loadWorkspace().catch((err) => setStatus(`error: ${err.message}`));
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

function signalingOutcome(state) {
  if (state === "stable") return "connected";
  if (state === "closed") return "disconnected";
  if ([
    "have-local-offer", "have-remote-offer",
    "have-local-pranswer", "have-remote-pranswer",
  ].includes(state)) return "success";
  return "error";
}

function connectionOutcome(state) {
  if (["connected", "completed"].includes(state)) return "connected";
  if (state === "failed") return "failed";
  if (["disconnected", "closed"].includes(state)) return "disconnected";
  if (["new", "checking", "connecting"].includes(state)) return "success";
  return "error";
}

function getStatsState(connection) {
  let state = statsStates.get(connection);
  if (!state) {
    state = {
      timer: null,
      inFlight: null,
      finalSampled: false,
      previousBytes: null,
      previousAt: null,
    };
    statsStates.set(connection, state);
  }
  return state;
}

function selectedCandidatePair(report) {
  let selectedPair = null;
  for (const stat of report.values()) {
    if (stat.type === "transport" && stat.selectedCandidatePairId) {
      selectedPair = report.get(stat.selectedCandidatePairId) || selectedPair;
    }
    if (
      !selectedPair && stat.type === "candidate-pair" &&
      (stat.selected || (stat.nominated && stat.state === "succeeded"))
    ) {
      selectedPair = stat;
    }
  }
  return selectedPair;
}

function buildWebRtcSample(report, state) {
  const event = { name: "webrtc_sample", outcome: "success" };
  let bytesReceived = 0;
  let hasBytes = false;
  let concealedSamples = 0;
  let hasConcealedSamples = false;
  let packetsLost = 0;
  let packetsReceived = 0;
  let hasPackets = false;
  const selectedPair = selectedCandidatePair(report);
  let rtt = Number.isFinite(selectedPair?.currentRoundTripTime)
    ? selectedPair.currentRoundTripTime : null;
  let jitter = null;

  for (const stat of report.values()) {
    if (stat.type === "remote-inbound-rtp" && Number.isFinite(stat.currentRoundTripTime)) {
      rtt ??= stat.currentRoundTripTime;
    }
    if (stat.type !== "inbound-rtp") continue;
    if (stat.kind && stat.kind !== "audio") continue;
    if (stat.mediaType && stat.mediaType !== "audio") continue;
    if (Number.isFinite(stat.jitter)) jitter ??= stat.jitter;
    if (Number.isFinite(stat.bytesReceived)) {
      bytesReceived += stat.bytesReceived;
      hasBytes = true;
    }
    if (Number.isFinite(stat.concealedSamples)) {
      concealedSamples += stat.concealedSamples;
      hasConcealedSamples = true;
    }
    if (Number.isFinite(stat.packetsLost) && Number.isFinite(stat.packetsReceived)) {
      packetsLost += Math.max(0, stat.packetsLost);
      packetsReceived += Math.max(0, stat.packetsReceived);
      hasPackets = true;
    }
  }

  if (rtt !== null) event.rtt_ms = rtt * 1000;
  if (jitter !== null) event.jitter_ms = jitter * 1000;
  if (hasPackets && packetsLost + packetsReceived > 0) {
    event.packet_loss_ratio = Math.min(1, Math.max(0,
      packetsLost / (packetsLost + packetsReceived)));
  }
  if (hasConcealedSamples) {
    event.concealed_samples = Math.min(1000000000, Math.round(concealedSamples));
  }
  const sampledAt = performance.now();
  if (hasBytes && state.previousBytes !== null && state.previousAt !== null) {
    const bytesDelta = Math.min(
      10000000000,
      Math.max(0, Math.round(bytesReceived - state.previousBytes)),
    );
    const elapsedMs = sampledAt - state.previousAt;
    event.bytes_delta = bytesDelta;
    if (elapsedMs > 0) event.bitrate_kbps = bytesDelta * 8 / elapsedMs;
  }
  if (hasBytes) {
    state.previousBytes = bytesReceived;
    state.previousAt = sampledAt;
  }
  const localCandidate = selectedPair?.localCandidateId
    ? report.get(selectedPair.localCandidateId) : null;
  const candidateType = localCandidate?.candidateType;
  if (["relay", "host", "srflx"].includes(candidateType)) {
    event.candidate_type = candidateType;
  }
  // Candidate IDs are used only for this lookup; no network or media identifiers leave the page.
  return event;
}

function sampleWebRtcStats(connection) {
  const state = getStatsState(connection);
  if (state.inFlight) return state.inFlight;
  const request = (async () => {
    try {
      telemetry.enqueue(buildWebRtcSample(await connection.getStats(), state));
    } catch {
      telemetry.enqueue({ name: "webrtc_sample", outcome: "error" });
    }
  })();
  state.inFlight = request;
  void request.finally(() => {
    if (state.inFlight === request) state.inFlight = null;
  });
  return request;
}

function startStatsSampling(connection) {
  const state = getStatsState(connection);
  if (state.timer !== null) return;
  state.timer = setInterval(() => {
    if (connection.connectionState === "connected") void sampleWebRtcStats(connection);
  }, 12000);
}

function stopStatsSampling(connection, finalSample) {
  const state = getStatsState(connection);
  if (state.timer !== null) clearInterval(state.timer);
  state.timer = null;
  if (!finalSample || state.finalSampled) return;
  state.finalSampled = true;
  let finalRequest;
  if (state.inFlight) {
    finalRequest = state.inFlight.then(() => sampleWebRtcStats(connection));
  } else {
    finalRequest = sampleWebRtcStats(connection);
  }
  void finalRequest.then(() => telemetry.flush());
}

function disconnectConversation() {
  const connection = pc;
  const peerId = peerConnectionId;
  pc = null;
  peerConnectionId = null;
  if (channel) channel.close();
  channel = null;
  if (micTrack) micTrack.stop();
  micTrack = null;
  if (connection) {
    intentionalConnections.add(connection);
    stopStatsSampling(connection, true);
    if (!disconnectedConnections.has(connection)) {
      disconnectedConnections.add(connection);
      telemetry.enqueue({ name: "disconnect", outcome: "success" });
    }
    connection.close();
  }
  botAudio.srcObject = null;
  return peerId;
}

async function disconnectVoice() {
  const peerId = disconnectConversation();
  if (!peerId) return;
  await api(`/api/voice/disconnect?pc_id=${encodeURIComponent(peerId)}`, { method: "POST" });
}

window.addEventListener("pagehide", () => {
  const peerId = disconnectConversation();
  if (peerId) {
    void fetch(`/api/voice/disconnect?pc_id=${encodeURIComponent(peerId)}`, {
      method: "POST", keepalive: true,
    });
  }
  void telemetry.flush();
});

async function connect(opts = {}) {
  setStatus("connecting…");
  historyLoaded = false;
  chat.replaceChildren(); // history reloads below (HTTP) or via the replay
  disconnectConversation();
  let media;
  try {
    media = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true },
    });
    telemetry.enqueue({ name: "media_permission", outcome: "granted" });
  } catch (error) {
    telemetry.enqueue({ name: "media_permission", outcome: "denied" });
    throw error;
  }
  micTrack = media.getAudioTracks()[0];
  // Honor the persisted mic mode for this fresh track — push mode starts muted
  // until the user holds Talk (companion.js reflects the mode on #mic-mode).
  pushToTalk = $("mic-mode")?.getAttribute("aria-pressed") === "true";
  applyMicMode();

  pc = new RTCPeerConnection({ iceServers: await iceServers() });
  pc.addTrack(micTrack, media);
  pc.addTransceiver("audio", { direction: "recvonly" });

  channel = pc.createDataChannel("chat", { ordered: true });
  let offerPostedAt = null;
  channel.onopen = () => {
    telemetry.enqueue({
      name: "data_channel_state",
      outcome: "connected",
      duration_ms: offerPostedAt === null
        ? undefined : Math.min(600000, performance.now() - offerPostedAt),
    });
    sendReplyLanguage(); // replay the persisted pin (SPEC §7)
    // Fallback server-truth chat state, in case the HTTP load in connect()
    // didn't render (the primary path). renderHistoryOnce dedupes the two.
    channel.send(JSON.stringify({ type: "client_ready" }));
  };
  channel.onmessage = (event) => {
    let msg;
    try { msg = JSON.parse(event.data); } catch { return; }
    if (msg.type === "transcript" && msg.text) {
      if (msg.role !== "user" || !reconcileTypedEcho(msg.text, msg.language)) {
        addMessage(msg.role, msg.text, msg.language);
      }
    }
    if (msg.type === "session") {
      // Fallback: the HTTP load in connect() is the primary path.
      renderHistoryOnce(msg.turns || [], msg.resumed);
    }
    // Authoritative presence from the pipeline (phase C): the companion latches
    // onto it and stops inferring. A dropped message is harmless — inference
    // stays in charge until the next one arrives.
    if (msg.type === "presence" && window.Companion) {
      Companion.setServerPresence(msg.state);
    }
  };
  channel.onclose = () => {
    telemetry.enqueue({ name: "data_channel_state", outcome: "disconnected" });
  };

  pc.ontrack = (event) => {
    botAudio.srcObject = event.streams[0];
    const playback = botAudio.play();
    if (playback) {
      playback.catch(() => telemetry.enqueue({
        name: "playback_failure", outcome: "error",
      }));
    }
  };
  const connection = pc;
  pc.onsignalingstatechange = () => {
    telemetry.enqueue({
      name: "signaling_state",
      outcome: signalingOutcome(connection.signalingState),
    });
  };
  pc.oniceconnectionstatechange = () => {
    telemetry.enqueue({
      name: "ice_state",
      outcome: connectionOutcome(connection.iceConnectionState),
    });
  };
  pc.onconnectionstatechange = () => {
    const state = connection.connectionState;
    telemetry.enqueue({ name: "peer_state", outcome: connectionOutcome(state) });
    if (state === "connected") {
      startStatsSampling(connection);
      void telemetry.flush();
    } else if (state === "disconnected") {
      stopStatsSampling(connection, false);
    } else if (["failed", "closed"].includes(state)) {
      stopStatsSampling(connection, true);
    }
    if (
      ["failed", "disconnected", "closed"].includes(state) &&
      !intentionalConnections.has(connection) && !disconnectedConnections.has(connection)
    ) {
      disconnectedConnections.add(connection);
      telemetry.enqueue({ name: "disconnect", outcome: "error" });
    }
    if (connection !== pc) return;
    if (state === "connected") setStatus("listening", "listening");
    if (["failed", "disconnected", "closed"].includes(state)) {
      setStatus("disconnected");
      $("connect").hidden = false;
      $("controls").hidden = true;
      refreshConnectLabel(); // a drop usually makes the next connect a resume
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
  offerPostedAt = performance.now();
  const response = await fetch(offerUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json", traceparent: telemetry.traceparent },
    body: JSON.stringify({
      sdp: pc.localDescription.sdp,
      type: pc.localDescription.type,
    }),
  });
  const answer = await response.json();
  peerConnectionId = answer.pc_id || null;
  // Render the resumed transcript now, synchronously, before the data channel
  // can open and deliver a live turn — no async fetch to race a reconnect.
  renderHistoryOnce(answer.turns || [], answer.resumed);
  await pc.setRemoteDescription(answer);

  $("connect").hidden = true;
  $("controls").hidden = false;
  applySpeaker(true);
}

function sendText() {
  const input = $("text");
  const text = input.value.trim();
  if (!text || !channel || channel.readyState !== "open") return;
  channel.send(JSON.stringify({ type: "user_text", text }));
  const element = addMessage("user", text, null);
  element.dataset.awaitingEcho = "true";
  const pending = { text, element, timer: null };
  pending.timer = setTimeout(() => {
    element.removeAttribute("data-awaiting-echo");
    telemetry.enqueue({ name: "transcript_echo_timeout", outcome: "timeout" });
  }, 30000);
  pendingTypedEchoes.push(pending);
  if (pendingTypedEchoes.length > 20) {
    clearTimeout(pendingTypedEchoes.shift().timer);
  }
  // The server echo later supplies the authoritative language without duplicating it.
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
workspaceButton.addEventListener("click", () => setWorkspaceVisible(workspace.hidden));
$("model-close").addEventListener("click", () => setWorkspaceVisible(false));
$("history-back").addEventListener("click", () => {
  sessionDetail.hidden = true;
  $("session-list-view").hidden = false;
});

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(payload.detail || `Request failed (${response.status})`);
    error.status = response.status;
    throw error;
  }
  return payload;
}

async function afterVoiceStops(path, options) {
  try { await disconnectVoice(); }
  catch (error) {
    if (error.status !== 503) throw error;
  }
  for (let attempt = 0; attempt < 40; attempt += 1) {
    try { return await api(path, options); }
    catch (error) {
      if (error.status !== 409 || attempt === 39) throw error;
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
  }
}

function textElement(tag, text, className = "") {
  const element = document.createElement(tag);
  element.textContent = text;
  if (className) element.className = className;
  return element;
}

function actionButton(label, action, className = "") {
  const button = textElement("button", label, className);
  button.type = "button";
  button.addEventListener("click", action);
  return button;
}

async function mutateClaim(kind, id, action, body = null) {
  const plural = kind === "node" ? "nodes" : "edges";
  const options = { method: action === "edit" ? "PATCH" : action === "delete" ? "DELETE" : "POST" };
  if (body) {
    options.headers = { "Content-Type": "application/json" };
    options.body = JSON.stringify(body);
  }
  const suffix = ["confirm", "reject"].includes(action) ? `/${action}` : "";
  await api(`/api/graph/${plural}/${id}${suffix}`, options);
  await loadWorkspace();
}

async function showClaimDetail(kind, id) {
  const plural = kind === "node" ? "nodes" : "edges";
  const payload = await api(`/api/graph/${plural}/${id}`);
  const claim = payload[kind];
  $("model-detail-title").textContent = claim.statement;
  const body = $("model-detail-body");
  body.replaceChildren();
  const facts = document.createElement("dl");
  for (const [label, value] of [
    ["Type", claim.type], ["Status", claim.status], ["Source", claim.source],
    ["Evidence", claim.n_auditable_occurrences], ["First seen", claim.first_seen],
    ["Last seen", claim.last_seen],
  ]) {
    facts.append(textElement("dt", label), textElement("dd", String(value ?? "—")));
  }
  body.append(facts, textElement("h3", "Evidence and provenance"));
  const evidence = document.createElement("ul");
  for (const item of payload.evidence || []) {
    const source = item.source_state === "deleted"
      ? "Deleted source (quote sanitized)"
      : `${item.source_type}${item.source_session_id ? ` · session ${item.source_session_id}` : ""}`;
    evidence.appendChild(textElement("li", `${source}${item.quote_text ? ` — “${item.quote_text}”` : ""}`));
  }
  if (!evidence.children.length) evidence.appendChild(textElement("li", "No retained evidence."));
  body.append(evidence, textElement("h3", "Lifecycle"));
  const lifecycle = document.createElement("ol");
  for (const event of payload.lifecycle || []) {
    lifecycle.appendChild(textElement("li", `${event.created_at}: ${event.event_type} → ${event.to_status}`));
  }
  body.appendChild(lifecycle);
  $("model-detail").showModal();
}

function renderClaim(claim, kind, nodeNames) {
  const item = document.createElement("article");
  item.className = "model-item";
  item.append(textElement("strong", claim.statement));
  const relationship = kind === "edge"
    ? ` · ${nodeNames.get(claim.src) || claim.src} → ${nodeNames.get(claim.dst) || claim.dst}`
    : "";
  item.append(textElement("p", `${claim.type} · ${claim.status} · ${claim.source}${relationship}`, "meta"));
  const actions = document.createElement("div");
  actions.className = "item-actions";
  actions.append(actionButton("Audit", () => showClaimDetail(kind, claim.id)));
  actions.append(actionButton("Edit", async () => {
    const statement = window.prompt("Correct statement", claim.statement);
    if (!statement || statement.trim() === claim.statement) return;
    await mutateClaim(kind, claim.id, "edit", { statement: statement.trim() });
  }));
  if (claim.status === "proposed") {
    actions.append(actionButton("Confirm", () => mutateClaim(kind, claim.id, "confirm"), "primary"));
    actions.append(actionButton("Reject", () => mutateClaim(kind, claim.id, "reject")));
  }
  actions.append(actionButton("Delete", async () => {
    if (window.confirm("Delete this claim and prevent unchanged relearning?")) {
      await mutateClaim(kind, claim.id, "delete");
    }
  }, "danger"));
  item.appendChild(actions);
  return item;
}

function renderGraphList(payload) {
  const nodes = $("model-nodes");
  const edges = $("model-edges");
  nodes.replaceChildren();
  edges.replaceChildren();
  const names = new Map(payload.nodes.map((node) => [node.id, node.statement]));
  for (const node of payload.nodes) nodes.appendChild(renderClaim(node, "node", names));
  for (const edge of payload.edges) edges.appendChild(renderClaim(edge, "edge", names));
  if (!payload.nodes.length) nodes.appendChild(textElement("p", "No matching claims."));
  if (!payload.edges.length) edges.appendChild(textElement("p", "No matching relationships."));
}

function drawGraph(payload) {
  const svg = $("model-graph");
  svg.replaceChildren();
  const namespace = "http://www.w3.org/2000/svg";
  const positions = new Map();
  const total = Math.max(payload.nodes.length, 1);
  payload.nodes.forEach((node, index) => {
    const angle = (2 * Math.PI * index) / total;
    positions.set(node.id, [360 + Math.cos(angle) * 260, 150 + Math.sin(angle) * 105]);
  });
  for (const edge of payload.edges) {
    const src = positions.get(edge.src);
    const dst = positions.get(edge.dst);
    if (!src || !dst) continue;
    const line = document.createElementNS(namespace, "line");
    line.setAttribute("x1", src[0]); line.setAttribute("y1", src[1]);
    line.setAttribute("x2", dst[0]); line.setAttribute("y2", dst[1]);
    line.setAttribute("class", `graph-edge status-${edge.status}`);
    const title = document.createElementNS(namespace, "title");
    title.textContent = edge.statement;
    line.appendChild(title); svg.appendChild(line);
  }
  for (const node of payload.nodes) {
    const [x, y] = positions.get(node.id);
    const group = document.createElementNS(namespace, "g");
    group.setAttribute("class", `graph-node status-${node.status}`);
    group.setAttribute("tabindex", "0");
    group.setAttribute("role", "button");
    const circle = document.createElementNS(namespace, "circle");
    circle.setAttribute("cx", x); circle.setAttribute("cy", y); circle.setAttribute("r", "22");
    const title = document.createElementNS(namespace, "title"); title.textContent = node.statement;
    circle.appendChild(title); group.appendChild(circle);
    const label = document.createElementNS(namespace, "text");
    label.setAttribute("x", x); label.setAttribute("y", y + 38);
    label.textContent = node.statement.length > 24 ? `${node.statement.slice(0, 22)}…` : node.statement;
    group.appendChild(label);
    group.addEventListener("click", () => showClaimDetail("node", node.id));
    group.addEventListener("keydown", (event) => { if (event.key === "Enter") showClaimDetail("node", node.id); });
    svg.appendChild(group);
  }
}

function renderInsights(insights) {
  const root = $("pending-insights");
  root.replaceChildren();
  const pending = insights.filter((item) => ["queued", "delivered", "snoozed"].includes(item.state));
  for (const insight of pending) {
    const item = document.createElement("article"); item.className = "model-item";
    item.append(textElement("strong", insight.statement_snapshot));
    item.append(textElement("p", `${insight.claim_kind} · ${insight.state}`, "meta"));
    const actions = document.createElement("div"); actions.className = "item-actions";
    for (const action of ["confirm", "reject", "snooze", "dismiss"]) {
      actions.append(actionButton(action[0].toUpperCase() + action.slice(1), async () => {
        const body = action === "snooze" ? JSON.stringify({ days: 7 }) : null;
        await api(`/api/insights/${insight.id}/${action}`, {
          method: "POST", headers: body ? { "Content-Type": "application/json" } : {}, body,
        });
        await loadWorkspace();
      }, action === "confirm" ? "primary" : ""));
    }
    item.appendChild(actions); root.appendChild(item);
  }
  if (!pending.length) root.appendChild(textElement("p", "No reflections waiting."));
}

function renderBoundaries(boundaries) {
  const root = $("boundary-list"); root.replaceChildren();
  for (const boundary of boundaries) {
    const row = document.createElement("div"); row.className = "boundary-row";
    row.append(textElement("span", `${boundary.kind}: ${boundary.value}`));
    row.append(actionButton("Remove", async () => {
      await api("/api/graph/boundaries", {
        method: "DELETE", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind: boundary.kind, value: boundary.value }),
      });
      await loadWorkspace();
    }, "danger"));
    root.appendChild(row);
  }
  if (!boundaries.length) root.appendChild(textElement("p", "No boundaries configured."));
}

async function loadResearchDocuments() {
  const payload = await api("/api/research");
  const root = $("research-documents"); root.replaceChildren();
  for (const source of payload.documents || []) {
    const item = document.createElement("article"); item.className = "model-item";
    item.append(textElement("strong", source.title || source.filename || source.ref));
    item.append(textElement("p", `${source.format || "text"} · ${source.status || "indexed"} · ${source.review_count || 0} review`, "meta"));
    const actions = document.createElement("div"); actions.className = "item-actions";
    actions.append(actionButton("Preview / correct", async () => {
      const detail = (await api(`/api/research/${source.id}`)).document;
      $("model-detail-title").textContent = detail.source_title;
      const body = $("model-detail-body"); body.replaceChildren();
      body.append(textElement("p", `${detail.source_ref} · ${detail.filename}`));
      for (const block of detail.blocks || []) {
        const article = document.createElement("article"); article.className = "model-item";
        article.append(textElement("strong", `${block.anchor}${block.needs_review ? " · review required" : ""}`));
        article.append(textElement("p", block.text));
        article.append(actionButton("Correct", async () => {
          const correction = window.prompt("Correct extracted text", block.text);
          if (!correction || correction.trim() === block.text) return;
          await api(`/api/research/${detail.id}/blocks/${encodeURIComponent(block.anchor)}`, {
            method: "PATCH", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text: correction.trim() }),
          });
          $("model-detail").close(); await loadResearchDocuments();
        }));
        body.appendChild(article);
      }
      $("model-detail").showModal();
    }));
    actions.append(actionButton("Reindex", async () => {
      await api(`/api/research/${source.id}/reindex`, { method: "POST" });
      await loadResearchDocuments();
    }));
    actions.append(actionButton("Delete", async () => {
      if (!window.confirm("Delete this local source and its index?")) return;
      await api(`/api/research/${source.id}`, { method: "DELETE" });
      await loadResearchDocuments();
    }, "danger"));
    item.appendChild(actions);
    root.appendChild(item);
  }
  if (!root.children.length) root.appendChild(textElement("p", "No local research imported."));
}

function settingField(label, input) {
  const wrapper = document.createElement("label");
  wrapper.append(textElement("span", label), input);
  return wrapper;
}

function renderProactivity(channels) {
  const root = $("proactivity-channels"); root.replaceChildren();
  for (const channel of channels) {
    const form = document.createElement("form"); form.className = "channel-settings";
    form.appendChild(textElement("h4", channel.channel.replace("_", " ")));
    const enabled = document.createElement("input"); enabled.type = "checkbox"; enabled.checked = channel.enabled;
    const timezone = document.createElement("input"); timezone.value = channel.timezone || Intl.DateTimeFormat().resolvedOptions().timeZone;
    const quietStart = document.createElement("input"); quietStart.type = "time"; quietStart.value = channel.quiet_start;
    const quietEnd = document.createElement("input"); quietEnd.type = "time"; quietEnd.value = channel.quiet_end;
    const schedule = document.createElement("input"); schedule.type = "time"; schedule.value = channel.schedule_time;
    const day = document.createElement("select");
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].forEach((name, index) => day.appendChild(new Option(name, String(index))));
    day.value = String(channel.schedule_day);
    const frequency = document.createElement("select");
    frequency.append(new Option("Daily", "daily"), new Option("Weekly", "weekly")); frequency.value = channel.frequency;
    const topic = document.createElement("input"); topic.maxLength = 500; topic.value = channel.topic || ""; topic.placeholder = "Optional check-in topic";
    form.append(
      settingField("Enabled", enabled), settingField("Timezone", timezone),
      settingField("Quiet from", quietStart), settingField("Quiet until", quietEnd),
      settingField("Delivery time", schedule), settingField("Day", day),
      settingField("Frequency", frequency), settingField("Topic", topic),
    );
    const save = textElement("button", "Save channel"); save.type = "submit"; form.appendChild(save);
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      await api(`/api/proactivity/${channel.channel}`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          enabled: enabled.checked, timezone: timezone.value,
          quiet_start: quietStart.value, quiet_end: quietEnd.value,
          schedule_time: schedule.value, schedule_day: Number(day.value),
          frequency: frequency.value, topic: topic.value || null,
        }),
      });
      await loadProactivity();
    });
    root.appendChild(form);
  }
}

async function loadProactivity() {
  const [settings, outreach, digests] = await Promise.all([
    api("/api/proactivity"),
    api("/api/proactivity/in-app?consume=true"),
    api("/api/proactivity/digests"),
  ]);
  renderProactivity(settings.channels || []);
  const messages = $("in-app-outreach"); messages.replaceChildren();
  for (const message of outreach.messages || []) {
    messages.appendChild(textElement("p", message.message, "outreach-message"));
  }
  const digestRoot = $("digest-list"); digestRoot.replaceChildren();
  for (const digest of digests.digests || []) {
    const article = document.createElement("article"); article.className = "model-item";
    article.append(textElement("strong", `${digest.period_start} – ${digest.period_end}`));
    const content = textElement("pre", digest.content); article.appendChild(content);
    digestRoot.appendChild(article);
  }
}

function urlBase64ToUint8Array(value) {
  const padding = "=".repeat((4 - value.length % 4) % 4);
  const raw = atob((value + padding).replace(/-/g, "+").replace(/_/g, "/"));
  return Uint8Array.from([...raw].map((character) => character.charCodeAt(0)));
}

$("push-subscribe").addEventListener("click", async () => {
  if (!("serviceWorker" in navigator) || !("PushManager" in window)) {
    throw new Error("Encrypted push is unavailable in this browser");
  }
  const permission = await Notification.requestPermission();
  if (permission !== "granted") return;
  const registration = await navigator.serviceWorker.ready;
  const { public_key: publicKey } = await api("/api/push/public-key");
  const subscription = await registration.pushManager.subscribe({
    userVisibleOnly: true, applicationServerKey: urlBase64ToUint8Array(publicKey),
  });
  const serialized = subscription.toJSON();
  await api("/api/push/subscriptions", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      endpoint: serialized.endpoint, p256dh: serialized.keys.p256dh,
      auth: serialized.keys.auth,
    }),
  });
  $("push-subscribe").textContent = "Encrypted push enabled for this browser";
});

async function loadWorkspace() {
  const params = new URLSearchParams();
  const type = $("graph-type").value;
  if (NODE_TYPES.includes(type)) params.set("node_type", type);
  if (EDGE_TYPES.includes(type)) params.set("edge_type", type);
  if ($("graph-status").value) params.set("status", $("graph-status").value);
  if ($("graph-source").value) params.set("source", $("graph-source").value);
  graphPayload = await api(`/api/graph?${params}`);
  renderGraphList(graphPayload); drawGraph(graphPayload);
  renderInsights(graphPayload.pending_insights || []);
  renderBoundaries(graphPayload.boundaries || []);
  await Promise.all([loadResearchDocuments(), loadProactivity()]);
}

for (const type of [...NODE_TYPES, ...EDGE_TYPES]) {
  $("graph-type").appendChild(new Option(type, type));
}
$("graph-filters").addEventListener("submit", (event) => {
  event.preventDefault(); loadWorkspace().catch((err) => setStatus(`error: ${err.message}`));
});
$("graph-view-toggle").addEventListener("click", () => {
  const show = $("model-graph").hasAttribute("hidden");
  $("model-graph").toggleAttribute("hidden", !show);
  $("model-list").toggleAttribute("hidden", show);
  $("graph-view-toggle").textContent = show ? "Show list" : "Show graph";
  $("graph-view-toggle").setAttribute("aria-pressed", String(show));
});
$("boundary-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  await api("/api/graph/boundaries", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ kind: $("boundary-kind").value, value: $("boundary-value").value }),
  });
  $("boundary-value").value = ""; await loadWorkspace();
});
$("research-upload").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = $("research-file").files[0]; if (!file) return;
  const body = new FormData(); body.append("file", file);
  await api("/api/research/ingest", { method: "POST", body });
  $("research-file").value = ""; await loadResearchDocuments();
});
$("data-export").addEventListener("click", () => { location.href = "/api/data/export"; });
$("data-restore").addEventListener("click", async () => {
  const file = $("data-restore-file").files[0];
  if (!file) return;
  if (window.prompt(
    "Restore replaces all current local data. Type RESTORE to continue.",
  ) !== "RESTORE") return;
  const body = new FormData(); body.append("file", file);
  try {
    await afterVoiceStops("/api/data/restore", { method: "POST", body });
  } catch (error) {
    setStatus(`error: ${error.message}`);
    return;
  }
  $("data-restore-file").value = "";
  await loadWorkspace();
});
$("data-delete").addEventListener("click", async () => {
  if (window.prompt("Type DELETE EVERYTHING to erase all local TheraPy data.") !== "DELETE EVERYTHING") return;
  try {
    await afterVoiceStops("/api/data", { method: "DELETE", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ confirmation: "DELETE EVERYTHING" }) });
  } catch (error) {
    setStatus(`error: ${error.message}`);
    return;
  }
  await loadWorkspace();
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
$("session-rename").addEventListener("click", async () => {
  if (!viewingSessionId) return;
  const current = $("session-title").textContent;
  const title = window.prompt("Session title", current);
  if (!title || !title.trim() || title.trim() === current) return;
  const response = await fetch(
    `/api/sessions/${encodeURIComponent(viewingSessionId)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: title.trim() }),
    },
  );
  if (!response.ok) {
    setStatus("error: could not rename");
    return;
  }
  $("session-title").textContent = (await response.json()).title;
});

$("session-delete").addEventListener("click", async () => {
  if (!viewingSessionId) return;
  const choice = window.prompt(
    "Delete transcript/audio. Type KEEP to preserve sanitized learned knowledge, " +
    "or REMOVE to also erase knowledge derived only from this conversation.",
    "KEEP",
  );
  if (!choice || !["KEEP", "REMOVE"].includes(choice.trim().toUpperCase())) return;
  const mode = choice.trim().toUpperCase() === "REMOVE"
    ? "remove_derived" : "keep_knowledge";
  const response = await fetch(
    `/api/sessions/${encodeURIComponent(viewingSessionId)}?mode=${mode}`,
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

// Push-to-talk wiring. companion.js renders the Hold button + mode toggle and
// fires these callbacks; the mic track lives here, so the gating does too.
function micToggleOn() {
  return $("mic").getAttribute("aria-pressed") !== "false";
}
function applyMicMode() {
  if (!micTrack) return;
  // Open mode leaves the mic under the 🎙️ toggle; push mode mutes it until Hold.
  micTrack.enabled = pushToTalk ? false : micToggleOn();
}
if (window.Companion) {
  Companion.onMicMode = (mode) => { pushToTalk = mode === "push"; applyMicMode(); };
  Companion.onHoldStart = () => { if (micTrack && pushToTalk) micTrack.enabled = true; };
  Companion.onHoldEnd = () => { if (micTrack && pushToTalk) micTrack.enabled = false; };
}

if ("serviceWorker" in navigator) {
  // Registration failure means no SW lifecycle events will ever arrive, so
  // the page itself must record the outcome (O4 audit F-07).
  navigator.serviceWorker.register("/sw.js").then(
    () => telemetry.enqueue({ name: "sw_lifecycle", outcome: "success" }),
    () => telemetry.enqueue({ name: "sw_lifecycle", outcome: "error" }),
  );
}

refreshConnectLabel();
if (location.hash === "#model") setWorkspaceVisible(true);
