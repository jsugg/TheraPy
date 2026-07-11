/*
New companion ids: #avatar, #presence, #presence-label, #presence-dot,
#avatar-name, #avatar-tagline, #avatar-pick, #avatar-picker, #talk, #mic-mode.
*/
(() => {
  "use strict";

  const DEFAULT_MANIFEST = Object.freeze({
    id: "rowan",
    name: "Rowan",
    tagline: "Your wise companion",
    light: { accent: "#1b4332", accentSoft: "#d8e8dd" },
    dark: { accent: "#74c69d", accentSoft: "#24352b" },
  });

  const PRESENCE_LABELS = Object.freeze({
    offline: "Offline",
    connecting: "Connecting…",
    listening: "Ready to listen",
    thinking: "Thinking…",
    speaking: "Speaking",
    "mic-off": "Mic off",
  });

  const SAFE_ID = /^[a-z0-9_-]{1,64}$/i;
  const root = document.documentElement;
  const callbacks = {
    onHoldStart: null,
    onHoldEnd: null,
    onMicMode: null,
  };

  const state = {
    initPromise: null,
    index: { default: DEFAULT_MANIFEST.id, avatars: [DEFAULT_MANIFEST.id] },
    manifest: DEFAULT_MANIFEST,
    avatarId: DEFAULT_MANIFEST.id,
    avatarSm: avatarUrl(DEFAULT_MANIFEST.id, "portrait-sm.webp"),
    presence: "offline",
    micMode: "open",
    holding: false,
    schemeQuery: null,
    playbackAudio: null,
    activePlayButton: null,
  };

  const api = {
    init,
    setPresence,
    avatarSmUrl: () => state.avatarSm,
    decorateAssistant,
    addPlayButton,
  };

  defineCallback("onHoldStart");
  defineCallback("onHoldEnd");
  defineCallback("onMicMode");
  window.Companion = api;

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => { init(); }, { once: true });
  } else {
    queueMicrotask(() => { init(); });
  }

  function defineCallback(name) {
    Object.defineProperty(api, name, {
      configurable: true,
      enumerable: true,
      get() {
        return callbacks[name];
      },
      set(fn) {
        callbacks[name] = typeof fn === "function" ? fn : null;
        if (name === "onMicMode" && callbacks[name]) {
          queueMicrotask(() => safeCall(callbacks[name], state.micMode));
        }
      },
    });
  }

  function init() {
    if (state.initPromise) return state.initPromise;

    setPresence("offline");
    setupSchemeListener();
    setupPickerShell();
    setupMicMode();
    setupStatusObserver();
    setupMessageObservers();
    setupAudioPresence();

    state.initPromise = (async () => {
      const index = await loadAvatarIndex();
      state.index = index;
      const stored = safeAvatarId(storageGet("avatar"));
      const chosen = stored && index.avatars.includes(stored) ? stored : index.default;
      await selectAvatar(chosen, { persist: Boolean(stored) });
      await buildAvatarPicker();
      decorateExistingMessages();
      setPresence("offline");
    })().catch(() => {
      applyManifest(DEFAULT_MANIFEST);
      buildAvatarPicker();
      decorateExistingMessages();
      setPresence("offline");
    });

    return state.initPromise;
  }

  async function loadAvatarIndex() {
    try {
      const raw = await fetchJson("/avatars/index.json");
      const defaultId = safeAvatarId(raw.default) || DEFAULT_MANIFEST.id;
      const avatars = Array.isArray(raw.avatars)
        ? raw.avatars.map(safeAvatarId).filter(Boolean)
        : [];
      const unique = [...new Set([defaultId, ...avatars])];
      return { default: defaultId, avatars: unique.length ? unique : [DEFAULT_MANIFEST.id] };
    } catch {
      return { default: DEFAULT_MANIFEST.id, avatars: [DEFAULT_MANIFEST.id] };
    }
  }

  async function selectAvatar(id, options = {}) {
    const wanted = safeAvatarId(id) || state.index.default || DEFAULT_MANIFEST.id;
    let manifest = await loadManifest(wanted);
    if (!manifest && wanted !== state.index.default) {
      manifest = await loadManifest(state.index.default);
    }
    applyManifest(manifest || DEFAULT_MANIFEST);
    if (options.persist) storageSet("avatar", state.avatarId);
    updatePickerState();
  }

  async function loadManifest(id) {
    const safeId = safeAvatarId(id);
    if (!safeId) return null;
    try {
      return normalizeManifest(await fetchJson(`/avatars/${safeId}/manifest.json`), safeId);
    } catch {
      if (safeId === DEFAULT_MANIFEST.id) return DEFAULT_MANIFEST;
      return null;
    }
  }

  function applyManifest(manifest) {
    state.manifest = normalizeManifest(manifest, DEFAULT_MANIFEST.id);
    state.avatarId = state.manifest.id;
    state.avatarSm = avatarUrl(state.avatarId, "portrait-sm.webp");
    applyPalette();
    updatePortrait();
    updateCompanionText();
    refreshDecorations();
  }

  function normalizeManifest(raw, fallbackId) {
    const obj = isObject(raw) ? raw : {};
    const id = safeAvatarId(obj.id) || safeAvatarId(fallbackId) || DEFAULT_MANIFEST.id;
    return {
      id,
      name: nonEmptyString(obj.name) || titleFromId(id),
      tagline: nonEmptyString(obj.tagline) || DEFAULT_MANIFEST.tagline,
      light: normalizePalette(obj.light, DEFAULT_MANIFEST.light),
      dark: normalizePalette(obj.dark, DEFAULT_MANIFEST.dark),
    };
  }

  function normalizePalette(raw, fallback) {
    const obj = isObject(raw) ? raw : {};
    return {
      accent: cssColor(obj.accent) || fallback.accent,
      accentSoft: cssColor(obj.accentSoft) || fallback.accentSoft,
    };
  }

  function applyPalette() {
    const palette = state.manifest[isDarkScheme() ? "dark" : "light"] || state.manifest.light;
    root.style.setProperty("--accent", palette.accent);
    root.style.setProperty("--accent-soft", palette.accentSoft);
    const theme = document.querySelector('meta[name="theme-color"]');
    if (theme) theme.setAttribute("content", palette.accent);
  }

  function updatePortrait() {
    const img = byId("avatar");
    if (!img) return;
    const large = avatarUrl(state.avatarId, "portrait.webp");
    const small = state.avatarSm;
    img.hidden = false;
    img.alt = `${state.manifest.name} companion portrait`;
    img.sizes = "(max-width: 520px) 136px, 150px";
    img.srcset = `${small} 96w, ${large} 512w`;
    img.onerror = () => {
      if (!img.dataset.smallFallback) {
        img.dataset.smallFallback = "true";
        img.removeAttribute("srcset");
        img.src = small;
        return;
      }
      img.hidden = true;
    };
    delete img.dataset.smallFallback;
    img.src = large;
  }

  function updateCompanionText() {
    setText("avatar-name", state.manifest.name);
    setText("avatar-tagline", state.manifest.tagline);
    const pickerButton = byId("avatar-pick");
    if (pickerButton) {
      pickerButton.setAttribute("aria-label", `Choose companion, current ${state.manifest.name}`);
      pickerButton.title = `Choose companion, current ${state.manifest.name}`;
    }
  }

  function setPresence(next) {
    const presence = Object.hasOwn(PRESENCE_LABELS, next) ? next : "offline";
    state.presence = presence;

    const label = byId("presence-label");
    if (label) label.textContent = PRESENCE_LABELS[presence];

    const pill = byId("presence");
    if (pill) pill.dataset.presence = presence;

    const frame = byId("avatar-frame") || byId("avatar")?.parentElement;
    if (frame) {
      frame.dataset.presence = presence;
      frame.classList.toggle("is-pulsing", presence === "listening" || presence === "speaking");
    }
  }

  function setupSchemeListener() {
    if (!window.matchMedia) return;
    state.schemeQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => applyPalette();
    if (typeof state.schemeQuery.addEventListener === "function") {
      state.schemeQuery.addEventListener("change", onChange);
    } else if (typeof state.schemeQuery.addListener === "function") {
      state.schemeQuery.addListener(onChange);
    }
  }

  function setupPickerShell() {
    const button = byId("avatar-pick");
    const picker = byId("avatar-picker");
    if (!button || !picker) return;

    button.addEventListener("click", () => {
      setPickerOpen(picker.hidden);
    });

    document.addEventListener("click", (event) => {
      if (picker.hidden || !(event.target instanceof Node)) return;
      if (picker.contains(event.target) || button.contains(event.target)) return;
      setPickerOpen(false);
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") setPickerOpen(false);
    });
  }

  async function buildAvatarPicker() {
    const picker = byId("avatar-picker");
    if (!picker) return;
    picker.replaceChildren();

    for (const id of state.index.avatars) {
      const manifest = await loadManifest(id) || normalizeManifest({ id }, id);
      const option = document.createElement("button");
      option.type = "button";
      option.className = "avatar-option";
      option.dataset.avatarId = manifest.id;
      option.setAttribute("role", "menuitemradio");
      option.setAttribute("aria-checked", String(manifest.id === state.avatarId));

      const img = document.createElement("img");
      img.src = avatarUrl(manifest.id, "portrait-sm.webp");
      img.alt = "";
      img.loading = "lazy";
      img.decoding = "async";

      const text = document.createElement("span");
      const name = document.createElement("strong");
      name.textContent = manifest.name;
      const tagline = document.createElement("small");
      tagline.textContent = manifest.tagline;
      text.append(name, tagline);

      const check = document.createElement("span");
      check.className = "avatar-check";
      check.setAttribute("aria-hidden", "true");
      check.textContent = manifest.id === state.avatarId ? "✓" : "";

      option.append(img, text, check);
      option.addEventListener("click", async () => {
        await selectAvatar(manifest.id, { persist: true });
        setPickerOpen(false);
      });
      picker.appendChild(option);
    }
  }

  function setPickerOpen(open) {
    const picker = byId("avatar-picker");
    const button = byId("avatar-pick");
    if (!picker || !button) return;
    picker.hidden = !open;
    button.setAttribute("aria-expanded", String(open));
  }

  function updatePickerState() {
    document.querySelectorAll(".avatar-option").forEach((option) => {
      const active = option.dataset.avatarId === state.avatarId;
      option.setAttribute("aria-checked", String(active));
      const check = option.querySelector(".avatar-check");
      if (check) check.textContent = active ? "✓" : "";
    });
  }

  function setupMicMode() {
    const stored = storageGet("micMode");
    state.micMode = stored === "push" ? "push" : "open";
    renderMicMode();

    const mode = byId("mic-mode");
    if (mode) {
      mode.addEventListener("click", () => {
        setMicMode(state.micMode === "push" ? "open" : "push", true);
      });
    }

    const talk = byId("talk");
    if (!talk) return;

    talk.addEventListener("pointerdown", (event) => {
      if (typeof talk.setPointerCapture === "function") {
        talk.setPointerCapture(event.pointerId);
      }
      beginHold(event);
    });
    talk.addEventListener("pointerup", endHold);
    talk.addEventListener("pointercancel", endHold);
    talk.addEventListener("pointerleave", endHold);
    talk.addEventListener("touchstart", beginHold, { passive: false });
    talk.addEventListener("touchend", endHold);
    talk.addEventListener("touchcancel", endHold);
    talk.addEventListener("keydown", (event) => {
      if ((event.key === " " || event.key === "Enter") && !event.repeat) beginHold(event);
    });
    talk.addEventListener("keyup", (event) => {
      if (event.key === " " || event.key === "Enter") endHold(event);
    });
    window.addEventListener("blur", endHold);
  }

  function setMicMode(mode, notify) {
    const next = mode === "push" ? "push" : "open";
    if (state.micMode === next) return;
    if (state.holding) endHold();
    state.micMode = next;
    storageSet("micMode", next);
    renderMicMode();
    if (notify) safeCall(callbacks.onMicMode, next);
  }

  function renderMicMode() {
    const push = state.micMode === "push";
    const talk = byId("talk");
    if (talk) {
      talk.hidden = !push;
      talk.setAttribute("aria-pressed", String(state.holding));
      talk.classList.toggle("is-holding", state.holding);
    }

    const mode = byId("mic-mode");
    if (mode) {
      mode.textContent = push ? "Push mode" : "Open mic";
      mode.setAttribute("aria-pressed", String(push));
    }
  }

  function beginHold(event) {
    if (state.micMode !== "push" || state.holding) return;
    if (event?.cancelable) event.preventDefault();
    state.holding = true;
    renderMicMode();
    safeCall(callbacks.onHoldStart);
  }

  function endHold(event) {
    if (!state.holding) return;
    if (event?.cancelable) event.preventDefault();
    state.holding = false;
    renderMicMode();
    safeCall(callbacks.onHoldEnd);
  }

  function setupStatusObserver() {
    const status = byId("status");
    if (!status || typeof MutationObserver !== "function") return;
    const observer = new MutationObserver(() => {
      setPresence(presenceFromStatus(status));
    });
    observer.observe(status, {
      attributes: true,
      attributeFilter: ["data-state"],
      childList: true,
      characterData: true,
      subtree: true,
    });
    setPresence(presenceFromStatus(status));
  }

  function presenceFromStatus(status) {
    const text = (status.textContent || "").trim().toLowerCase();
    const dataState = (status.dataset.state || "").trim().toLowerCase();
    const mic = byId("mic");
    const micOn = mic?.getAttribute("aria-pressed") !== "false";

    if (text.startsWith("error:") || text === "disconnected") return "offline";
    if (text === "connecting…" || text === "connecting..." || dataState === "connecting") {
      return "connecting";
    }
    if (text === "mic off" || !micOn && controlsVisible()) return "mic-off";
    if (dataState === "listening" || text === "listening" || text === "resumed") {
      return liveAudioPlaying() ? "speaking" : "listening";
    }
    return "offline";
  }

  function setupMessageObservers() {
    observeMessages(byId("chat"), true);
    observeMessages(byId("session-turns"), false);
  }

  function observeMessages(container, live) {
    if (!container || typeof MutationObserver !== "function") return;
    const observer = new MutationObserver((mutations) => {
      for (const mutation of mutations) {
        for (const node of mutation.addedNodes) {
          processAddedMessageNode(node, live);
        }
      }
    });
    observer.observe(container, { childList: true, subtree: true });
  }

  function processAddedMessageNode(node, live) {
    if (!(node instanceof Element)) return;
    const messages = node.matches(".msg") ? [node] : [...node.querySelectorAll(".msg")];
    for (const msg of messages) {
      if (msg.classList.contains("assistant")) {
        decorateAssistant(msg);
        const audioUrl = msg.dataset.audioUrl || msg.getAttribute("data-audio-url");
        if (audioUrl) addPlayButton(msg, audioUrl);
        if (live) setPresence(liveAudioPlaying() ? "speaking" : baseConnectedPresence());
      } else if (live && msg.classList.contains("user")) {
        setPresence("thinking");
      }
    }
  }

  function decorateExistingMessages() {
    document.querySelectorAll(".msg.assistant").forEach((msg) => decorateAssistant(msg));
  }

  function decorateAssistant(msgEl) {
    if (!(msgEl instanceof HTMLElement) || !msgEl.classList.contains("assistant")) return;
    if (msgEl.dataset.companionDecorated === "true") return;

    const chip = document.createElement("span");
    chip.className = "assistant-chip";
    chip.setAttribute("aria-hidden", "true");

    const img = document.createElement("img");
    img.src = state.avatarSm;
    img.alt = "";
    img.loading = "lazy";
    img.decoding = "async";
    chip.appendChild(img);

    const name = document.createElement("span");
    name.className = "assistant-name";
    name.textContent = state.manifest.name;

    msgEl.prepend(chip, name);
    msgEl.dataset.companionDecorated = "true";
  }

  function refreshDecorations() {
    document.querySelectorAll(".assistant-chip img").forEach((img) => {
      img.src = state.avatarSm;
    });
    document.querySelectorAll(".assistant-name").forEach((name) => {
      name.textContent = state.manifest.name;
    });
  }

  function addPlayButton(msgEl, audioUrl) {
    if (!(msgEl instanceof HTMLElement)) return;
    const url = safeAudioUrl(audioUrl);
    if (!url) return;

    let wrap = msgEl.querySelector(".message-playback");
    if (wrap) {
      const button = wrap.querySelector("button");
      if (button) button.dataset.audioUrl = url;
      return;
    }

    wrap = document.createElement("span");
    wrap.className = "message-playback";

    const button = document.createElement("button");
    button.type = "button";
    button.className = "play-button";
    button.dataset.audioUrl = url;
    button.setAttribute("aria-label", "Play message audio");
    button.textContent = "▶";
    button.addEventListener("click", () => playAudio(url, button));

    wrap.appendChild(button);
    msgEl.appendChild(wrap);
  }

  function playAudio(url, button) {
    const audio = sharedPlaybackAudio();
    if (state.activePlayButton && state.activePlayButton !== button) {
      setPlayButton(state.activePlayButton, false);
    }

    if (!audio.paused && audio.src === url) {
      audio.pause();
      setPlayButton(button, false);
      return;
    }

    audio.pause();
    audio.src = url;
    state.activePlayButton = button;
    setPlayButton(button, true);
    audio.play().catch(() => {
      setPlayButton(button, false);
    });
  }

  function sharedPlaybackAudio() {
    if (state.playbackAudio) return state.playbackAudio;
    const audio = document.createElement("audio");
    audio.preload = "metadata";
    audio.hidden = true;
    audio.addEventListener("ended", () => setPlayButton(state.activePlayButton, false));
    audio.addEventListener("pause", () => setPlayButton(state.activePlayButton, false));
    document.body.appendChild(audio);
    state.playbackAudio = audio;
    return audio;
  }

  function setPlayButton(button, playing) {
    if (!button) return;
    button.textContent = playing ? "⏸" : "▶";
    button.setAttribute("aria-label", playing ? "Pause message audio" : "Play message audio");
    if (!playing && state.activePlayButton === button) state.activePlayButton = null;
  }

  function setupAudioPresence() {
    const audio = byId("bot-audio");
    if (!audio) return;
    audio.addEventListener("playing", () => setPresence("speaking"));
    audio.addEventListener("play", () => setPresence("speaking"));
    audio.addEventListener("pause", () => setPresence(baseConnectedPresence()));
    audio.addEventListener("ended", () => setPresence(baseConnectedPresence()));
    audio.addEventListener("emptied", () => setPresence(baseConnectedPresence()));
  }

  function baseConnectedPresence() {
    const status = byId("status");
    if (!status) return "offline";
    const dataState = (status.dataset.state || "").trim().toLowerCase();
    const text = (status.textContent || "").trim().toLowerCase();
    const mic = byId("mic");
    if (controlsVisible() && mic?.getAttribute("aria-pressed") === "false") return "mic-off";
    if (dataState === "listening" || text === "listening" || text === "resumed") return "listening";
    return presenceFromStatus(status);
  }

  function liveAudioPlaying() {
    const audio = byId("bot-audio");
    return Boolean(audio && !audio.paused && !audio.ended && audio.readyState > 2);
  }

  function controlsVisible() {
    const controls = byId("controls");
    return Boolean(controls && !controls.hidden);
  }

  async function fetchJson(url) {
    const response = await fetch(url, { cache: "no-cache" });
    if (!response.ok) throw new Error(`Could not load ${url}`);
    return response.json();
  }

  function safeAudioUrl(value) {
    if (typeof value !== "string" || !value.trim()) return "";
    try {
      const url = new URL(value, window.location.href);
      if (url.protocol === "blob:" || url.origin === window.location.origin) return url.href;
    } catch {
      return "";
    }
    return "";
  }

  function avatarUrl(id, file) {
    return `/avatars/${encodeURIComponent(id)}/${file}`;
  }

  function byId(id) {
    return document.getElementById(id);
  }

  function setText(id, value) {
    const el = byId(id);
    if (el) el.textContent = value;
  }

  function isDarkScheme() {
    return Boolean(state.schemeQuery?.matches);
  }

  function isObject(value) {
    return typeof value === "object" && value !== null;
  }

  function nonEmptyString(value) {
    return typeof value === "string" && value.trim() ? value.trim() : "";
  }

  function cssColor(value) {
    const text = nonEmptyString(value);
    if (/^#[0-9a-f]{3,8}$/i.test(text)) return text;
    if (/^(rgb|hsl)a?\(/i.test(text)) return text;
    return "";
  }

  function safeAvatarId(value) {
    const text = nonEmptyString(value);
    return SAFE_ID.test(text) ? text : "";
  }

  function titleFromId(id) {
    return id
      .split(/[-_]+/)
      .filter(Boolean)
      .map((part) => part.charAt(0).toUpperCase() + part.slice(1).toLowerCase())
      .join(" ") || DEFAULT_MANIFEST.name;
  }

  function storageGet(key) {
    try {
      return window.localStorage.getItem(key);
    } catch {
      return null;
    }
  }

  function storageSet(key, value) {
    try {
      window.localStorage.setItem(key, value);
    } catch {
      /* Storage can be unavailable in privacy modes; the UI still works. */
    }
  }

  function safeCall(fn, ...args) {
    if (typeof fn !== "function") return;
    try {
      fn(...args);
    } catch (error) {
      console.warn("Companion callback failed", error);
    }
  }
})();
