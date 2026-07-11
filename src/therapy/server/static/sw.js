/* Minimal service worker: app-shell cache for installability. The live
 * conversation (WebRTC, /api/*) is never cached. Shell fetches carry a
 * timeout: a hung server (wedged Docker VM) must degrade to the cached
 * shell in seconds, not spin forever. */
const CACHE = "therapy-shell-v10";
const SHELL = [
  "/", "/styles.css", "/app.js", "/companion.js", "/manifest.webmanifest",
  "/icon.svg", "/icon-192.png", "/icon-512.png",
  // Default companion pack, so the presence layer renders offline on first load;
  // other avatars runtime-cache on demand via the fetch handler below.
  "/avatars/index.json", "/avatars/rowan/manifest.json",
  "/avatars/rowan/portrait.webp", "/avatars/rowan/portrait-sm.webp",
];
const FETCH_TIMEOUT_MS = 8000;

self.addEventListener("install", (e) => {
  // Activate as soon as installed instead of waiting for every old tab to
  // close. A device that has loaded many versions of this app otherwise keeps
  // a stale worker in charge while the new one sits "waiting" — a controlling
  // worker that never updates is a known way to lose Chrome's install option.
  self.skipWaiting();
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    Promise.all([
      caches.keys().then((keys) =>
        Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
      ),
      // Control already-open pages immediately (no reload needed), so the
      // page is service-worker-controlled on first visit.
      self.clients.claim(),
    ])
  );
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.pathname.startsWith("/api/")) return;
  e.respondWith(
    fetch(e.request, { signal: AbortSignal.timeout(FETCH_TIMEOUT_MS) })
      .then((res) => {
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put(e.request, copy));
        return res;
      })
      .catch(async () => {
        const cached = await caches.match(e.request);
        if (cached) return cached;
        return new Response("TheraPy server is not reachable right now.", {
          status: 503,
          headers: { "Content-Type": "text/plain" },
        });
      })
  );
});
