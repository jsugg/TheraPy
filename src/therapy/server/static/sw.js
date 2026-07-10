/* Minimal service worker: app-shell cache for installability. The live
 * conversation (WebRTC, /api/*) is never cached. Shell fetches carry a
 * timeout: a hung server (wedged Docker VM) must degrade to the cached
 * shell in seconds, not spin forever. */
const CACHE = "therapy-shell-v3";
const SHELL = ["/", "/styles.css", "/app.js", "/manifest.webmanifest", "/icon.svg"];
const FETCH_TIMEOUT_MS = 8000;

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)));
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    )
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
