/* Minimal service worker: app-shell cache for installability. The live
 * conversation (WebRTC, /api/*) is never cached. Shell fetches carry a
 * timeout: a hung server (wedged Docker VM) must degrade to the cached
 * shell in seconds, not spin forever. */
const CACHE = "therapy-shell-v18";
const SHELL = [
  "/", "/styles.css", "/app.js", "/companion.js", "/manifest.webmanifest",
  "/icon.svg", "/icon-192.png", "/icon-512.png",
  // Default companion pack, so the presence layer renders offline on first load;
  // other avatars runtime-cache on demand via the fetch handler below.
  "/avatars/index.json", "/avatars/rowan/manifest.json",
  "/avatars/rowan/portrait.webp", "/avatars/rowan/portrait-sm.webp",
];
const FETCH_TIMEOUT_MS = 8000;
const SHELL_FETCH_REPORT_MS = 30000;
const SHELL_FETCH_REPORT_CAP = 10000;
let shellFetchCounts = { success: 0, error: 0 };
let shellFetchTimer = null;
let shellFallbackActive = false;

function postTelemetry(event) {
  // Events stay content-free; the controlled page owns validation and delivery.
  // A push normally wakes this worker with NO open window — in that case fall
  // back to one direct bounded POST (single event, no SW-side queue, no
  // retries) so the diagnostic is not silently lost.
  return self.clients.matchAll({ type: "window", includeUncontrolled: true })
    .then((clients) => {
      if (clients.length === 0) {
        return fetch("/api/telemetry/client", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ schema_version: 1, events: [event] }),
        }).then(() => undefined, () => undefined);
      }
      for (const client of clients) client.postMessage({ type: "telemetry", event });
    });
}

async function reportWaitUntil(promise) {
  // Completion of each waitUntil'd handler is observable through its own
  // terminal lifecycle event (installed/activated/shown/...); this hook makes
  // the REJECTION leg observable with the bounded "failed" outcome.
  try {
    return await promise;
  } catch (error) {
    await postTelemetry({ name: "sw_lifecycle", outcome: "failed" }).catch(() => {});
    throw error;
  }
}

function waitUntilWithTelemetry(event, promise) {
  event.waitUntil(reportWaitUntil(promise));
}

function reportShellFetches() {
  shellFetchTimer = null;
  const counts = shellFetchCounts;
  shellFetchCounts = { success: 0, error: 0 };
  for (const outcome of ["success", "error"]) {
    if (counts[outcome] > 0) {
      void postTelemetry({
        name: "shell_fetch", outcome, dropped_events: counts[outcome],
      }).catch(() => {});
    }
  }
}

function countShellFetch(outcome) {
  shellFetchCounts[outcome] = Math.min(
    SHELL_FETCH_REPORT_CAP, shellFetchCounts[outcome] + 1,
  );
  if (shellFetchCounts.success + shellFetchCounts.error >= SHELL_FETCH_REPORT_CAP) {
    if (shellFetchTimer !== null) clearTimeout(shellFetchTimer);
    reportShellFetches();
  } else if (shellFetchTimer === null) {
    shellFetchTimer = setTimeout(reportShellFetches, SHELL_FETCH_REPORT_MS);
  }
}

self.addEventListener("install", (e) => {
  // Activate as soon as installed instead of waiting for every old tab to
  // close. A device that has loaded many versions of this app otherwise keeps
  // a stale worker in charge while the new one sits "waiting" — a controlling
  // worker that never updates is a known way to lose Chrome's install option.
  waitUntilWithTelemetry(e, (async () => {
    await caches.open(CACHE).then((c) => c.addAll(SHELL));
    await self.skipWaiting();
    await postTelemetry({ name: "sw_lifecycle", outcome: "installed" })
      .catch(() => {});
  })());
});

self.addEventListener("activate", (e) => {
  waitUntilWithTelemetry(e, (async () => {
    await Promise.all([
      caches.keys().then((keys) =>
        Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
      ),
      // Control already-open pages immediately (no reload needed), so the
      // page is service-worker-controlled on first visit.
      self.clients.claim(),
    ]);
    await postTelemetry({ name: "sw_lifecycle", outcome: "activated" })
      .catch(() => {});
  })());
});

self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.pathname.startsWith("/api/")) return;
  e.respondWith(
    fetch(e.request, { signal: AbortSignal.timeout(FETCH_TIMEOUT_MS) })
      .then(async (res) => {
        if (!res.ok) {
          // A resolved 4xx/5xx is NOT shell success: never cache it, never
          // clear fallback state, count it as an error (O4 audit F-07).
          countShellFetch("error");
          return res;
        }
        countShellFetch("success");
        if (shellFallbackActive) {
          shellFallbackActive = false;
          await postTelemetry({ name: "cache_recovery", outcome: "recovered" })
            .catch(() => {});
        }
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put(e.request, copy));
        return res;
      })
      .catch(async () => {
        shellFallbackActive = true;
        await postTelemetry({ name: "cache_fallback", outcome: "fallback" })
          .catch(() => {});
        const cached = await caches.match(e.request);
        if (cached) return cached;
        return new Response("TheraPy server is not reachable right now.", {
          status: 503,
          headers: { "Content-Type": "text/plain" },
        });
      })
  );
});

// Push payloads intentionally carry no reflection text, only local navigation.
self.addEventListener("push", (event) => {
  let payload = {};
  try { payload = event.data ? event.data.json() : {}; } catch { payload = {}; }
  const title = payload.title || "TheraPy";
  waitUntilWithTelemetry(event, (async () => {
    await postTelemetry({ name: "push_lifecycle", outcome: "received" })
      .catch(() => {});
    await self.registration.showNotification(title, {
      body: "A reflection is available whenever you want it.",
      icon: "/icon-192.png",
      badge: "/icon-192.png",
      data: { url: payload.url || "/#model" },
      tag: "therapy-reflection-available",
      renotify: false,
    });
    await postTelemetry({ name: "push_lifecycle", outcome: "shown" })
      .catch(() => {});
  })());
});

self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  const target = new URL(
    event.notification.data?.url || "/#model",
    self.location.origin,
  ).href;
  waitUntilWithTelemetry(event, (async () => {
    await postTelemetry({ name: "push_lifecycle", outcome: "clicked" })
      .catch(() => {});
    return self.clients.matchAll({ type: "window", includeUncontrolled: true })
      .then((clients) => {
        for (const client of clients) {
          if (client.url.startsWith(self.location.origin) && "focus" in client) {
            client.navigate(target);
            return client.focus();
          }
        }
        return self.clients.openWindow(target);
      });
  })());
});

self.addEventListener("pushsubscriptionchange", (event) => {
  const outcome = event.newSubscription ? "refreshed" : "deactivated";
  waitUntilWithTelemetry(event, postTelemetry({
    name: "push_lifecycle", outcome,
  }));
});
