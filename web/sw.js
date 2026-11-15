/* Roshni offline support: after the first visit, the page, the model and ONNX Runtime are kept on the device,
   so screening works with no internet. VERSION is rewritten by export_web.py from the model's hash, so a new
   model replaces the old cache in one step (model and thresholds always change together). */
const VERSION = "roshni-455f6fd79c0c";
const CORE = ["./", "index.html", "roshni.js", "roshni_web.json", "model.onnx", "manifest.json",
              "icons/icon-192.png", "icons/icon-512.png",
              "ort/ort.wasm.min.js", "ort/ort-wasm-simd-threaded.mjs", "ort/ort-wasm-simd-threaded.wasm"];

self.addEventListener("install", event => {
  event.waitUntil((async () => {
    const cache = await caches.open(VERSION);
    await cache.addAll(CORE);
    try {   // example photos listed in the config: cache them too, but never fail the install over one
      const cfg = await (await cache.match("roshni_web.json")).json();
      await Promise.allSettled((cfg.examples || []).map(e => cache.add(e.file)));
    } catch (e) { /* no examples */ }
    await self.skipWaiting();
  })());
});

self.addEventListener("activate", event => {
  event.waitUntil((async () => {
    for (const key of await caches.keys()) if (key !== VERSION) await caches.delete(key);
    await self.clients.claim();
  })());
});

self.addEventListener("fetch", event => {
  if (event.request.method !== "GET") return;
  event.respondWith((async () => {
    const cache = await caches.open(VERSION);
    const hit = await cache.match(event.request, { ignoreSearch: true });
    if (hit) return hit;
    try {
      const res = await fetch(event.request);
      if (res.ok && new URL(event.request.url).origin === self.location.origin) cache.put(event.request, res.clone());
      return res;
    } catch (e) {
      if (event.request.mode === "navigate") return cache.match("index.html");
      throw e;
    }
  })());
});
