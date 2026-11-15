/* Roshni in the browser: the same frozen tool as the Python code, run with ONNX Runtime Web.
   The photo is processed on this device and never uploaded.
   Pure functions at the top are also tested in Node (tests/test_web.mjs). */

const R = {};

/* PIL "L" conversion, used to find the round fundus inside its black border. */
R.cropBox = function (rgba, w, h, tol) {
  let x0 = w, y0 = h, x1 = -1, y1 = -1, count = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const L = (rgba[i] * 19595 + rgba[i + 1] * 38470 + rgba[i + 2] * 7471 + 0x8000) >> 16;
      if (L > tol) {
        count++;
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  if (count < 0.05 * w * h) return { x: 0, y: 0, w: w, h: h };   // almost all dark: leave it to the quality check
  return { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 };
};

/* Resize exactly like Pillow's Image.resize(..., BILINEAR), which the Python tool uses:
   a triangle filter widened when shrinking, horizontal pass then vertical, 22-bit fixed point. */
R._coeffs = function (inSize, outSize) {
  const scale = inSize / outSize, fscale = Math.max(scale, 1), support = fscale, ss = 1 / fscale;
  const PREC = 22, list = [];
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    let xmin = Math.trunc(center - support + 0.5); if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5); if (xmax > inSize) xmax = inSize;
    const n = xmax - xmin, k = new Float64Array(n);
    let ww = 0;
    for (let x = 0; x < n; x++) { const t = Math.abs((x + xmin - center + 0.5) * ss); k[x] = t < 1 ? 1 - t : 0; ww += k[x]; }
    const ki = new Int32Array(n);
    for (let x = 0; x < n; x++) { const v = (ww !== 0 ? k[x] / ww : k[x]) * (1 << PREC); ki[x] = Math.trunc(v < 0 ? v - 0.5 : v + 0.5); }
    list.push({ xmin: xmin, k: ki });
  }
  return list;
};

R.resizePIL = function (src, sw, sh, dw, dh) {
  const PREC = 22, half = 1 << (PREC - 1), div = 1 << PREC;
  const clip = v => (v < 0 ? 0 : v > 255 ? 255 : v);
  let w = sw, h = sh, buf = src;
  if (dw !== sw) {
    const cs = R._coeffs(sw, dw), out = new Uint8ClampedArray(dw * h * 4);
    for (let y = 0; y < h; y++) for (let xx = 0; xx < dw; xx++) {
      const { xmin, k } = cs[xx];
      let r = half, g = half, b = half;
      for (let x = 0; x < k.length; x++) { const i = (y * w + xmin + x) * 4; r += buf[i] * k[x]; g += buf[i + 1] * k[x]; b += buf[i + 2] * k[x]; }
      const o = (y * dw + xx) * 4;
      out[o] = clip(Math.floor(r / div)); out[o + 1] = clip(Math.floor(g / div)); out[o + 2] = clip(Math.floor(b / div)); out[o + 3] = 255;
    }
    buf = out; w = dw;
  }
  if (dh !== sh) {
    const cs = R._coeffs(sh, dh), out = new Uint8ClampedArray(w * dh * 4);
    for (let yy = 0; yy < dh; yy++) {
      const { xmin, k } = cs[yy];
      for (let x = 0; x < w; x++) {
        let r = half, g = half, b = half;
        for (let y = 0; y < k.length; y++) { const i = ((xmin + y) * w + x) * 4; r += buf[i] * k[y]; g += buf[i + 1] * k[y]; b += buf[i + 2] * k[y]; }
        const o = (yy * w + x) * 4;
        out[o] = clip(Math.floor(r / div)); out[o + 1] = clip(Math.floor(g / div)); out[o + 2] = clip(Math.floor(b / div)); out[o + 3] = 255;
      }
    }
    buf = out; h = dh;
  }
  return buf;
};

R.softmax = function (z) {
  const m = Math.max(...z);
  const e = z.map(v => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / s);
};

R.referableProb = function (P, from) {
  let s = 0;
  for (let k = from; k < P.length; k++) s += P[k];
  return s;
};

/* StandardScaler + logistic regression, exported from scikit-learn. */
R.poorProb = function (emb, gate) {
  if (!gate) return 0;
  let z = gate.intercept;
  for (let c = 0; c < emb.length; c++) z += ((emb[c] - gate.mean[c]) / gate.scale[c]) * gate.coef[c];
  return 1 / (1 + Math.exp(-z));
};

/* Refer whenever referable disease is suspected (a poor photo never blocks a referral);
   otherwise a poor photo gets RETAKE instead of an all-clear. */
R.decide = function (pRef, pPoor, refThr, poorThr) {
  if (pRef >= refThr) return "REFER";
  if (pPoor >= poorThr) return "RETAKE";
  return "RESCREEN";
};

/* Grad-CAM for the referable score, computed exactly without gradients: the network ends in
   global average pooling and one linear layer, so the gradient at the last feature map is the same at
   every position: sum_k (dscore/dlogit_k) * W[k][c]. Matches drscreen.gradcam up to a positive scale. */
R.heatmap = function (fmap, C, H, W, P, headW, from) {
  let sRef = 0;
  for (let k = from; k < P.length; k++) sRef += P[k];
  const g = P.map((p, k) => (k >= from ? p / sRef : 0) - p);
  const alpha = new Float32Array(C);
  for (let c = 0; c < C; c++) {
    let a = 0;
    for (let k = 0; k < P.length; k++) a += g[k] * headW[k][c];
    alpha[c] = a;
  }
  const cam = new Float32Array(H * W);
  for (let c = 0; c < C; c++) {
    const a = alpha[c], off = c * H * W;
    if (a === 0) continue;
    for (let i = 0; i < H * W; i++) cam[i] += a * fmap[off + i];
  }
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < cam.length; i++) {
    cam[i] = Math.max(0, cam[i]);
    if (cam[i] < mn) mn = cam[i];
    if (cam[i] > mx) mx = cam[i];
  }
  for (let i = 0; i < cam.length; i++) cam[i] = (cam[i] - mn) / (mx - mn + 1e-8);
  return cam;
};

if (typeof module !== "undefined") module.exports = R;

/* ------------------------------------------------------------------ browser app */
if (typeof window !== "undefined") {
  const COLORS = { REFER: "#b42318", RETAKE: "#b54708", RESCREEN: "#067647" };
  const TITLES = { REFER: "Refer to an eye specialist", RETAKE: "Retake the photo", RESCREEN: "Rescreen in 12 months" };
  let cfg = null, session = null, lastDecision = null;
  const $ = id => document.getElementById(id);

  function setStatus(t) { $("status").textContent = t; }

  async function init() {
    try {
      setStatus("Loading the model onto this device…");
      cfg = await (await fetch("roshni_web.json")).json();
      ort.env.wasm.numThreads = 1;
      session = await ort.InferenceSession.create("model.onnx", { executionProviders: ["wasm"] });
      buildLanguagePicker();
      buildExamples();
      buildFooter();
      setStatus("Ready. Choose a retinal photo or an example.");
      $("file").disabled = false;
      enableOffline();
    } catch (e) {
      setStatus("Could not load the model: " + e.message);
      console.error(e);
    }
  }

  /* Keep the page, model and runtime on this device so screening works with no internet next time. */
  async function enableOffline() {
    if (!("serviceWorker" in navigator) || !window.caches) return;
    try {
      await navigator.serviceWorker.register("sw.js");
      await navigator.serviceWorker.ready;
      for (let i = 0; i < 60; i++) {   // the first visit downloads about 30 MB into the cache
        if (await caches.match("model.onnx") && await caches.match("ort/ort-wasm-simd-threaded.wasm")) {
          $("offline").textContent = "Saved on this device: works without internet from now on.";
          window.roshniOfflineReady = true;
          return;
        }
        await new Promise(r => setTimeout(r, 1000));
      }
    } catch (e) {
      console.warn("offline mode unavailable", e);
    }
  }

  function buildLanguagePicker() {
    const box = $("langs");
    for (const [code, v] of Object.entries(cfg.langs)) {
      const lab = document.createElement("label");
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.value = code;
      cb.checked = cfg.default_langs.includes(code);
      cb.addEventListener("change", () => { if (lastDecision) renderNote(lastDecision); });
      lab.appendChild(cb);
      lab.appendChild(document.createTextNode(" " + v.name + (v.checked ? "" : " (translation being checked)")));
      box.appendChild(lab);
    }
  }

  function buildExamples() {
    const box = $("examples");
    for (const ex of cfg.examples || []) {
      const b = document.createElement("button");
      b.className = "ex";
      b.title = ex.caption;
      const img = document.createElement("img");
      img.src = ex.file;
      img.alt = ex.caption;
      b.appendChild(img);
      const cap = document.createElement("span");
      cap.textContent = ex.caption;
      b.appendChild(cap);
      b.addEventListener("click", () => loadURL(ex.file));
      box.appendChild(b);
    }
    if (cfg.examples_credit) $("excredit").textContent = cfg.examples_credit;
  }

  function buildFooter() {
    const s = cfg.summary || {};
    const parts = [];
    if (s.fine_tuned) parts.push(`On ${s.fine_tuned.n.toLocaleString("en-US")} unseen photos from India: sensitivity ${(100 * s.fine_tuned.sensitivity).toFixed(1)}%, specificity ${(100 * s.fine_tuned.specificity).toFixed(1)}%.`);
    if (s.external_all) parts.push(`On ${s.external_all.n.toLocaleString("en-US")} photos from Chinese hospitals it never saw: sensitivity ${(100 * s.external_all.sensitivity).toFixed(1)}%, specificity ${(100 * s.external_all.specificity).toFixed(1)}%.`);
    $("results").textContent = parts.join(" ");
    $("thr").textContent = `Referral threshold ${cfg.ref_threshold.toFixed(3)}, frozen before testing (set for ${(100 * cfg.target_sensitivity).toFixed(0)}% sensitivity on validation photos).`;
    if (cfg.repo_url) { $("repo").href = cfg.repo_url; $("repo").hidden = false; }
  }

  function loadURL(url) {
    const img = new Image();
    img.onload = () => run(img);
    img.onerror = () => setStatus("Could not open that image.");
    img.src = url;
  }

  $("file").addEventListener("change", e => {
    const f = e.target.files[0];
    if (!f) return;
    const url = URL.createObjectURL(f);
    loadURL(url);
  });

  function prepare(img) {
    const S = cfg.size;
    const c0 = document.createElement("canvas");
    c0.width = img.naturalWidth; c0.height = img.naturalHeight;
    const x0 = c0.getContext("2d", { willReadFrequently: true });
    x0.drawImage(img, 0, 0);
    const box = R.cropBox(x0.getImageData(0, 0, c0.width, c0.height).data, c0.width, c0.height, cfg.crop_tol);
    const side = Math.max(box.w, box.h);
    const c1 = document.createElement("canvas");
    c1.width = side; c1.height = side;
    const x1 = c1.getContext("2d", { willReadFrequently: true });
    x1.fillStyle = "#000"; x1.fillRect(0, 0, side, side);
    x1.drawImage(c0, box.x, box.y, box.w, box.h, Math.floor((side - box.w) / 2), Math.floor((side - box.h) / 2), box.w, box.h);
    const src = x1.getImageData(0, 0, side, side).data;
    const px = side === S ? src : R.resizePIL(src, side, side, S, S);
    const c2 = $("photo");
    c2.width = S; c2.height = S;
    c2.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(px), S, S), 0, 0);
    const t = new Float32Array(3 * S * S);
    for (let i = 0; i < S * S; i++) {
      for (let ch = 0; ch < 3; ch++) t[ch * S * S + i] = (px[i * 4 + ch] / 255 - cfg.mean[ch]) / cfg.std[ch];
    }
    return { tensor: new ort.Tensor("float32", t, [1, 3, S, S]), pixels: px };
  }

  async function run(img) {
    try {
      setStatus("Screening on this device…");
      const t0 = performance.now();
      const { tensor, pixels } = prepare(img);
      const out = await session.run({ image: tensor });
      const P = R.softmax(Array.from(out.logits.data));
      const emb = out.features.data;
      const pRef = R.referableProb(P, cfg.referable_from);
      const pPoor = R.poorProb(emb, cfg.gate);
      const d = R.decide(pRef, pPoor, cfg.ref_threshold, cfg.poor_threshold);
      const ms = performance.now() - t0;
      window.roshniLast = { P: P, features: Array.from(emb) };   // for automated checks
      const [, C, H, W] = out.feature_map.dims;
      const cam = R.heatmap(out.feature_map.data, C, H, W, P, cfg.head_weight, cfg.referable_from);
      drawHeat(cam, H, W, pixels);
      renderCard(d, pRef, pPoor, ms);
      renderGrades(P);
      renderNote(d);
      lastDecision = d;
      setStatus("Done. Nothing was uploaded.");
    } catch (e) {
      setStatus("Something went wrong: " + e.message);
      console.error(e);
    }
  }

  function drawHeat(cam, H, W, pixels) {
    const S = cfg.size;
    const small = document.createElement("canvas");
    small.width = W; small.height = H;
    const xs = small.getContext("2d");
    const im = xs.createImageData(W, H);
    for (let i = 0; i < H * W; i++) { const v = Math.round(cam[i] * 255); im.data.set([v, v, v, 255], i * 4); }
    xs.putImageData(im, 0, 0);
    const big = document.createElement("canvas");
    big.width = S; big.height = S;
    const xb = big.getContext("2d", { willReadFrequently: true });
    xb.imageSmoothingEnabled = true;
    xb.drawImage(small, 0, 0, S, S);
    const h = xb.getImageData(0, 0, S, S).data;
    const c = $("heat");
    c.width = S; c.height = S;
    const xc = c.getContext("2d");
    const o = xc.createImageData(S, S);
    for (let i = 0; i < S * S; i++) {
      const a = 0.45 * h[i * 4] / 255;
      o.data[i * 4] = pixels[i * 4] * (1 - a) + 255 * a;
      o.data[i * 4 + 1] = pixels[i * 4 + 1] * (1 - a) + 40 * a;
      o.data[i * 4 + 2] = pixels[i * 4 + 2] * (1 - a);
      o.data[i * 4 + 3] = 255;
    }
    xc.putImageData(o, 0, 0);
    $("heatbox").hidden = false;
  }

  function renderCard(d, pRef, pPoor, ms) {
    const card = $("card");
    card.hidden = false;
    card.style.borderLeftColor = COLORS[d];
    $("decision").textContent = TITLES[d];
    $("decision").style.color = COLORS[d];
    const poor = cfg.gate ? `${pPoor.toFixed(2)} (retake at ${cfg.poor_threshold.toFixed(2)} or above)` : "not available";
    $("scores").textContent = `Referable-disease score ${pRef.toFixed(2)} (refer at ${cfg.ref_threshold.toFixed(2)} or above) · Poor-photo score ${poor}`;
    $("timing").textContent = `Screened on this device in ${(ms / 1000).toFixed(2)} s.`;
  }

  function renderGrades(P) {
    const box = $("grades");
    box.replaceChildren();
    P.forEach((p, k) => {
      const row = document.createElement("div");
      row.className = "g";
      const lab = document.createElement("span");
      lab.textContent = `${k} ${cfg.grades[k]}`;
      const bar = document.createElement("span");
      bar.className = "bar";
      const fill = document.createElement("span");
      fill.style.width = `${(100 * p).toFixed(1)}%`;
      bar.appendChild(fill);
      const val = document.createElement("span");
      val.textContent = `${(100 * p).toFixed(0)}%`;
      row.append(lab, bar, val);
      box.appendChild(row);
    });
  }

  function renderNote(d) {
    const chosen = Array.from(document.querySelectorAll("#langs input:checked")).map(x => x.value);
    const langs = chosen.length ? chosen : ["en"];
    const box = $("note");
    box.replaceChildren();
    for (const code of Object.keys(cfg.langs)) {
      if (!langs.includes(code)) continue;
      const div = document.createElement("div");
      div.dir = cfg.langs[code].dir;
      div.lang = code;
      div.className = "n";
      const b = document.createElement("b");
      b.textContent = cfg.langs[code].name + ": ";
      div.appendChild(b);
      div.appendChild(document.createTextNode(cfg.report[d][code]));
      const small = document.createElement("small");
      small.textContent = cfg.report.DISCLAIMER[code];
      div.appendChild(document.createElement("br"));
      div.appendChild(small);
      box.appendChild(div);
    }
    $("notebox").hidden = false;
  }

  init();
}
