"""Browser check of the Roshni web demo: load the page in headless Chrome, screen each example,
and compare the browser's scores with the Python tool on the same photos."""
import json, os, re, shutil, subprocess, sys, time, threading, http.server, functools
sys.path.insert(0, "/home/claude/roshni")
web, results = sys.argv[1], sys.argv[2]
test = "/tmp/webcheck"
shutil.rmtree(test, ignore_errors=True)
shutil.copytree(web, test)
os.makedirs(f"{test}/ort", exist_ok=True)
for f in ("ort.wasm.min.js", "ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"):
    shutil.copy(f"/tmp/ortweb/node_modules/onnxruntime-web/dist/{f}", f"{test}/ort/{f}")
html = open(f"{test}/index.html", encoding="utf-8").read()
html = re.sub(r'<script src="https://cdn[^"]+"></script>', '<script src="ort/ort.wasm.min.js"></script>', html)
open(f"{test}/index.html", "w", encoding="utf-8").write(html)

handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=test)
http.server.SimpleHTTPRequestHandler.extensions_map[".wasm"] = "application/wasm"
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8765), handler)
threading.Thread(target=srv.serve_forever, daemon=True).start()

import torch, pickle, numpy as np, drscreen as ds
cfg = json.load(open(f"{test}/roshni_web.json"))
res = json.load(open(f"{results}/results.json"))
fr = res["frozen"]
model = ds.build_model(fr["arch"], pretrained=False)
model.load_state_dict(torch.load(f"{results}/model_best.pt", map_location="cpu")); model.eval()
gate = pickle.load(open(f"{results}/quality_gate.pkl", "rb"))

from playwright.sync_api import sync_playwright
worst = 0
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
    pg = b.new_page()
    logs = []
    pg.on("console", lambda m: logs.append(m.text))
    pg.goto("http://127.0.0.1:8765/index.html")
    pg.wait_for_function("document.getElementById('status').textContent.startsWith('Ready')", timeout=60000)
    n = pg.locator(".ex").count()
    print("examples on page:", n)
    for i in range(n):
        pg.locator("#status").evaluate("e => e.textContent = ''")
        pg.locator(".ex").nth(i).click()
        pg.wait_for_function("document.getElementById('status').textContent.startsWith('Done')", timeout=60000)
        dec = pg.locator("#decision").text_content()
        sc = pg.locator("#scores").text_content()
        t = pg.locator("#timing").text_content()
        note_langs = pg.locator("#note .n").evaluate_all("els => els.map(e => e.lang + ':' + e.dir)")
        js_ref = float(re.search(r"score ([0-9.]+) \(refer", sc).group(1))
        js_poor = float(re.search(r"Poor-photo score ([0-9.]+)", sc).group(1))
        img = f"{test}/{cfg['examples'][i]['file']}"
        d, p_ref, p_poor, P, e = ds.triage_one(model, gate, img, fr["size"], fr["ref_threshold"], fr["poor_threshold"])
        js = pg.evaluate("window.roshniLast")
        ej, ep = np.array(js["features"]), e[0]
        rel = float(np.linalg.norm(ej - ep) / (np.linalg.norm(ep) + 1e-12))
        dP = float(np.abs(np.array(js["P"]) - P).max())
        worst = max(worst, rel, dP)
        print(f"   features relative difference {rel:.2e}, grade-probability difference {dP:.2e}")
        print(f"{cfg['examples'][i]['file']}: browser {dec} ref {js_ref:.2f} poor {js_poor:.2f} | python "
              f"{ {'REFER':'Refer to an eye specialist','RETAKE':'Retake the photo','RESCREEN':'Rescreen in 12 months'}[d]} "
              f"ref {p_ref:.3f} poor {p_poor:.3f} | {t} | notes {note_langs}")
    pg.set_input_files("#file", sorted(__import__('glob').glob('/tmp/ddr/regular_fundus_images/regular-fundus-validation/Images/*/*.jpg'))[5])
    pg.wait_for_function("document.getElementById('status').textContent.startsWith('Done')", timeout=60000)
    print("uploaded raw photo:", pg.locator("#decision").text_content(), "|", pg.locator("#timing").text_content())
    raw = sorted(__import__('glob').glob('/tmp/ddr/regular_fundus_images/regular-fundus-validation/Images/*/*.jpg'))[5]
    js = pg.evaluate("window.roshniLast")
    d, p_ref, p_poor, P, e = ds.triage_one(model, gate, raw, fr["size"], fr["ref_threshold"], fr["poor_threshold"])
    print(f"   raw photo: features relative difference {np.linalg.norm(np.array(js['features']) - e[0]) / np.linalg.norm(e[0]):.2e}, "
          f"grade-probability difference {np.abs(np.array(js['P']) - P).max():.2e}")
    pg.screenshot(path="/tmp/webcheck_screenshot.png", full_page=True)
    errs = [l for l in logs if "rror" in l]
    print("console errors:", errs[:5])
    b.close()
srv.shutdown()
print(f"largest difference on examples (relative features / probabilities): {worst:.2e}")
