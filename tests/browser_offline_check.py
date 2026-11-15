"""Check the browser demo works with no internet after one online visit (Chrome, service worker, localhost)."""
import functools, glob, http.server, sys, threading
from playwright.sync_api import sync_playwright
site = sys.argv[1]
http.server.SimpleHTTPRequestHandler.extensions_map[".wasm"] = "application/wasm"
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8770), functools.partial(http.server.SimpleHTTPRequestHandler, directory=site))
threading.Thread(target=srv.serve_forever, daemon=True).start()
raw = sorted(glob.glob('/tmp/ddr/regular_fundus_images/regular-fundus-validation/Images/*/*.jpg'))[7]
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--no-sandbox"])
    ctx = b.new_context()
    pg = ctx.new_page()
    errs = []
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    pg.goto("http://127.0.0.1:8770/index.html")
    pg.wait_for_function("document.getElementById('status').textContent.startsWith('Ready')", timeout=60000)
    pg.wait_for_function("window.roshniOfflineReady === true", timeout=90000)
    print("online visit:", pg.locator("#offline").text_content())
    srv.shutdown()                  # the server is gone too, not only the network
    ctx.set_offline(True)
    pg.reload()
    pg.wait_for_function("document.getElementById('status').textContent.startsWith('Ready')", timeout=60000)
    print("offline reload: model loaded")
    pg.locator(".ex").first.click()
    pg.wait_for_function("document.getElementById('status').textContent.startsWith('Done')", timeout=30000)
    print("offline example:", pg.locator("#decision").text_content(), "|", pg.locator("#timing").text_content())
    pg.set_input_files("#file", raw)
    pg.wait_for_function("document.getElementById('status').textContent.startsWith('Done') && window.roshniLast", timeout=30000)
    print("offline raw photo:", pg.locator("#decision").text_content(), "|", pg.locator("#timing").text_content())
    m = pg.evaluate("fetch('manifest.json').then(r => r.json()).then(j => j.name)")
    print("manifest offline:", m)
    print("console errors:", errs[:3])
    b.close()
