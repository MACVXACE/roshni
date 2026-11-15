"""Re-make docs/roshni_demo.png (the screenshot at the top of the README) from the browser demo.

Run after any change to the page's look, so the README shows the real current demo:
  python tests/make_screenshot.py web docs/roshni_demo.png
Needs: pip install playwright && python -m playwright install chromium (or Google Chrome installed).
It serves web/ locally, opens it at 1280 px, screens the first example photo (a DeepDRiD photo) and saves the
photo panel and the result panel.
"""
import functools
import http.server
import shutil
import sys
import threading

from PIL import Image
from playwright.sync_api import sync_playwright


def main(site, out, port=8790):
    http.server.SimpleHTTPRequestHandler.log_message = lambda *a: None
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=site)
    http.server.SimpleHTTPRequestHandler.extensions_map[".wasm"] = "application/wasm"
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            chrome = shutil.which("google-chrome") or shutil.which("chromium")
            b = p.chromium.launch(executable_path=chrome, args=["--no-sandbox"]) if chrome else p.chromium.launch()
            pg = b.new_page(viewport={"width": 1280, "height": 900}, device_scale_factor=2)
            pg.goto(f"http://127.0.0.1:{port}/index.html")
            pg.wait_for_function("document.getElementById('status').textContent.startsWith('Ready')", timeout=90000)
            pg.locator(".ex").nth(0).click()
            pg.wait_for_function("document.getElementById('status').textContent.startsWith('Done')", timeout=30000)
            pg.wait_for_timeout(800)
            tool = pg.locator("#tool")
            if tool.count():                       # the photo panel and the result panel, side by side
                tool.screenshot(path="/tmp/_roshni_shot.png")
                im = Image.open("/tmp/_roshni_shot.png")
                w, h = im.size
                top = im.crop((0, 0, w, min(h, int(0.80 * w))))
            else:                                  # older page layout: header + both panels
                pg.evaluate("window.scrollTo(0, 0)")
                pg.screenshot(path="/tmp/_roshni_shot.png", full_page=True)
                im = Image.open("/tmp/_roshni_shot.png")
                w, h = im.size
                top = im.crop((int(0.07 * w), 0, int(0.93 * w), min(h, int(0.78 * w))))
            b.close()
    finally:
        srv.shutdown()
    top.thumbnail((1000, 1000), Image.LANCZOS)
    top.convert("P", palette=Image.ADAPTIVE, colors=256).save(out, optimize=True)
    print(f"Wrote {out} ({top.size[0]}x{top.size[1]})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "web", sys.argv[2] if len(sys.argv) > 2 else "docs/roshni_demo.png")
