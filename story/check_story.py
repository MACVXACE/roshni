"""Check the story page: loads with no errors, every image and link works, scrolling runs, the demo is reachable.
Run from the repository root:  python3 story/check_story.py   (needs: pip install playwright; python3 -m playwright install chromium)
Optional:  python3 story/check_story.py --shots out_dir   to save screenshots while scrolling."""
import argparse, functools, http.server, os, sys, threading
from playwright.sync_api import sync_playwright

a = argparse.ArgumentParser(); a.add_argument("--shots"); a.add_argument("--chrome"); x = a.parse_args()
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
class Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass
Q.extensions_map[".wasm"] = "application/wasm"
srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8790), functools.partial(Q, directory=root))
threading.Thread(target=srv.serve_forever, daemon=True).start()
ok = True
with sync_playwright() as p:
    b = p.chromium.launch(executable_path=x.chrome, args=["--no-sandbox"]) if x.chrome else p.chromium.launch()
    for name, vp, dsf in [("laptop", (1440, 900), 1), ("video", (1280, 720), 1), ("phone", (390, 844), 2)]:
        ctx = b.new_context(viewport={"width": vp[0], "height": vp[1]}, device_scale_factor=dsf)
        pg = ctx.new_page(); errs, bad = [], []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.on("response", lambda r: bad.append(f"{r.status} {r.url}") if r.status >= 400 else None)
        pg.goto("http://127.0.0.1:8790/story/index.html"); pg.wait_for_load_state("networkidle")
        broken = pg.evaluate("[...document.images].filter(i => !i.complete || i.naturalWidth === 0).map(i => i.src)")
        total = pg.evaluate("document.documentElement.scrollHeight - innerHeight")
        n = 14
        for i in range(n + 1):
            pg.evaluate(f"window.scrollTo(0, {int(total * i / n)})"); pg.wait_for_timeout(450)
            if x.shots:
                os.makedirs(x.shots, exist_ok=True); pg.screenshot(path=f"{x.shots}/{name}_{i:02d}.png")
        demo = pg.evaluate("fetch('../web/index.html').then(r => r.status)")
        print(f"{name}: errors={errs} http_errors={bad} broken_images={broken} demo_link_status={demo}")
        ok &= not errs and not bad and not broken and demo == 200
        ctx.close()
    b.close()
srv.shutdown()
print("STORY PAGE OK" if ok else "STORY PAGE HAS PROBLEMS"); sys.exit(0 if ok else 1)
