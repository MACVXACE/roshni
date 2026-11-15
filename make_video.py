"""Build the 2-minute demo video from the real results: slides plus a screen recording of the browser demo.

  python make_video.py --publish publish --out roshni_video.mp4 [--voice-dir voices] [--ort-dir path/to/onnxruntime-web/dist]

Timeline (the Hacker Guide's structure): problem 0:00-0:15, demo 0:15-1:15, how we know 1:15-1:45, limits 1:45-2:00.
Voices: put problem.m4a, demo.m4a, how.m4a and limits.m4a (any format ffmpeg reads) in --voice-dir; each is laid over
its part. Without voices the video is silent, with the script as captions, ready for a voice-over.
Every number on a slide comes from results/results.json. No names appear (Round 1 is judged blind).
"""
import argparse
import functools
import html
import http.server
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading

W, H = 1280, 720
PARTS = [("problem", 15), ("demo", 60), ("how", 30), ("limits", 15)]
CSS = """
*{box-sizing:border-box} body{margin:0;width:1280px;height:720px;font-family:system-ui,-apple-system,'Segoe UI',Roboto,'Noto Sans',sans-serif;
color:#1f2328;background:#fff;display:flex;flex-direction:column;justify-content:center;padding:64px 88px}
h1{font-size:64px;margin:0 0 12px;letter-spacing:-0.02em} h2{font-size:40px;margin:0 0 28px;letter-spacing:-0.01em}
.sub{font-size:28px;color:#4b5563;max-width:1000px;line-height:1.35} .tag{display:inline-block;margin-top:28px;font-size:20px;
background:#fff4e5;color:#7a4100;border-radius:8px;padding:6px 14px} .row{display:flex;gap:28px}
.stat{flex:1;border:1px solid #e1e4e8;border-radius:14px;padding:26px} .big{font-size:56px;font-weight:700;color:#2f6fdb}
.lbl{font-size:22px;color:#374151;margin-top:8px;line-height:1.3} .src{font-size:15px;color:#6b7280;margin-top:22px}
table{border-collapse:collapse;font-size:24px;width:100%} td,th{padding:12px 14px;border-bottom:1px solid #e1e4e8;text-align:left}
th{color:#6b7280;font-weight:600;font-size:19px} tr.hl td{font-weight:700;color:#1d4ed8} .note{font-size:20px;color:#4b5563;margin-top:20px}
img.fig{max-width:100%;max-height:470px;display:block;margin:0 auto} li{font-size:28px;margin:10px 0;line-height:1.35}
"""


def slide(body):
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"


def pct(v):
    return f"{100 * v:.1f}%"


def stress_chart(rows, path):
    """One panel, every condition: referable eyes given an all-clear, with vs without the quality check."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    st = pd.DataFrame(rows)
    name = "DeepDRiD validation" if (st.photos == "DeepDRiD validation").any() else "APTOS test"
    d = st[st.photos == name].reset_index(drop=True)
    labels = ["original photos" if c == "clean" else f"{c.replace('_', ' ')}, {s}" for c, s in zip(d.change, d.strength)]
    y = np.arange(len(d))[::-1]
    h = 0.38
    a, b = d.referable_all_clear_without_check, d.referable_all_clear_with_check
    fig, ax = plt.subplots(figsize=(11, 4.9), dpi=150)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#e1e4e8")
    ax.tick_params(labelsize=12, colors="#4b5563")
    ax.grid(axis="x", color="#e1e4e8")
    ax.set_axisbelow(True)
    ax.barh(y + h / 2, a, height=h, color="#6e7781", label="without quality check")
    ax.barh(y - h / 2, b, height=h, color="#2f6fdb", label="with quality check")
    top = max(a.max(), b.max(), 1)
    for yy, va, vb in zip(y, a, b):
        ax.text(va + top * 0.01, yy + h / 2, str(int(va)), va="center", fontsize=11, color="#4b5563")
        ax.text(vb + top * 0.01, yy - h / 2, str(int(vb)), va="center", fontsize=11, color="#1f2328", fontweight="bold")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, top * 1.1)
    ax.set_xlabel(f"Referable eyes given an all-clear (of {int(d.referable.iloc[0])})", fontsize=12, color="#4b5563")
    ax.legend(frameon=False, fontsize=12, loc="lower right")
    fig.suptitle(f"{name} photos ({int(d.n.iloc[0])}), each made worse in a fixed way", fontsize=12, x=0.01, ha="left", color="#4b5563")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build_slides(pub, R, tmp):
    r = R["results"]
    slides = {}
    slides["title"] = slide("<h1>Roshni</h1><div class='sub'>Open, fail-safe diabetic eye triage for primary-care clinics in Pakistan"
                            "</div><span class='tag'>Prototype for a student hackathon · not medical advice</span>")
    slides["problem"] = slide(
        "<h2>Pakistan: diabetes is common, eye specialists are not</h2><div class='row'>"
        "<div class='stat'><div class='big'>31.4%</div><div class='lbl'>of adults have diabetes, the highest prevalence in the world</div></div>"
        "<div class='stat'><div class='big'>~29%</div><div class='lbl'>of people with diabetes had diabetic retinopathy in Pakistani studies</div></div>"
        "<div class='stat'><div class='big'>~15</div><div class='lbl'>ophthalmologists per million people, mostly in cities</div></div></div>"
        "<div class='src'>Sources: IDF Diabetes Atlas (Pakistan); Mumtaz et al., Pak J Med Sci 2018; The News, 2026.</div>")
    rows = []
    for k, label in (("colour_baseline", "Baseline: colour histogram"), ("linear_probe", "Baseline: untuned image features"),
                     ("fine_tuned", "Roshni, unseen photos (India)"), ("external_all", "Roshni, new hospitals (China)")):
        if k in r:
            x = r[k]
            rows.append(f"<tr class='{'hl' if k in ('fine_tuned', 'external_all') else ''}'><td>{label}</td><td>{x['AUC']:.2f}</td>"
                        f"<td>{pct(x['sensitivity'])}</td><td>{pct(x['specificity'])}</td><td>{x['missed_FN']}</td></tr>")
    slides["how"] = slide(
        "<h2>How we know it works</h2><table><tr><th>Photos never used for training</th><th>AUC</th><th>Sensitivity</th>"
        "<th>Specificity</th><th>Missed</th></tr>" + "".join(rows) + "</table>"
        f"<div class='note'>Referral threshold frozen on validation photos before testing "
        f"({100 * R['frozen']['target_sensitivity']:.0f}% sensitivity target) · near-duplicate photos kept apart.</div>")
    st = R.get("stress_test") or []
    if st:
        stress_chart(st, os.path.join(tmp, "stress_chart.png"))
        slides["stress"] = slide("<h2>Deliberately worse photos (stress test)</h2>"
                                 "<img class='fig' src='stress_chart.png'>")
    ft = r["fine_tuned"]
    miss = os.path.join(pub, "results", "missed_external.png")
    if os.path.exists(miss):
        shutil.copy(miss, os.path.join(tmp, "missed_external.png"))
        slides["misses"] = slide("<h2>Where it fails</h2><img class='fig' src='missed_external.png' style='width:100%'>"
                                 "<div class='note'>Referable eyes given an all-clear on the external photos. Labels show the "
                                 "experts' grade and the model's score. Photos: DeepDRiD (Liu et al., Patterns 2022), CC BY-SA 4.0.</div>")
    slides["limits"] = slide(
        "<h2>Limits and next step</h2><ul>"
        f"<li>Missed {ft['missed_FN']} of {ft['referable']} referable eyes in the unseen test photos from India.</li>"
        f"<li>On new hospitals' photos it caught only {pct(r['external_all']['sensitivity'])}: each clinic must re-set the cut-off on its own photos.</li>"
        "<li>Not yet tested on Pakistani patients or on low-cost cameras.</li>"
        "<li><b>Next:</b> test the frozen tool with a Pakistani clinic that already screens.</li></ul>"
        "<span class='tag'>Roshni is a prototype, not medical advice</span>")
    return slides


def render(slides, tmp, chrome):
    from playwright.sync_api import sync_playwright
    out = {}
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=chrome, args=["--no-sandbox"]) if chrome else p.chromium.launch()
        pg = b.new_page(viewport={"width": W, "height": H})
        for name, doc in slides.items():
            f = os.path.join(tmp, f"{name}.html")
            open(f, "w", encoding="utf-8").write(doc)
            pg.goto("file://" + f)
            pg.wait_for_timeout(300)
            out[name] = os.path.join(tmp, f"{name}.png")
            pg.screenshot(path=out[name])
        b.close()
    return out


def record_demo(web_dir, tmp, chrome, ort_dir, seconds=60):
    """Screen-record the real browser demo: refer, retake, a poor photo that still refers, the Urdu note, the heat map."""
    from playwright.sync_api import sync_playwright
    site = os.path.join(tmp, "site")
    shutil.copytree(web_dir, site)
    if ort_dir:   # serve ONNX Runtime locally (for machines without internet access to the CDN)
        os.makedirs(os.path.join(site, "ort"), exist_ok=True)
        for f in ("ort.wasm.min.js", "ort-wasm-simd-threaded.wasm", "ort-wasm-simd-threaded.mjs"):
            shutil.copy(os.path.join(ort_dir, f), os.path.join(site, "ort", f))
        page = open(os.path.join(site, "index.html"), encoding="utf-8").read()
        page = re.sub(r'<script src="https://cdn[^"]+"></script>', '<script src="ort/ort.wasm.min.js"></script>', page)
        open(os.path.join(site, "index.html"), "w", encoding="utf-8").write(page)
    http.server.SimpleHTTPRequestHandler.extensions_map[".wasm"] = "application/wasm"
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 8766), functools.partial(http.server.SimpleHTTPRequestHandler, directory=site))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cfg = json.load(open(os.path.join(site, "roshni_web.json"), encoding="utf-8"))
    vw, vh = W, H
    vids = os.path.join(tmp, "vid")
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=chrome, args=["--no-sandbox"]) if chrome else p.chromium.launch()
        ctx = b.new_context(viewport={"width": vw, "height": vh}, device_scale_factor=1, record_video_dir=vids,
                            record_video_size={"width": vw, "height": vh})
        pg = ctx.new_page()
        pg.goto("http://127.0.0.1:8766/index.html")
        pg.wait_for_function("document.getElementById('status').textContent.startsWith('Ready')", timeout=60000)
        pg.wait_for_timeout(9000)   # the opening sentences of the voice-over play over the start screen
        caps = [e["caption"] for e in cfg.get("examples", [])]

        def show(match, hold):
            idx = next((i for i, c in enumerate(caps) if match(c)), None)
            if idx is None:
                return 0
            ex = pg.locator(".ex").nth(idx)
            ex.scroll_into_view_if_needed()
            pg.wait_for_timeout(1200)
            ex.click()
            pg.wait_for_function("document.getElementById('status').textContent.startsWith('Done')", timeout=30000)
            pg.locator("#card").evaluate("e => e.scrollIntoView({behavior: 'smooth', block: 'start'})")
            pg.wait_for_timeout(hold)
            return 1
        used = 0
        used += show(lambda c: "grade 3" in c or "grade 4" in c, 9000)             # referable disease: refer + heat map
        pg.locator("#heatbox").evaluate("e => e.scrollIntoView({behavior: 'smooth', block: 'start'})")
        pg.wait_for_timeout(5000)
        pg.evaluate("window.scrollTo({top: 0, behavior: 'smooth'})")
        pg.wait_for_timeout(1500)
        used += show(lambda c: "poor" in c and ("grade 0" in c or "grade 1" in c), 9000)  # poor photo: retake
        used += show(lambda c: "poor" in c and ("grade 2" in c or "grade 3" in c or "grade 4" in c), 9000)  # poor + disease
        pg.locator("#notebox").evaluate("e => e.scrollIntoView({behavior: 'smooth', block: 'start'})")
        pg.wait_for_timeout(10000)                                                   # the Urdu note, to the end of the part
        path = pg.video.path()
        ctx.close()
        b.close()
    srv.shutdown()
    return path, used


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stderr[-2000:])


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--publish", default="publish")
    a.add_argument("--out", default="roshni_video.mp4")
    a.add_argument("--voice-dir", default=None)
    a.add_argument("--ort-dir", default=None)
    a.add_argument("--chrome", default=shutil.which("google-chrome"))
    x = a.parse_args()
    R = json.load(open(os.path.join(x.publish, "results", "results.json")))
    tmp = tempfile.mkdtemp()
    png = render(build_slides(x.publish, R, tmp), tmp, x.chrome)
    demo, used = record_demo(os.path.join(x.publish, "web"), tmp, x.chrome, x.ort_dir)
    print(f"demo recorded with {used} example photos")
    enc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30"]

    def still(img, sec, out):
        run(["ffmpeg", "-y", "-loop", "1", "-t", str(sec), "-i", img, "-vf", f"scale={W}:{H}", *enc, out])
        return out
    seg = {}
    seg["problem"] = os.path.join(tmp, "p.mp4")
    a1, a2 = still(png["title"], 4, os.path.join(tmp, "t.mp4")), still(png["problem"], 11, os.path.join(tmp, "p2.mp4"))
    run(["ffmpeg", "-y", "-i", a1, "-i", a2, "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]", *enc, seg["problem"]])
    # demo: browser recording on a plain background with a caption, padded or trimmed to 60 s
    seg["demo"] = os.path.join(tmp, "d.mp4")
    capf = os.path.join(tmp, "caption.txt")
    open(capf, "w").write("The real browser demo. The photo is screened on this device and never uploaded.")
    run(["ffmpeg", "-y", "-i", demo, "-filter_complex",
         f"[0:v]scale={W}:{H},tpad=stop_mode=clone:stop_duration=60,"
         f"drawbox=x=0:y=ih-44:w=iw:h=44:color=black@0.55:t=fill,"
         f"drawtext=textfile='{capf}':fontcolor=white:fontsize=20:x=(w-tw)/2:y=h-32[v]",
         "-map", "[v]", "-t", "60", *enc, seg["demo"]])
    seg["how"] = os.path.join(tmp, "h.mp4")
    if "stress" in png:
        b1, b2 = still(png["how"], 18, os.path.join(tmp, "h1.mp4")), still(png["stress"], 12, os.path.join(tmp, "h2.mp4"))
        run(["ffmpeg", "-y", "-i", b1, "-i", b2, "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]", *enc, seg["how"]])
    else:
        still(png["how"], 30, seg["how"])
    seg["limits"] = os.path.join(tmp, "l.mp4")
    if "misses" in png:
        c1, c2 = still(png["misses"], 7, os.path.join(tmp, "l1.mp4")), still(png["limits"], 8, os.path.join(tmp, "l2.mp4"))
        run(["ffmpeg", "-y", "-i", c1, "-i", c2, "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]", "-map", "[v]", *enc, seg["limits"]])
    else:
        still(png["limits"], 15, seg["limits"])
    # voices (optional): one file per part, laid over that part
    parts = []
    for name, sec in PARTS:
        v = None
        if x.voice_dir:
            v = next((os.path.join(x.voice_dir, f) for f in sorted(os.listdir(x.voice_dir)) if f.startswith(name + ".")), None)
        out = os.path.join(tmp, f"{name}_av.mp4")
        if v:
            # trim silence at both ends; if the take is still a little long, speed it up slightly (at most 1.2x) to fit
            trim = ("silenceremove=start_periods=1:start_threshold=-45dB,areverse,"
                    "silenceremove=start_periods=1:start_threshold=-45dB,areverse")
            tv = os.path.join(tmp, f"{name}_trim.wav")
            run(["ffmpeg", "-y", "-i", v, "-af", trim, "-ac", "2", "-ar", "48000", tv])
            vlen = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", tv],
                                        capture_output=True, text=True).stdout.strip())
            tempo = max(1.0, vlen / (sec - 0.3))
            if tempo > 1.2:
                print(f"WARNING: {name} voice is {vlen:.1f} s for a {sec} s slot; it will be cut. Re-record faster or shorter.")
                tempo = 1.2
            print(f"voice {name}: {vlen:.1f} s for {sec} s slot, tempo {tempo:.2f}")
            run(["ffmpeg", "-y", "-i", seg[name], "-i", tv, "-filter_complex",
                 f"[1:a]atempo={tempo:.3f},adelay=150|150,apad,atrim=0:{sec}[a]", "-map", "0:v",
                 "-map", "[a]", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-t", str(sec), out])
        else:
            run(["ffmpeg", "-y", "-i", seg[name], "-f", "lavfi", "-t", str(sec), "-i", "anullsrc=r=48000:cl=stereo",
                 "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-t", str(sec), out])
        parts.append(out)
    lst = os.path.join(tmp, "list.txt")
    open(lst, "w").write("".join(f"file '{p}'\n" for p in parts))
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", x.out])
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", x.out],
                         capture_output=True, text=True).stdout.strip()
    print(f"Wrote {x.out} ({float(dur):.1f} s)")
    for name, f in png.items():
        shutil.copy(f, os.path.splitext(x.out)[0] + f"_{name}.png")


if __name__ == "__main__":
    main()
