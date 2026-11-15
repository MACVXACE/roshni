"""Check the submission against the official rules and our own plan before anyone presses Submit.

  python check_submission.py --publish-dir publish [--video roshni.mp4]

Prints PASS / FIX / CHECK BY HAND for each item and exits with 1 if anything must be fixed.
"""
import argparse
import sys as _sys
_sys.dont_write_bytecode = True
import json
import os
import re
import subprocess
import sys

DISCLAIMER = "This project is a prototype and is not medical advice."
SECTIONS = ["## The problem", "## What we built", "## What we added", "## How it works", "## Results",
            "## What we'd do next", "## Sources"]
APTOS_FIGURES = {"missed_cases.png", "false_alarms.png", "gradcam.png"}   # contain APTOS photos: keep them off public pages
MAIN_WORD_LIMIT = 950


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--publish-dir", default="publish")
    a.add_argument("--video", default=None)
    x = a.parse_args()
    fixes, hand = [], []

    def ok(msg):
        print("PASS  ", msg)

    def fix(msg):
        fixes.append(msg)
        print("FIX   ", msg)

    readme_p = os.path.join(x.publish_dir, "README.md")
    if not os.path.exists(readme_p):
        fix(f"{readme_p} is missing")
        sys.exit(1)
    r = open(readme_p, encoding="utf-8").read()

    ok("disclaimer line present") if DISCLAIMER in r else fix(f'add the exact line: "{DISCLAIMER}"')
    missing = [s for s in SECTIONS if s not in r]
    ok("all sections of the official template present") if not missing else fix(f"missing sections: {missing}")
    ok('"What we added" says "Our own idea"') if "Our own idea" in r else fix('"What we added" must say "Our own idea"')
    left = re.findall(r"\[TEAM_TO_WRITE[^\]]*\]|\[FROM_RESULTS_JSON[^\]]*\]", r)
    ok("no placeholders left") if not left else fix(f"{len(left)} placeholders left, e.g. {left[0][:70]}")
    m = re.search(r"AI tools:(.*)", r)
    ok("AI tools credited") if m and len(m.group(1).strip()) > 10 and "TEAM_TO_WRITE" not in m.group(1) else fix("credit every AI tool used under Sources")
    for need in ("APTOS", "DeepDRiD", "EfficientNet"):
        ok(f"{need} cited") if need in r.split("## Sources")[-1] else fix(f"cite {need} under Sources")
    main_part = r.split("## Sources")[0]
    words = len(re.sub(r"<!--.*?-->|[|#*>`-]", " ", main_part, flags=re.S).split())
    (ok if words <= MAIN_WORD_LIMIT else fix)(f"main write-up is {words} words (aim for {MAIN_WORD_LIMIT} or fewer before Sources)")
    if "Team:" in r and re.search(r"\*\*Team:\*\*\s*\S", r):
        ok("team names listed")
    else:
        fix("list every team member's name")

    # languages shown to users must be checked by a native speaker
    sys.path.insert(0, x.publish_dir)
    try:
        import drscreen as ds
        unchecked = [c for c in ds.shown_langs() if not ds.LANGS[c]["checked"]]
        ok("every language shown has been checked by a native speaker") if not unchecked else \
            fix(f"languages shown but not yet checked: {unchecked} (reviewer: {[ds.LANGS[c]['reviewer'] for c in unchecked]})")
    except Exception as e:  # noqa: BLE001
        fix(f"could not import drscreen.py from {x.publish_dir}: {e!r}")

    # public folder hygiene
    found = [os.path.join(d, f) for d, _, fs in os.walk(x.publish_dir) for f in fs if f in APTOS_FIGURES]
    ok("no APTOS photos in the public folder") if not found else fix(f"remove APTOS figures from the public folder: {found}")
    for must in ("web/index.html", "web/roshni.js", "web/model.onnx", "web/roshni_web.json", "web/sw.js", "web/manifest.json",
                 "web/ort/ort.wasm.min.js", "web/ort/ort-wasm-simd-threaded.wasm", "web/ort/ort-wasm-simd-threaded.mjs",
                 "Roshni_DR_Screening.ipynb"):
        ok(f"{must} present") if os.path.exists(os.path.join(x.publish_dir, must)) else fix(f"{must} missing")
    nbp = os.path.join(x.publish_dir, "Roshni_Results.ipynb")
    if os.path.exists(nbp):
        nbj = json.load(open(nbp, encoding="utf-8"))
        cells = [cl for cl in nbj["cells"] if cl["cell_type"] == "code"]
        errs = [o for cl in cells for o in cl.get("outputs", []) if o.get("output_type") == "error"]
        empty = [i for i, cl in enumerate(cells) if not cl.get("outputs")]
        ok("Roshni_Results.ipynb saved with outputs visible, no errors (official rule)") if not errs and not empty else \
            fix(f"Roshni_Results.ipynb: {len(errs)} error output(s), {len(empty)} code cell(s) without output; re-execute it")
    else:
        fix("Roshni_Results.ipynb missing: the rules ask for a notebook saved with its outputs visible")
    ok("LICENSE present") if os.path.exists(os.path.join(x.publish_dir, "LICENSE")) else fix("LICENSE missing")
    src_block = r.split("## Sources", 1)[1].split("\n---", 1)[0] if "## Sources" in r else ""
    unlinked = [ln.strip() for ln in src_block.splitlines() if ln.strip().startswith("- ") and "http" not in ln and "TEAM_TO_WRITE" not in ln]
    ok("every credit in Sources has a link (official rule)") if not unlinked else fix(f"credits without a link: {unlinked}")
    big = [(os.path.join(d, f), os.path.getsize(os.path.join(d, f)) / 1e6) for d, _, fs in os.walk(x.publish_dir) for f in fs
           if ".git" not in d.split(os.sep) and os.path.getsize(os.path.join(d, f)) > 25e6]   # git's own packs are not uploaded files
    ok("no file over GitHub's 25 MB web-upload limit") if not big else fix(f"files over 25 MB: {big}")

    if x.video:
        try:
            dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", x.video],
                                       capture_output=True, text=True).stdout.strip())
            ok(f"video is {dur:.0f} s") if dur <= 120 else fix(f"video is {dur:.0f} s; the limit is 120 s")
        except ValueError:
            fix(f"could not read {x.video}")

    links = sorted(set(re.findall(r"https?://[^\s)>\]]+", r)))
    hand.append("Open each of these in a private/incognito window, logged out (the official checklist asks for this):")
    hand += [f"   {u}" for u in links]
    hand.append("Video: unlisted on YouTube or 'anyone with the link' on Google Drive; open it logged out.")
    hand.append("Form: every teammate's name, email and school spelled correctly; track = AI + Health.")
    print("\nCHECK BY HAND")
    for h in hand:
        print("  ", h)
    print(f"\n{len(fixes)} item(s) to fix." if fixes else "\nNothing to fix. Do the hand checks, then submit.")
    sys.exit(1 if fixes else 0)


if __name__ == "__main__":
    main()
