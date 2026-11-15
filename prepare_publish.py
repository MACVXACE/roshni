"""Build the exact folder to put on GitHub, from the code and the Kaggle outputs.

  python prepare_publish.py --outputs roshni_outputs.zip --out publish

Copies the code, the notebook, the filled README and video script, the results (without figures that show
APTOS photos, which we could not confirm may be republished) and the browser demo. Re-exports the browser
demo from the results so it carries the current language list and the repository link.
"""
import argparse
import os
import shutil
import tempfile
import zipfile

CODE = ["drscreen.py", "run_dr.py", "app.py", "fill_report.py", "export_web.py", "local_calibration.py", "build_notebook.py",
        "check_submission.py", "prepare_publish.py", "make_video.py", "requirements.txt",
        "Roshni_DR_Screening.ipynb", "Roshni_Results.ipynb", "PREREGISTRATION_v2.md", "build_v2_notebook.py", "Roshni_v2_OIA-DDR.ipynb", "build_results_notebook.py", "LICENSE", ".gitignore", "README.template.md", "VIDEO_SCRIPT.template.md", "TEAM_INPUTS.md"]
SKIP_RESULTS = {"missed_cases.png", "false_alarms.png", "gradcam.png"}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--outputs", required=True, help="roshni_outputs.zip from Kaggle, or its unzipped folder")
    a.add_argument("--out", default="publish")
    a.add_argument("--repo-url", default=None)
    a.add_argument("--allow-quick", action="store_true", help="testing only")
    x = a.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))
    src = x.outputs
    if src.endswith(".zip"):
        tmp = tempfile.mkdtemp()
        with zipfile.ZipFile(src) as z:
            z.extractall(tmp)
        src = os.path.join(tmp, "roshni_outputs")
    shutil.rmtree(x.out, ignore_errors=True)
    os.makedirs(os.path.join(x.out, "results"))
    for f in CODE:
        if os.path.exists(os.path.join(here, f)):
            shutil.copy(os.path.join(here, f), x.out)
    shutil.copytree(os.path.join(here, "tests"), os.path.join(x.out, "tests"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for f in os.listdir(os.path.join(src, "results")):
        if f not in SKIP_RESULTS:
            shutil.copy(os.path.join(src, "results", f), os.path.join(x.out, "results", f))
    if os.path.isdir(os.path.join(src, "results_v2")):      # v2 (pre-registered): published next to v1, no APTOS photos
        os.makedirs(os.path.join(x.out, "results_v2"), exist_ok=True)
        for f in os.listdir(os.path.join(src, "results_v2")):
            if f not in SKIP_RESULTS:
                shutil.copy(os.path.join(src, "results_v2", f), os.path.join(x.out, "results_v2", f))
    # README.md / VIDEO_SCRIPT.md: re-filled from the results + TEAM_INPUTS.md with the current templates
    cwd = os.getcwd()
    os.chdir(x.out)
    try:
        import subprocess
        import sys
        subprocess.run([sys.executable, "fill_report.py", "--results", "results", "--team", "TEAM_INPUTS.md"]
                       + (["--allow-quick"] if x.allow_quick else []), check=False)
        sys.path.insert(0, os.getcwd())
        import export_web
        shutil.copytree(os.path.join(here, "web"), "web", ignore=shutil.ignore_patterns("model.onnx", "roshni_web.json", "examples"))
        export_web.export("results", "web", [], x.repo_url)
        # examples come from the Kaggle bundle (DeepDRiD photos, CC BY-SA 4.0); the cache is not available here
        ex = os.path.join(src, "web", "examples")
        if os.path.isdir(ex):
            shutil.copytree(ex, "web/examples", dirs_exist_ok=True)
            import json
            cfg = json.load(open("web/roshni_web.json", encoding="utf-8"))
            old = json.load(open(os.path.join(src, "web", "roshni_web.json"), encoding="utf-8"))
            cfg["examples"], cfg["examples_credit"] = old.get("examples", []), old.get("examples_credit", "")
            json.dump(cfg, open("web/roshni_web.json", "w", encoding="utf-8"), ensure_ascii=False)
            export_web._stamp_offline_cache("web")
    finally:
        os.chdir(cwd)
    for d, dirs, _ in os.walk(x.out):
        for sub in [s for s in dirs if s in ("__pycache__", ".pytest_cache")]:
            shutil.rmtree(os.path.join(d, sub), ignore_errors=True)
    print(f"Ready to publish: {x.out}/  (run: python check_submission.py --publish-dir {x.out})")


if __name__ == "__main__":
    main()
