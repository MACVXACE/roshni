"""Builds Roshni_DR_Screening.ipynb from the .py files so the notebook runs on its own (Kaggle or Colab)."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
src = {f: open(f, encoding="utf-8").read() for f in ["drscreen.py", "run_dr.py", "app.py", "fill_report.py", "export_web.py",
                                                       "README.template.md", "VIDEO_SCRIPT.template.md",
                                                       "web/index.html", "web/roshni.js"]}

cells = [
    md("# Roshni: fail-safe diabetic eye triage for primary-care diabetes clinics in Pakistan\n\n"
       "**Track:** AI + Health. **Team:** Varun Bajpai (India), Laziza Farrukhovna (Uzbekistan), Zainab Jamshed (Pakistan).  \n"
       "**User:** a doctor or nurse at a primary-care diabetes clinic in Pakistan that can take retinal photos but has no eye specialist on site. "
       "**Moment:** right after the photo is taken at a routine diabetes visit.  \n"
       "**Output:** REFER to an eye specialist, RETAKE the photo, or RESCREEN in 12 months, with a note in English and Urdu.\n\n"
       "> This project is a prototype and is not medical advice.\n\n"
       "### Before you run\n"
       "**Kaggle (recommended, no big download):** New Notebook → *Add Input* → search **APTOS 2019 Blindness Detection** "
       "(join the competition and accept its rules first) → *Settings*: Accelerator **GPU**, Internet **On** "
       "(needs a phone-verified account). Then **Save Version → Save & Run All (Commit)**: it runs in the background, "
       "so you can close the tab. When it finishes, open the version's **Output** and download `roshni_outputs.zip`.\n\n"
       "**Colab:** Runtime → GPU. You will be asked for your `kaggle.json` API token (Kaggle → Settings → Create New Token), "
       "and the competition rules must already be accepted on Kaggle. The download is about 10 GB.\n\n"
       "Full run: roughly 45 to 60 minutes on a free GPU, including a quick check, the stress test and the web-demo export.\n\n"
       "**What it produces:** results with baselines, a held-out test and an external test; a stress test on deliberately worse photos "
       "(planned before any result); the write-up and video script filled from the results; and `web/`, a demo that runs "
       "entirely in the browser (publish it with GitHub Pages)."),
    md("## 1. Code\nWritten to disk so the notebook is self-contained."),
    code("%%writefile drscreen.py\n" + src["drscreen.py"]),
    code("%%writefile run_dr.py\n" + src["run_dr.py"]),
    code("%%writefile app.py\n" + src["app.py"]),
    code("%%writefile fill_report.py\n" + src["fill_report.py"]),
    code("%%writefile README.template.md\n" + src["README.template.md"]),
    code("%%writefile VIDEO_SCRIPT.template.md\n" + src["VIDEO_SCRIPT.template.md"]),
    code("%%writefile export_web.py\n" + src["export_web.py"]),
    code("%%writefile web_index.html\n" + src["web/index.html"]),
    code("%%writefile web_roshni.js\n" + src["web/roshni.js"]),
    code("import os, subprocess, sys\n"
         "subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'onnx', 'onnxruntime'], check=False)   # for the browser-demo export\n"
         "import torch, torchvision\n"
         "print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (this will be slow)')"),
    md("## 2. Data\n**Training data:** APTOS 2019 (3,662 graded photos, Aravind Eye Hospital, rural India).  \n"
       "**External test, never used to train the grading model:** DeepDRiD regular fundus photos, training and validation parts "
       "(1,600 graded photos from 400 patients, with photo-quality labels; China; CC BY-SA 4.0)."),
    code("import glob\n"
         "APTOS = os.environ.get('APTOS_DIR') or next((c for c in ['/kaggle/input/aptos2019-blindness-detection',\n"
         "        '/kaggle/input/competitions/aptos2019-blindness-detection', '/content/aptos', 'aptos'] if os.path.exists(os.path.join(c, 'train.csv'))), None)\n"
         "if APTOS is None:   # Colab route through the Kaggle API\n"
         "    from google.colab import files\n"
         "    print('Upload kaggle.json'); files.upload()\n"
         "    os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)\n"
         "    os.replace('kaggle.json', os.path.expanduser('~/.kaggle/kaggle.json')); os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)\n"
         "    os.system('kaggle competitions download -c aptos2019-blindness-detection -p /content/aptos_zip')\n"
         "    os.system('cd /content/aptos_zip && unzip -q -o aptos2019-blindness-detection.zip \"train_images/*\" train.csv -d /content/aptos')\n"
         "    APTOS = '/content/aptos'\n"
         "print('APTOS folder:', APTOS, '| photos:', len(glob.glob(os.path.join(APTOS, 'train_images', '*'))))\n\n"
         "DDR = os.environ.get('DDR_ROOT', 'DeepDRiD')\n"
         "if not os.path.exists(os.path.join(DDR, 'regular_fundus_images', 'regular-fundus-validation', 'regular-fundus-validation.csv')):\n"
         "    os.system(f'git clone -q --depth 1 --filter=blob:none --sparse https://github.com/deepdrdoc/DeepDRiD.git {DDR}')\n"
         "    os.system(f'cd {DDR} && git sparse-checkout set regular_fundus_images/regular-fundus-training regular_fundus_images/regular-fundus-validation')\n"
         "print('DeepDRiD photos:', len(glob.glob(os.path.join(DDR, 'regular_fundus_images', '*', 'Images', '*', '*.jpg'))))"),
    md("## 3. Quick check (about 5 minutes)\nRuns everything on 300 photos for 2 epochs, so a broken step shows up early. "
       "Its numbers mean nothing; do not report them."),
    code("import run_dr\n"
         "WORKERS = min(4, os.cpu_count() or 2)\n"
         "SIZE = int(os.environ.get('SIZE', 384)); PRETRAINED = os.environ.get('PRETRAINED', '1') == '1'\n"
         "QUICK_EPOCHS = int(os.environ.get('QUICK_EPOCHS', 2))\n"
         "quick = run_dr.run(run_dr.RunConfig(f'{APTOS}/train.csv', f'{APTOS}/train_images', DDR, out='quick_check', cache='cache',\n"
         "                                    size=SIZE, pretrained=PRETRAINED, epochs=QUICK_EPOCHS, workers=WORKERS, max_images=300))\n"
         "quick['summary']"),
    md("## 4. Full run\nThe 90% sensitivity target, the photo-quality cut-off (0.5) and every training setting were fixed "
       "before any result was seen. The referral threshold is set on the validation split and frozen; the APTOS test split "
       "and DeepDRiD are scored once, after that."),
    code("EPOCHS = int(os.environ.get('EPOCHS', 12))\n"
         "cfg = run_dr.RunConfig(f'{APTOS}/train.csv', f'{APTOS}/train_images', DDR, out='results', cache='cache',\n"
         "                       size=SIZE, pretrained=PRETRAINED, epochs=EPOCHS, workers=WORKERS)\n"
         "out = run_dr.run(cfg)\n"
         "out['summary']"),
    md("## 5. Results the Health track asks for: missed cases and false alarms, by severity and by group"),
    code("from IPython.display import Image as Show, display\n"
         "import pandas as pd, json, traceback\n"
         "try:\n"
         "    print('Frozen settings:', out['frozen'])\n"
         "    print('Near-duplicate photos found in APTOS:', out['duplicates'])\n"
         "    display(out['aptos_flagged_by_grade'])          # how many of each grade were sent to a specialist\n"
         "    print('5-grade agreement (quadratic weighted kappa):', round(out['aptos_grades']['QWK'], 3))\n"
         "    if 'workflow' in out:\n"
         "        display(pd.DataFrame(out['workflow']).drop(index=[], errors='ignore'))\n"
         "        print('Photo-quality check:', out['quality_gate'])\n"
         "        display(pd.DataFrame(out['external_by_source']))\n"
         "    print('Speed on the CPU alone (no graphics card):', out.get('cpu_speed'))\n"
         "    display(Show('results/roc.png'))\n"
         "    if os.path.exists('results/stress_test.csv'):\n"
         "        print('Stress test (planned before results): the frozen tool on deliberately worse photos')\n"
         "        display(pd.read_csv('results/stress_test.csv'))\n"
         "        display(Show('results/stress_examples.png')); display(Show('results/stress_test.png'))\n"
         "except Exception:\n"
         "    traceback.print_exc()   # a display problem must never stop the run"),
    md("## 6. Where it fails\nLook at each missed eye and write down a reason (blur, small lesions, edge of the photo, label doubt)."),
    code("te = pd.read_csv('results/aptos_test_predictions.csv')\n"
         "try:\n"
         "    for f in ['results/missed_cases.png', 'results/false_alarms.png', 'results/missed_external.png']:\n"
         "        if os.path.exists(f): display(Show(f))\n"
         "    thr = out['frozen']['ref_threshold']\n"
         "    display(te[(te.grade >= 2) & (te.p_referable < thr)].sort_values('p_referable').head(10))\n"
         "except Exception:\n"
         "    traceback.print_exc()"),
    md("## 7. Where the model looked (Grad-CAM) on a few test photos"),
    code("import drscreen as ds, matplotlib.pyplot as plt\n"
         "from PIL import Image as PILImage\n"
         "try:\n"
         "    pick = pd.concat([te[te.grade == g].head(1) for g in range(5)])\n"
         "    fig, axes = plt.subplots(1, len(pick), figsize=(3 * len(pick), 3.3))\n"
         "    for ax, r in zip(axes, pick.itertuples()):\n"
         "        ax.imshow(ds.gradcam(out['model'], PILImage.open(r.cached), cfg.size)); ax.axis('off')\n"
         "        ax.set_title(f'grade {r.grade}, score {r.p_referable:.2f}', fontsize=9)\n"
         "    plt.tight_layout(); plt.savefig('results/gradcam.png', dpi=120); plt.show()\n"
         "except Exception:\n"
         "    traceback.print_exc()"),
    md("## 8. Write-up and video script, filled from results/\nEvery number comes from the files this run wrote. "
       "It refuses quick-check numbers and writes nothing if any value is missing. Then fill the [TEAM_TO_WRITE] parts by hand."),
    code("r = subprocess.run([sys.executable, 'fill_report.py'] + (['--allow-quick'] if os.environ.get('ALLOW_QUICK') == '1' else []),\n"
         "                   capture_output=True, text=True)\n"
         "print(r.stdout, r.stderr)"),
    md("## 9. Browser demo (the main demo)\nExports the frozen model to ONNX and checks that it gives the same answers as "
       "PyTorch. The `web/` folder is a complete website: the photo is screened inside the visitor's browser and never "
       "uploaded. Put the folder in the GitHub repository and switch on GitHub Pages to get a permanent link."),
    code("import shutil\n"
         "try:\n"
         "    import export_web\n"
         "    os.makedirs('web', exist_ok=True)\n"
         "    shutil.copy('web_index.html', 'web/index.html'); shutil.copy('web_roshni.js', 'web/roshni.js')\n"
         "    check = list(pd.read_csv('results/deepdrid_predictions.csv').query(\"part == 'validation'\").path[:5]) "
         "if os.path.exists('results/deepdrid_predictions.csv') else list(te.path[:5])\n"
         "    export_web.export('results', 'web', check)\n"
         "    print(sorted(os.listdir('web')))\n"
         "except Exception:\n"
         "    traceback.print_exc()   # the results are already saved; the browser demo can be rebuilt from them later"),
    md("## 10. Download everything\nOne zip with the results, the filled write-up and video script, and the browser demo. "
       "On a committed (Save & Run All) run, download it from the version's Output tab."),
    code("pack = 'roshni_outputs'\n"
         "shutil.rmtree(pack, ignore_errors=True); os.makedirs(pack)\n"
         "for f in ('README.md', 'VIDEO_SCRIPT.md'):\n"
         "    if os.path.exists(f): shutil.copy(f, pack)\n"
         "shutil.copytree('results', f'{pack}/results')\n"
         "if os.path.isdir('web'): shutil.copytree('web', f'{pack}/web')\n"
         "shutil.make_archive(pack, 'zip', '.', pack)\n"
         "print(f'{pack}.zip', round(os.path.getsize(pack + '.zip') / 1e6, 1), 'MB')\n"
         "# keep the saved output small: the raw DeepDRiD copy and the image cache are no longer needed\n"
         "if os.environ.get('KEEP_DATA') != '1':\n"
         "    shutil.rmtree('DeepDRiD', ignore_errors=True); shutil.rmtree('cache', ignore_errors=True); shutil.rmtree(pack, ignore_errors=True)"),
    md("## 11. Gradio app (optional, interactive sessions only)\nThe main demo is the browser version in section 9, which keeps "
       "working after this notebook stops. This app gives a public link for 72 hours only. To try it, set `LAUNCH_GRADIO = True`, "
       "set `KEEP_DATA=1` before running section 10 (so the example photos are kept), and run this cell."),
    code("LAUNCH_GRADIO = False\n"
         "if LAUNCH_GRADIO:\n"
         "    try:\n"
         "        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gradio'], check=False)\n"
         "        os.makedirs('examples', exist_ok=True)\n"
         "        if os.path.exists('results/deepdrid_predictions.csv'):\n"
         "            dd = pd.read_csv('results/deepdrid_predictions.csv')\n"
         "            v = dd[dd.part == 'validation']\n"
         "            for name in ('RETAKE', 'REFER', 'RESCREEN'):\n"
         "                for r in v[v.decision == name].head(2).itertuples():\n"
         "                    if os.path.exists(r.cached): shutil.copy(r.cached, f'examples/deepdrid_{name.lower()}_{os.path.basename(r.cached)}')\n"
         "        import app\n"
         "        app.build_app('results', 'examples').launch(share=True)\n"
         "    except Exception:\n"
         "        traceback.print_exc()"),
    md("## Credits\nAPTOS 2019 Blindness Detection (Aravind Eye Hospital, Kaggle); DeepDRiD (Liu et al., Patterns 2022, CC BY-SA 4.0); "
       "PyTorch and torchvision (EfficientNet-B0, ImageNet weights); scikit-learn; Gradio; Grad-CAM (Selvaraju et al. 2017). "
       "AI tools used: list every assistant here."),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"},
                                                 "accelerator": "GPU", "colab": {"provenance": []}})
nbf.write(nb, "Roshni_DR_Screening.ipynb")
print("wrote Roshni_DR_Screening.ipynb with", len(cells), "cells")
