"""Builds Roshni_v2_OIA-DDR.ipynb: the pre-registered v2 run (PREREGISTRATION_v2.md) on Kaggle.
Same code as v1 plus the OIA-DDR photos in training and validation. Writes results_v2/ and roshni_v2_outputs.zip."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
src = {f: open(f, encoding="utf-8").read() for f in ["drscreen.py", "run_dr.py", "PREREGISTRATION_v2.md"]}

cells = [
    md("# Roshni v2: more cameras in training (pre-registered)\n\n"
       "This run follows `PREREGISTRATION_v2.md`, committed to GitHub **before** this notebook was run. "
       "It is the v1 pipeline with one change: the OIA-DDR photos (147 hospitals, 42 camera models) are added to "
       "training and validation. APTOS test and DeepDRiD are scored once, after the cut-off is frozen.\n\n"
       "### Before you run (Kaggle)\n"
       "1. *Add Input*: **APTOS 2019 Blindness Detection** (competition) and the dataset **ddrdataset** (by mariaherrerot).\n"
       "2. *Settings*: Accelerator **GPU**, Internet **On**.\n"
       "3. **Save Version → Save & Run All (Commit)**. About 1 to 1.5 hours. Then download `roshni_v2_outputs.zip` from the "
       "version's Output.\n\n> This project is a prototype and is not medical advice."),
    md("## The pre-registration (as committed)"),
    md(src["PREREGISTRATION_v2.md"]),
    md("## 1. Code (the same files as the GitHub repository)"),
    code("%%writefile drscreen.py\n" + src["drscreen.py"]),
    code("%%writefile run_dr.py\n" + src["run_dr.py"]),
    code("import os, sys, glob, subprocess, hashlib\n"
         "import torch, torchvision\n"
         "print('torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none (this will be slow)')\n"
         "print('code fingerprint (sha256, first 12):', {f: hashlib.sha256(open(f, 'rb').read()).hexdigest()[:12] for f in ['drscreen.py', 'run_dr.py']})"),
    md("## 2. Data"),
    code("APTOS = os.environ.get('APTOS_DIR') or next((c for c in ['/kaggle/input/aptos2019-blindness-detection',\n"
         "        '/kaggle/input/competitions/aptos2019-blindness-detection', 'aptos'] if os.path.exists(os.path.join(c, 'train.csv'))), None)\n"
         "assert APTOS, 'Attach the APTOS 2019 Blindness Detection competition data (Add Input)'\n"
         "print('APTOS:', APTOS, '| photos:', len(glob.glob(os.path.join(APTOS, 'train_images', '*'))))\n\n"
         "DDR = os.environ.get('DDR_ROOT', 'DeepDRiD')      # DeepDRiD: the external test, never trained on\n"
         "if not os.path.exists(os.path.join(DDR, 'regular_fundus_images', 'regular-fundus-validation', 'regular-fundus-validation.csv')):\n"
         "    os.system(f'git clone -q --depth 1 --filter=blob:none --sparse https://github.com/deepdrdoc/DeepDRiD.git {DDR}')\n"
         "    os.system(f'cd {DDR} && git sparse-checkout set regular_fundus_images/regular-fundus-training regular_fundus_images/regular-fundus-validation')\n"
         "print('DeepDRiD photos:', len(glob.glob(os.path.join(DDR, 'regular_fundus_images', '*', 'Images', '*', '*.jpg'))))\n\n"
         "def find_oiaddr():\n"
         "    if os.environ.get('OIADDR_ROOT'): return os.environ['OIADDR_ROOT']\n"
         "    for d, subdirs, _ in os.walk('/kaggle/input'):\n"
         "        name = os.path.basename(d).lower()\n"
         "        if 'ddr' in name and 'aptos' not in d.lower() and 'deepdrid' not in d.lower():\n"
         "            return d\n"
         "        if d.count(os.sep) > 5: subdirs[:] = []\n"
         "OIADDR = find_oiaddr()\n"
         "assert OIADDR, 'Attach the ddrdataset (mariaherrerot) with Add Input, or set OIADDR_ROOT'\n"
         "print('OIA-DDR folder:', OIADDR)\n"
         "for d, subdirs, fs in os.walk(OIADDR):     # a short map of the folder, in case the layout needs checking\n"
         "    depth = d[len(OIADDR):].count(os.sep)\n"
         "    if depth <= 3:\n"
         "        imgs = sum(f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')) for f in fs)\n"
         "        other = [f for f in fs if not f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))][:5]\n"
         "        print('  ' * depth + (os.path.basename(d) or d) + f'/  images: {imgs}  other files: {other}')\n"
         "    if depth >= 3: subdirs[:] = []\n"
         "import run_dr\n"
         "ox = run_dr.load_oiaddr(OIADDR)\n"
         "assert len(ox) > int(os.environ.get('MIN_OIADDR', 5000)), f'Only {len(ox)} OIA-DDR photos with grades found: send the folder map above to Claude before running'"),
    md("## 3. Quick check (about 5 minutes)\nEverything on a few hundred photos for 2 epochs, so a broken step shows up early. Its numbers mean nothing."),
    code("WORKERS = min(4, os.cpu_count() or 2)\n"
         "TESTQ = os.environ.get('ROSHNI_TEST') == '1'\n"
         "quick = run_dr.run(run_dr.RunConfig(f'{APTOS}/train.csv', f'{APTOS}/train_images', DDR, oiaddr_root=OIADDR,\n"
         "                                    out='quick_check_v2', cache='cache', epochs=1 if TESTQ else 2, workers=WORKERS, max_images=150 if TESTQ else 300,\n"
         "                                    size=128 if TESTQ else 384, pretrained=not TESTQ))\n"
         "quick['summary']"),
    md("## 4. Full v2 run (settings as pre-registered)"),
    code("TEST = os.environ.get('ROSHNI_TEST') == '1'     # small runs to check the code only; never for results\n"
         "cfg = run_dr.RunConfig(f'{APTOS}/train.csv', f'{APTOS}/train_images', DDR, oiaddr_root=OIADDR,\n"
         "                       out='results_v2', cache='cache', workers=WORKERS)\n"
         "if TEST: cfg.epochs, cfg.size, cfg.pretrained, cfg.max_images = 1, 128, False, 150\n"
         "assert TEST or (cfg.arch, cfg.size, cfg.epochs, cfg.lr, cfg.batch_size, cfg.seed, cfg.target_sensitivity, cfg.poor_threshold) == \\\n"
         "       ('efficientnet_b0', 384, 12, 3e-4, 32, 0, 0.90, 0.50), 'settings differ from the pre-registration'\n"
         "out = run_dr.run(cfg)\n"
         "out['summary']"),
    md("## 5. The pre-registered decision"),
    code("import json\n"
         "r = json.load(open('results_v2/results.json'))\n"
         "ext, wf, ap = r['results']['external_all'], r['workflow']['with_quality_gate'], r['results']['fine_tuned']\n"
         "checks = {'DeepDRiD sensitivity >= 66.1%': ext['sensitivity'] >= 0.661,\n"
         "          'DeepDRiD specificity >= 85.0%': ext['specificity'] >= 0.850,\n"
         "          'APTOS test AUC >= 0.950': ap['AUC'] >= 0.950,\n"
         "          'false all-clears (400-photo workflow, with check) <= 46': wf['referable_given_all_clear'] <= 46,\n"
         "          'fewer than 16 DeepDRiD photos with a near-copy in OIA-DDR training/validation': r.get('deepdrid_near_copies_in_oiaddr', 0) < 16}\n"
         "for k, v in checks.items(): print(('PASS ' if v else 'FAIL ') + k)\n"
         "print(f\"DeepDRiD: sensitivity {ext['sensitivity']:.1%}, specificity {ext['specificity']:.1%}, AUC {ext['AUC']:.3f}; \"\n"
         "      f\"false all-clears {wf['referable_given_all_clear']}; APTOS test AUC {ap['AUC']:.3f}\")\n"
         "print('Near-copies of training photos among DeepDRiD photos:', r.get('deepdrid_near_copies_in_oiaddr'))\n"
         "print('v2 becomes the demo model' if all(checks.values()) else 'v1 stays the demo model; v2 is reported in full')"),
    md("## 6. Download\nDownload `roshni_v2_outputs.zip` from this version's Output tab. It holds `results_v2/` "
       "(no photos are copied into it; the gallery figures of APTOS photos stay team-only)."),
    code("import shutil\n"
         "shutil.make_archive('roshni_v2_outputs', 'zip', '.', 'results_v2')\n"
         "print(os.path.getsize('roshni_v2_outputs.zip') / 1e6, 'MB')"),
]
nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nb.metadata["kaggle"] = {"accelerator": "gpu", "isInternetEnabled": True}
nbf.write(nb, "Roshni_v2_OIA-DDR.ipynb")
print("Wrote Roshni_v2_OIA-DDR.ipynb with", len(cells), "cells")
