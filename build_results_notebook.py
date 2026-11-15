"""Builds Roshni_Results.ipynb: a CPU-only walkthrough of the finished run, using only files in this repository.

It recomputes every headline number from the per-photo predictions, re-runs the trained model on the bundled demo
photos (with heat maps and the clinic note), checks the browser model against PyTorch, and (optionally, if the
DeepDRiD photos are available) re-scores all 400 external validation photos and compares them with the saved run.
No APTOS photo is displayed: the competition's rules do not allow sharing its images.
Run:  python build_results_notebook.py   then execute it (Colab, Kaggle or Jupyter; no GPU needed)."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("# Roshni: results walkthrough (runs on a CPU in a few minutes)\n\n"
       "This notebook checks the finished run using only the files in this repository: the per-photo predictions, "
       "the trained model (`results/model_best.pt`), the photo-quality check (`results/quality_gate.pkl`) and the "
       "browser model (`web/model.onnx`). The training itself is in `Roshni_DR_Screening.ipynb` (needs a GPU, run on Kaggle).\n\n"
       "1. Recompute every headline number from the per-photo predictions, independently of the code that wrote the README\n"
       "2. The photo-quality check: false all-clears with and without it\n"
       "3. The stress test on deliberately worse photos\n"
       "4. Run the trained model live on the five demo photos, with heat maps and the clinic note\n"
       "5. Check that the browser model gives the same numbers as PyTorch\n"
       "6. Optional: re-score all 400 external validation photos and compare with the saved run\n\n"
       "No APTOS photo is shown here: the competition's rules do not allow sharing its images. "
       "The photos shown are from DeepDRiD (CC BY-SA 4.0).\n\n"
       "> This project is a prototype and is not medical advice."),
    md("## 0. Setup\nIn Colab or Kaggle, this cell downloads the repository first."),
    code("import os, sys, json, pickle, time, subprocess\n"
         "if not os.path.exists('results/results.json'):\n"
         "    subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/MACVXACE/roshni.git'], check=True)\n"
         "    os.chdir('roshni')\n"
         "sys.path.insert(0, os.getcwd())\n"
         "import numpy as np, pandas as pd, torch, sklearn, PIL\n"
         "import matplotlib.pyplot as plt\n"
         "from IPython.display import Image as Show, HTML, display\n"
         "from sklearn.metrics import roc_auc_score, cohen_kappa_score\n"
         "import drscreen as ds\n"
         "pd.set_option('display.width', 160)\n"
         "res = json.load(open('results/results.json'))\n"
         "fr = res['frozen']; THR, PT, SIZE = fr['ref_threshold'], fr['poor_threshold'], fr['size']\n"
         "print('torch', torch.__version__, '| scikit-learn', sklearn.__version__, '| Pillow', PIL.__version__)\n"
         "print(f\"Frozen before testing: model {fr['arch']} at {SIZE} px, referral threshold {THR:.3f} \"\n"
         "      f\"(set for {fr['target_sensitivity']:.0%} sensitivity on validation), quality cut-off {PT}\")"),
    md("### Training history\nThe model kept is the epoch with the best validation AUC (validation photos only; the test set was not used)."),
    code("h = pd.read_csv('results/training_history.csv')\n"
         "print(h.round(4).to_string(index=False))\n"
         "print('Best validation AUC at epoch', int(h.loc[h.val_auc.idxmax(), 'epoch']))"),
    md("## 1. Headline numbers, recomputed from the per-photo predictions\n"
       "`results/aptos_test_predictions.csv` has one row per held-out APTOS test photo, and `results/deepdrid_predictions.csv` "
       "one row per external DeepDRiD photo. A photo is **referable** at grade 2 or worse; Roshni says REFER when the "
       "referable score is at or above the frozen threshold."),
    code("ap = pd.read_csv('results/aptos_test_predictions.csv')\n"
         "dd = pd.read_csv('results/deepdrid_predictions.csv'); dd = dd[dd.grade <= 4]\n"
         "def metrics(name, y, p):\n"
         "    y = np.asarray(y, bool); pred = np.asarray(p) >= THR\n"
         "    return dict(set=name, photos=len(y), referable=int(y.sum()), AUC=round(roc_auc_score(y, p), 3),\n"
         "                sensitivity=f'{(pred & y).sum() / y.sum():.1%}', specificity=f'{(~pred & ~y).sum() / (~y).sum():.1%}',\n"
         "                missed=int((~pred & y).sum()), false_alarms=int((pred & ~y).sum()))\n"
         "rows = [metrics('APTOS test (India, held out)', ap.grade >= 2, ap.p_referable),\n"
         "        metrics('DeepDRiD (China, external)', dd.grade >= 2, dd.p_referable),\n"
         "        metrics('  external, good photos', dd[dd.poor_quality == 0].grade >= 2, dd[dd.poor_quality == 0].p_referable),\n"
         "        metrics('  external, poor photos', dd[dd.poor_quality == 1].grade >= 2, dd[dd.poor_quality == 1].p_referable)]\n"
         "display(pd.DataFrame(rows))\n"
         "print('Saved in results.json: APTOS AUC', round(res['results']['fine_tuned']['AUC'], 3),\n"
         "      '| external AUC', round(res['results']['external_all']['AUC'], 3))"),
    code("print('Baselines on the same APTOS test photos, thresholded the same way:')\n"
         "display(pd.read_csv('results/summary.csv').round(3))\n"
         "display(Show('results/roc.png', width=520))"),
    md("### By grade (APTOS test)"),
    code("by = pd.DataFrame({'grade': [f'{g} {ds.GRADES[g]}' for g in range(5)],\n"
         "                   'photos': [int((ap.grade == g).sum()) for g in range(5)],\n"
         "                   'sent to specialist': [int(((ap.grade == g) & (ap.p_referable >= THR)).sum()) for g in range(5)]})\n"
         "display(by)\n"
         "print('Five-grade agreement (quadratic weighted kappa):', round(cohen_kappa_score(ap.grade, ap.pred_grade, weights='quadratic'), 3))\n"
         "print('Confusion matrix (rows: expert grade, columns: predicted grade):')\n"
         "print(pd.crosstab(ap.grade, ap.pred_grade))"),
    md("### External results by hospital source\nDeepDRiD records no age or sex, so we compare by source and photo quality."),
    code("src = dd.assign(ref=dd.grade >= 2, pred=dd.p_referable >= THR).groupby('Source').apply(\n"
         "    lambda g: pd.Series({'photos': len(g), 'sensitivity': f'{(g.pred & g.ref).sum() / g.ref.sum():.1%}',\n"
         "                         'specificity': f'{(~g.pred & ~g.ref).sum() / (~g.ref).sum():.1%}'}))\n"
         "display(src)\n"
         "print('External AUC 95% interval, resampling whole patients:', np.round(res['results']['external_all']['AUC_CI_by_patient'], 3))"),
    md("### Leakage check\nNear-duplicate photos (image hashing) were grouped, and each group kept on one side of the split."),
    code("sp = pd.read_csv('results/aptos_split.csv')\n"
         "print(sp.split.value_counts().to_string())\n"
         "print('Largest number of splits any near-duplicate group appears in:', sp.groupby('dup_group').split.nunique().max(), '(1 = no leakage)')\n"
         "print('Test photos in the split file = test photos scored:', set(sp[sp.split == 'test'].id_code) == set(ap.id_code))"),
    md("## 2. Our own question: does the photo-quality check stop false all-clears?\n"
       "The 400 DeepDRiD **validation** photos were never used to train either the grading model or the quality check. "
       "The rule: REFER if the referable score is at or above the threshold; otherwise RETAKE if the quality check says poor; "
       "otherwise RESCREEN in 12 months (the all-clear)."),
    code("v = pd.read_csv('results/deepdrid_predictions.csv'); v = v[(v.part == 'validation') & (v.grade <= 4)]\n"
         "def run(gate):\n"
         "    d = np.where(v.p_referable >= THR, 'REFER', np.where(gate & (v.p_poor >= PT), 'RETAKE', 'RESCREEN'))\n"
         "    ref, sev = (v.grade >= 2).values, (v.grade >= 3).values\n"
         "    return {'REFER': (d == 'REFER').sum(), 'RETAKE': (d == 'RETAKE').sum(), 'RESCREEN (all-clear)': (d == 'RESCREEN').sum(),\n"
         "            'referable eyes given an all-clear': ((d == 'RESCREEN') & ref).sum(),\n"
         "            'grade 3-4 eyes given an all-clear': ((d == 'RESCREEN') & sev).sum()}\n"
         "display(pd.DataFrame({'without quality check': run(False), 'with quality check': run(True)}))\n"
         "print('Quality check finds poor photos with AUC', round(roc_auc_score(v.poor_quality, v.p_poor), 3))"),
    md("## 3. Stress test: photos made worse in four fixed ways, at two strengths\n"
       "Planned before any result: blur, lower resolution, under-exposure and messaging-app compression. Nothing was re-tuned."),
    code("st = pd.read_csv('results/stress_test.csv')\n"
         "cols = ['photos', 'change', 'strength', 'sensitivity', 'specificity', 'retake_share',\n"
         "        'referable_all_clear_without_check', 'referable_all_clear_with_check']\n"
         "display(st[cols].round(3))\n"
         "display(Show('results/stress_test.png', width=900))\n"
         "display(Show('results/stress_examples.png', width=900))"),
    md("## 4. The trained model, live, on the five demo photos\n"
       "These DeepDRiD validation photos are bundled with the browser demo (`web/examples`). Each goes through exactly the "
       "clinic path: open, crop, resize, score, quality check, decide. The saved scores from the Kaggle run are shown for "
       "comparison; small differences come from GPU vs CPU arithmetic."),
    code("torch.set_num_threads(max(1, os.cpu_count() or 1))\n"
         "model = ds.build_model(fr['arch'], pretrained=False)\n"
         "model.load_state_dict(torch.load('results/model_best.pt', map_location='cpu')); model.eval()\n"
         "import warnings\n"
         "with warnings.catch_warnings(record=True) as caught:\n"
         "    warnings.simplefilter('always')\n"
         "    gate = pickle.load(open('results/quality_gate.pkl', 'rb'))\n"
         "vers = {(w.message.original_sklearn_version, w.message.current_sklearn_version) for w in caught if hasattr(w.message, 'original_sklearn_version')}\n"
         "for old, now in vers:\n"
         "        print(f'Note: the quality check was saved with scikit-learn {old} on Kaggle and is loaded here '\n"
         "              f'with {now}; section 6 shows it still reproduces the saved scores.')\n"
         "cfg = json.load(open('web/roshni_web.json'))\n"
         "saved = pd.read_csv('results/deepdrid_predictions.csv').set_index('image_id')\n"
         "rows, times = [], []\n"
         "for ex in cfg['examples']:\n"
         "    f = os.path.join('web', ex['file']); iid = os.path.basename(f)[:-4]\n"
         "    t = time.perf_counter(); dec, pr, pp, P, _ = ds.triage_one(model, gate, f, SIZE, THR, PT); times.append(time.perf_counter() - t)\n"
         "    rows.append(dict(photo=iid, experts=ex['caption'], decision=dec, referable_score=round(pr, 3), saved_score=round(saved.loc[iid, 'p_referable'], 3),\n"
         "                     poor_photo_score=round(pp, 3), saved_poor=round(saved.loc[iid, 'p_poor'], 3), saved_decision=saved.loc[iid, 'decision']))\n"
         "live = pd.DataFrame(rows); display(live)\n"
         "print('Same decision as the saved run:', (live.decision == live.saved_decision).sum(), 'of', len(live))\n"
         "print(f'CPU time per photo here: median {np.median(times):.2f} s')"),
    code("from PIL import Image as PILImage\n"
         "fig, axes = plt.subplots(2, len(cfg['examples']), figsize=(3.3 * len(cfg['examples']), 7.4))\n"
         "for i, (ex, r) in enumerate(zip(cfg['examples'], rows)):\n"
         "    f = os.path.join('web', ex['file']); base = ds.prep_image(f, SIZE)\n"
         "    cam = np.asarray(PILImage.fromarray(np.uint8(ds.gradcam_map(model, f, SIZE, device='cpu') * 255)).resize((SIZE, SIZE), PILImage.BILINEAR)) / 255\n"
         "    grade = ex['caption'].split(', photo')[0].replace('Expert grade', 'experts: grade')\n"
         "    axes[0, i].imshow(base); axes[0, i].set_title(f\"{r['decision']} (score {r['referable_score']:.2f})\\n{grade}\", fontsize=10)\n"
         "    axes[1, i].imshow(base); axes[1, i].imshow(cam, cmap='inferno', alpha=0.55)\n"
         "    for a in axes[:, i]: a.set_xticks([]); a.set_yticks([])\n"
         "axes[1, 0].set_ylabel('Grad-CAM: where the referable\\nscore came from (bright = more)', fontsize=9)\n"
         "fig.suptitle('Demo photos (DeepDRiD, CC BY-SA 4.0): decision and heat map', fontsize=12); plt.tight_layout(); plt.show()"),
    md("**How to read the heat maps, honestly.** Grad-CAM is computed on the network's last 12 x 12 grid, so it is coarse, "
       "and it is rescaled per photo, so for a low score (the two RESCREEN photos) it only shows where the small score came from "
       "and means little. On several photos it lights up the edge of the field of view rather than lesions. We show it as a hint "
       "for the doctor, not as an explanation, and we have not checked it against lesion annotations."),
    md("### The note the clinic gets (English and Urdu)"),
    code("display(HTML(ds.slip_html(rows[0]['decision'])))\n"
         "print('Languages shown:', [ds.LANGS[c]['name'] + ('' if ds.LANGS[c]['checked'] else ' (translation being checked)') for c in ds.shown_langs()])"),
    md("## 5. The browser model gives the same numbers as PyTorch\n"
       "`web/model.onnx` is what runs in the browser demo. Here it is run with ONNX Runtime on the same inputs."),
    code("try:\n"
         "    import onnxruntime as ort\n"
         "    sess = ort.InferenceSession('web/model.onnx', providers=['CPUExecutionProvider'])\n"
         "    tf = ds.get_transforms(SIZE, False); worst = 0.0\n"
         "    for ex in cfg['examples']:\n"
         "        x = tf(ds.prep_image(os.path.join('web', ex['file']), SIZE)).unsqueeze(0)\n"
         "        with torch.no_grad(): ref = model(x).numpy()\n"
         "        out = sess.run(None, {sess.get_inputs()[0].name: x.numpy()})[0]\n"
         "        worst = max(worst, float(np.abs(out - ref).max()))\n"
         "    print(f'Largest difference in the model outputs, ONNX vs PyTorch, over {len(cfg[\"examples\"])} photos: {worst:.2e}')\n"
         "except ImportError:\n"
         "    print('pip install onnxruntime to run this check')"),
    md("## 6. Optional: re-score all 400 external validation photos\n"
       "Needs the DeepDRiD photos (CC BY-SA 4.0). Set `DDR_ROOT` to a local copy, or set `DOWNLOAD_DEEPDRID=1` to clone "
       "https://github.com/deepdrdoc/DeepDRiD (a large download). About 3 to 5 minutes on a laptop CPU."),
    code("root = os.environ.get('DDR_ROOT') or ('DeepDRiD' if os.path.isdir('DeepDRiD') else None)\n"
         "if root is None and os.environ.get('DOWNLOAD_DEEPDRID') == '1':\n"
         "    subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/deepdrdoc/DeepDRiD.git'], check=True); root = 'DeepDRiD'\n"
         "if root is None:\n"
         "    print('Skipped: DeepDRiD photos not available here.')\n"
         "else:\n"
         "    srcs = [os.path.join(root, p.split('DeepDRiD/', 1)[1]) for p in v.path]\n"
         "    t = time.time()\n"
         "    cached = ds.cache_images(srcs, 'cache/ddr_check', size=SIZE, workers=2)\n"
         "    P, E = ds.predict(model, ds.make_dataset(cached, v.grade.tolist(), SIZE, train=False), batch_size=16, device='cpu', workers=0, with_embeddings=True)\n"
         "    pr, pp = ds.referable_prob(P), gate.predict_proba(E)[:, 1]\n"
         "    dec = np.asarray(ds.decide(pr, pp, THR, PT))\n"
         "    print(f'{len(v)} photos in {time.time() - t:.0f} s')\n"
         "    print('Referable score vs saved run: median difference', f'{np.median(np.abs(pr - v.p_referable.values)):.1e}',\n"
         "          '| largest', f'{np.abs(pr - v.p_referable.values).max():.1e}')\n"
         "    print('Same decision as the saved run:', int((dec == v.decision.values).sum()), 'of', len(v))\n"
         "    diff = v.assign(now=dec, now_ref=pr.round(3), now_poor=pp.round(3))[dec != v.decision.values]\n"
         "    if len(diff): print('Different (each sits next to a cut-off; small numeric differences between the Kaggle GPU run and this CPU decide it):'); display(diff[['image_id', 'decision', 'now', 'p_referable', 'now_ref', 'p_poor', 'now_poor']].round(3))"),
    md("## Where it fails\nReferable eyes given an all-clear on the external photos (lowest scores first). "
       "Our reading of them is in the README's Results section."),
    code("display(Show('results/missed_external.png', width=1000))"),
]

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "Roshni_Results.ipynb")
print("Wrote Roshni_Results.ipynb with", len(cells), "cells")
