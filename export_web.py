"""Export the trained, frozen Roshni tool for the in-browser demo (web/).

  python export_web.py --results results --out web

Writes:
  web/model.onnx        the fine-tuned EfficientNet-B0: outputs grade logits, image features and the last feature map
  web/roshni_web.json   frozen thresholds, the photo-quality check, the classifier weights (for the heat map),
                        and the note texts, all copied from the run's own files
Then checks that ONNX Runtime gives the same answers as PyTorch on a few photos.
The demo page (web/index.html) runs everything in the browser: the photo never leaves the device.
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch

import drscreen as ds


class _Export(torch.nn.Module):
    """Same network, returning (logits, pooled features, last feature map) in one forward pass."""

    def __init__(self, m):
        super().__init__()
        self.features, self.avgpool, self.classifier = m.features, m.avgpool, m.classifier

    def forward(self, x):
        fmap = self.features(x)
        emb = torch.flatten(self.avgpool(fmap), 1)
        return self.classifier(emb), emb, fmap


def _examples(results_dir, out_dir, n_each=1):
    """Copy a few DeepDRiD validation photos (CC BY-SA 4.0) for the demo, with the experts' grade and quality label.
    One photo per rule the demo shows, each from a different patient, chosen well away from both frozen cut-offs so
    small numeric differences between browsers cannot flip the answer. The last one is a referable eye the tool misses,
    shown on purpose. APTOS photos are not copied: the competition's rules do not allow redistributing its images."""
    import shutil
    import pandas as pd
    p = os.path.join(results_dir, "deepdrid_predictions.csv")
    rj = os.path.join(results_dir, "results.json")
    if not os.path.exists(p) or not os.path.exists(rj):
        return []
    fr = json.load(open(rj))["frozen"]
    rt, pt = fr["ref_threshold"], fr["poor_threshold"]
    dd = pd.read_csv(p)
    v = dd[(dd.part == "validation") & (dd.grade <= 4)].copy()
    v["m_ref"], v["m_poor"] = (v.p_referable - rt).abs(), (v.p_poor - pt).abs()
    safe = v[(v.m_ref > 0.15) & (v.m_poor > 0.15)]
    picks = [  # (rows, sort key, note)
        (safe[(safe.decision == "REFER") & (safe.grade >= 3) & (safe.poor_quality == 0)], "p_referable", False, ""),
        (safe[(safe.decision == "REFER") & (safe.grade >= 2) & (safe.p_poor > pt)], "p_poor", False, ""),  # check says poor, still refers
        (safe[(safe.decision == "RETAKE") & (safe.poor_quality == 1) & (safe.grade <= 1)], "p_poor", False, ""),
        (safe[(safe.decision == "RESCREEN") & (safe.grade == 0) & (safe.poor_quality == 0)], "p_poor", True, ""),
        (safe[(safe.decision == "RESCREEN") & (safe.grade == 2) & (safe.poor_quality == 0)], "p_referable", True,
         ". Shown on purpose: one of the referable photos Roshni misses"),
    ]
    os.makedirs(os.path.join(out_dir, "examples"), exist_ok=True)
    out, patients = [], set()
    for rows, key, asc, note in picks:
        for r in rows.sort_values(key, ascending=asc).itertuples():
            if r.patient_id in patients or not os.path.exists(r.cached):
                continue
            patients.add(r.patient_id)
            name = f"examples/{r.image_id}.jpg"
            shutil.copy(r.cached, os.path.join(out_dir, name))
            out.append(dict(file=name, caption=f"Expert grade {r.grade} ({ds.GRADES[r.grade]}), "
                                                 f"photo quality {'poor' if r.poor_quality else 'good'}{note}"))
            break
    return out


def export(results_dir="results", out_dir="web", check_paths=(), repo_url=None):
    with open(os.path.join(results_dir, "results.json")) as f:
        res = json.load(f)
    fr = res["frozen"]
    if fr["arch"] != "efficientnet_b0":
        raise ValueError("web export supports efficientnet_b0 only")
    model = ds.build_model(fr["arch"], pretrained=False)
    model.load_state_dict(torch.load(os.path.join(results_dir, "model_best.pt"), map_location="cpu"))
    model.eval()
    net = _Export(model).eval()
    os.makedirs(out_dir, exist_ok=True)
    size = int(fr["size"])
    dummy = torch.zeros(1, 3, size, size)
    kw = dict(input_names=["image"], output_names=["logits", "features", "feature_map"], opset_version=17)
    try:
        torch.onnx.export(net, dummy, os.path.join(out_dir, "model.onnx"), dynamo=False, **kw)
    except TypeError:  # older torch without the dynamo argument
        torch.onnx.export(net, dummy, os.path.join(out_dir, "model.onnx"), **kw)

    gate = None
    gp = os.path.join(results_dir, "quality_gate.pkl")
    if os.path.exists(gp):
        with open(gp, "rb") as f:
            pipe = pickle.load(f)
        sc, lr = pipe.steps[0][1], pipe.steps[-1][1]
        gate = dict(mean=sc.mean_.tolist(), scale=sc.scale_.tolist(), coef=lr.coef_[0].tolist(),
                    intercept=float(lr.intercept_[0]))
    lin = model.classifier[1]
    cfg = dict(
        size=size, mean=list(ds.MEAN), std=list(ds.STD), crop_tol=10, referable_from=ds.REFERABLE_FROM,
        ref_threshold=float(fr["ref_threshold"]), poor_threshold=float(fr["poor_threshold"]),
        target_sensitivity=float(fr["target_sensitivity"]), grades=ds.GRADES, gate=gate,
        head_weight=lin.weight.detach().numpy().round(6).tolist(), head_bias=lin.bias.detach().numpy().round(6).tolist(),
        langs={k: dict(name=ds.LANGS[k]["name"], dir=ds.LANGS[k]["dir"], checked=ds.LANGS[k]["checked"]) for k in ds.shown_langs()},
        default_langs=list(ds.DEFAULT_LANGS),
        report={d: {k: t for k, t in v.items() if k in ds.shown_langs()} for d, v in ds.REPORT.items()}, summary=_summary(res), repo_url=repo_url,
        examples=_examples(results_dir, out_dir),
        examples_credit="Example photos: DeepDRiD (Liu et al., Patterns 2022), CC BY-SA 4.0. "
                        "Captions give the experts' grade and photo-quality label, not the tool's answer.",
    )
    with open(os.path.join(out_dir, "roshni_web.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False)
    _stamp_offline_cache(out_dir)
    return _check(model, pipe if gate else None, fr, out_dir, check_paths)


def _stamp_offline_cache(out_dir):
    """Name the offline cache after every file of the demo, so any change (model, config, page, examples) replaces the old cache."""
    import hashlib
    import re
    sw = os.path.join(out_dir, "sw.js")
    if not os.path.exists(sw):
        return
    h = hashlib.sha256()
    for d, _, files in sorted(os.walk(out_dir)):
        for f in sorted(files):
            p = os.path.join(d, f)
            if os.path.abspath(p) == os.path.abspath(sw):
                continue
            h.update(os.path.relpath(p, out_dir).encode())
            with open(p, "rb") as fh:
                h.update(fh.read())
    text = open(sw, encoding="utf-8").read()
    text = re.sub(r'const VERSION = "[^"]*";', f'const VERSION = "roshni-{h.hexdigest()[:12]}";', text)
    open(sw, "w", encoding="utf-8").write(text)


def _summary(res):
    """A few headline numbers for the page footer, copied from results.json (nothing typed by hand)."""
    r = res.get("results", {})
    out = {}
    for k in ("fine_tuned", "external_all"):
        if k in r:
            out[k] = dict(n=r[k]["n"], AUC=r[k]["AUC"], sensitivity=r[k]["sensitivity"], specificity=r[k]["specificity"])
    return out


def _check(model, gate, fr, out_dir, paths):
    """PyTorch and ONNX Runtime must agree on the same preprocessed photos."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed: skipped the agreement check")
        return None
    sess = ort.InferenceSession(os.path.join(out_dir, "model.onnx"), providers=["CPUExecutionProvider"])
    tf = ds.get_transforms(fr["size"], False)
    worst = 0.0
    for p in list(paths)[:5]:
        x = tf(ds.prep_image(p, fr["size"])).unsqueeze(0)
        d, p_ref, p_poor, P, e = ds.triage_one(model, gate, p, fr["size"], fr["ref_threshold"], fr["poor_threshold"], tf)
        logits, emb, _ = sess.run(None, {"image": x.numpy()})
        Po = np.exp(logits - logits.max(1, keepdims=True))
        Po /= Po.sum(1, keepdims=True)
        worst = max(worst, float(np.abs(Po[0] - P).max()), float(np.abs(emb - e).max()))
    print(f"ONNX vs PyTorch, largest difference on {min(5, len(list(paths)))} photos: {worst:.2e}")
    if worst > 1e-3:
        raise RuntimeError("ONNX export does not match PyTorch")
    return worst


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--results", default="results")
    a.add_argument("--out", default="web")
    a.add_argument("--check", nargs="*", default=[])
    a.add_argument("--repo-url", default=None)
    x = a.parse_args()
    export(x.results, x.out, x.check, x.repo_url)
