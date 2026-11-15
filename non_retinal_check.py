"""What does Roshni do with a picture that is not a retinal photo at all?

Added after the main results (it changes nothing in them). A judge, a nurse or a curious user may give the tool a
selfie, a document or a blank image. Roshni has no separate "is this a retina?" detector, so this checks the
safety question directly: does any non-retinal picture get an all-clear (RESCREEN)? Each picture is screened by
the frozen tool exactly as the clinic app does (drscreen.triage_one; the browser demo gives the same decisions).

Pictures (no photo of a person is used):
  * made here: plain white, grey and black, random noise, a flat skin-coloured image, a colour gradient, a page of text;
  * from this repository: a chart (results/roc.png);
  * if scikit-image is installed, its bundled sample photos: a cat, a coffee cup, a rocket, a deep-space image and a
    stained tissue slide (loaded at run time, not redistributed).

Writes results/non_retinal.json.
Run:  python non_retinal_check.py [--results results]
"""
import argparse
import json
import os
import pickle
import warnings

import numpy as np
from PIL import Image

import drscreen as ds


def pictures(results_dir):
    rng = np.random.default_rng(0)
    yield "plain white", Image.new("RGB", (800, 600), (255, 255, 255))
    yield "plain grey", Image.new("RGB", (800, 600), (128, 128, 128))
    yield "plain black", Image.new("RGB", (800, 600), (0, 0, 0))
    yield "random noise", Image.fromarray(rng.integers(0, 256, (600, 800, 3), dtype=np.uint8))
    yield "flat skin colour", Image.new("RGB", (800, 600), (200, 120, 90))
    g = np.linspace(0, 255, 800, dtype=np.float32)
    yield "colour gradient", Image.fromarray(np.stack([np.tile(g, (600, 1)), np.tile(g[::-1], (600, 1)),
                                                       np.full((600, 800), 128.0)], -1).astype(np.uint8))
    from PIL import ImageDraw
    page = Image.new("RGB", (800, 1000), (255, 255, 255))
    dr = ImageDraw.Draw(page)
    for i in range(30):
        dr.text((60, 60 + 30 * i), "Patient notes: blood sugar, blood pressure, next visit " + str(i), fill=(20, 20, 20))
    yield "page of text", page
    p = os.path.join(results_dir, "roc.png")
    if os.path.exists(p):
        yield "chart (ROC curve)", Image.open(p).convert("RGB")
    try:
        import skimage.data as sd
        for name, fn in (("cat photo", "chelsea"), ("coffee cup photo", "coffee"), ("rocket photo", "rocket"),
                         ("deep-space image", "hubble_deep_field"), ("stained tissue slide", "immunohistochemistry")):
            try:
                a = getattr(sd, fn)()
                yield name, Image.fromarray(a[..., :3].astype(np.uint8))
            except Exception:       # a sample that needs downloading: skip it
                pass
    except ImportError:
        pass


def run(results_dir="results"):
    import torch
    res = json.load(open(os.path.join(results_dir, "results.json")))
    fr = res["frozen"]
    model = ds.build_model(fr["arch"], pretrained=False)
    model.load_state_dict(torch.load(os.path.join(results_dir, "model_best.pt"), map_location="cpu"))
    model.eval()
    with open(os.path.join(results_dir, "quality_gate.pkl"), "rb") as f, warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gate = pickle.load(f)
    rows = []
    for name, img in pictures(results_dir):
        d, p_ref, p_poor, _, _ = ds.triage_one(model, gate, img, fr["size"], fr["ref_threshold"], fr["poor_threshold"])
        rows.append(dict(picture=name, decision=d, referable_score=round(p_ref, 3), poor_photo_score=round(p_poor, 3)))
    counts = {k: sum(r["decision"] == k for r in rows) for k in ("REFER", "RETAKE", "RESCREEN")}
    out = dict(pictures=len(rows), counts=counts, all_clears=counts["RESCREEN"], rows=rows)
    with open(os.path.join(results_dir, "non_retinal.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("--results", default="results")
    o = run(a.parse_args().results)
    for r in o["rows"]:
        print(f"{r['picture']:24s} {r['decision']:9s} referable {r['referable_score']:.2f}  poor photo {r['poor_photo_score']:.2f}")
    print(o["counts"])
