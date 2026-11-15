"""Two photos per eye: does referring when EITHER photo of an eye is suspicious recover missed referrals?

Added AFTER we saw the external result, like local_calibration.py. Nothing is re-trained or re-tuned: the frozen
model, the frozen referral cut-off and the frozen quality check are used exactly as they are. DeepDRiD takes two
photos of every eye (file names like 12_l1 and 12_l2), as the commercial systems do (LumineticsCore and EyeArt
both ask for two photos per eye), so the saved per-photo predictions are enough to answer it. No GPU, no photos.

Rules (chosen to keep Roshni's safety rule: a poor photo never blocks a referral):
  * referral:  an eye is referred if either of its two photos scores at or above the frozen cut-off;
  * workflow:  REFER if either photo says REFER; otherwise RETAKE if either says RETAKE; otherwise RESCREEN.
  * truth:     an eye is referable if its expert grade is 2 or worse (the higher grade if its two photos differ).

One patient (4 photos) whose file names do not pair into two photos per eye is left out of both columns.
Units change: results are per EYE (two photos), not per photo, so they are not directly comparable with the
per-photo results; both are reported. Confidence intervals resample whole patients (each has two eyes).

Writes results/two_photo.json.
Run:  python two_photo.py [--results results]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import drscreen as ds


def per_eye(d, ref_thr, poor_thr):
    d = d.copy()
    d["eye"] = d.image_id.str.extract(r"_([lr])\d")[0]
    d["dec"] = ds.decide(d.p_referable.values, d.p_poor.values, ref_thr, poor_thr)
    rank = {"RESCREEN": 0, "RETAKE": 1, "REFER": 2}
    d["rank"] = d.dec.map(rank)
    e = d.groupby(["patient_id", "eye"]).agg(grade=("grade", "max"), p_max=("p_referable", "max"),
                                             rank=("rank", "max"), photos=("image_id", "count")).reset_index()
    e["dec"] = e["rank"].map({v: k for k, v in rank.items()})
    return d, e


def binary(y, ref):
    y, ref = np.asarray(y, bool), np.asarray(ref, bool)
    tp, fn, tn, fp = int((ref & y).sum()), int((~ref & y).sum()), int((~ref & ~y).sum()), int((ref & ~y).sum())
    return dict(n=int(len(y)), positives=int(y.sum()), sensitivity=tp / (tp + fn), specificity=tn / (tn + fp),
                missed=fn, false_alarms=fp)


def patient_ci(e, ref_thr, n=2000, seed=0):
    """Percentile intervals for eye-level sensitivity and specificity, resampling whole patients."""
    rng = np.random.default_rng(seed)
    pats = e.patient_id.unique()
    groups = {p: g for p, g in e.groupby("patient_id")}
    sens, spec = [], []
    for _ in range(n):
        s = pd.concat([groups[p] for p in rng.choice(pats, len(pats), replace=True)])
        r = binary(s.grade >= 2, s.p_max >= ref_thr)
        sens.append(r["sensitivity"]); spec.append(r["specificity"])
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return q(sens), q(spec)


def workflow(units, sev_col="grade"):
    y, sev = (units[sev_col] >= 2).values, (units[sev_col] >= 3).values
    dec = units.dec.values
    return dict(REFER=int((dec == "REFER").sum()), RETAKE=int((dec == "RETAKE").sum()),
                RESCREEN=int((dec == "RESCREEN").sum()),
                referable_given_all_clear=int(((dec == "RESCREEN") & y).sum()),
                sight_threatening_given_all_clear=int(((dec == "RESCREEN") & sev).sum()))


def run(results_dir="results"):
    res = json.load(open(os.path.join(results_dir, "results.json")))
    fr = res["frozen"]
    ref_thr, poor_thr = fr["ref_threshold"], fr["poor_threshold"]
    d = pd.read_csv(os.path.join(results_dir, "deepdrid_predictions.csv"))
    photos, eyes = per_eye(d, ref_thr, poor_thr)
    odd = eyes[eyes.photos != 2]             # one DeepDRiD patient's file names do not split into 2 + 2 photos
    skipped = sorted(set(odd.patient_id))
    eyes = eyes[~eyes.patient_id.isin(skipped)]
    photos = photos[~photos.patient_id.isin(skipped)]
    out = dict(rule="refer if either photo of the eye scores at or above the frozen cut-off; "
                    "workflow takes the most cautious of the two decisions",
               post_hoc=True, ref_threshold=ref_thr, poor_threshold=poor_thr,
               grades_differ_within_eye=int((photos.groupby(["patient_id", "eye"]).grade.nunique() > 1).sum()),
               patients_skipped=[int(x) for x in skipped], photos_skipped=int(len(d) - len(photos)))
    for name, part in (("all", None), ("validation", "validation")):
        p = photos if part is None else photos[photos.part == part]
        e = eyes[eyes.patient_id.isin(p.patient_id.unique())]
        one = binary(p.grade >= 2, p.p_referable >= ref_thr)
        two = binary(e.grade >= 2, e.p_max >= ref_thr)
        two["sensitivity_ci"], two["specificity_ci"] = patient_ci(e, ref_thr)
        out[name] = dict(per_photo=one, per_eye_two_photos=two, patients=int(e.patient_id.nunique()))
        if part == "validation":
            out[name]["workflow_per_photo"] = workflow(p)
            out[name]["workflow_per_eye_two_photos"] = workflow(e)
    with open(os.path.join(results_dir, "two_photo.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("--results", default="results")
    o = run(a.parse_args().results)
    for k in ("all", "validation"):
        one, two = o[k]["per_photo"], o[k]["per_eye_two_photos"]
        print(f"{k:10s} one photo: sens {100*one['sensitivity']:.1f}% spec {100*one['specificity']:.1f}% (n={one['n']} photos) | "
              f"two photos per eye: sens {100*two['sensitivity']:.1f}% "
              f"({100*two['sensitivity_ci'][0]:.1f} to {100*two['sensitivity_ci'][1]:.1f}) "
              f"spec {100*two['specificity']:.1f}% (n={two['n']} eyes)")
    print("validation workflow, per photo:", o["validation"]["workflow_per_photo"])
    print("validation workflow, per eye:  ", o["validation"]["workflow_per_eye_two_photos"])
