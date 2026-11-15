"""Local re-calibration: does re-setting the referral cut-off on a few labelled photos from a new clinic fix the
external drop?

Added AFTER we saw the external result (sensitivity fell to about 56% at the frozen cut-off), to test our own
advice that "each clinic must re-set the cut-off on its own graded photos". The model, the quality check and the
frozen results are untouched and stay the main results. Uses only the saved per-photo predictions, so it needs
no GPU and no photos:

  * calibration photos: DeepDRiD *training* patients (the grading model never saw any DeepDRiD photo)
  * evaluation photos:  the 400 DeepDRiD *validation* photos, from 100 other patients
  * a "clinic sample" is drawn by patient (all of a patient's photos together), the cut-off is set for 90%
    sensitivity on it with the same function used to freeze the original cut-off, and the full workflow
    (refer / retake / rescreen, with the quality check) is scored on the validation photos. Repeated many times.

Writes results/local_calibration.json and results/local_calibration.png.
Run:  python local_calibration.py [--results results] [--draws 500]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

import drscreen as ds

SAMPLE_PATIENTS = (12, 25, 50, 100)      # about 4 photos per patient: 48, 100, 200, 400 photos


def workflow(d, ref_thr, poor_thr):
    y, sev = (d.grade >= 2).values, (d.grade >= 3).values
    ref = d.p_referable.values >= ref_thr
    dec = np.asarray(ds.decide(d.p_referable.values, d.p_poor.values, ref_thr, poor_thr))
    return dict(threshold=float(ref_thr), sensitivity=float((ref & y).sum() / y.sum()),
                specificity=float((~ref & ~y).sum() / (~y).sum()),
                REFER=int((dec == "REFER").sum()), RETAKE=int((dec == "RETAKE").sum()), RESCREEN=int((dec == "RESCREEN").sum()),
                referable_given_all_clear=int(((dec == "RESCREEN") & y).sum()),
                sight_threatening_given_all_clear=int(((dec == "RESCREEN") & sev).sum()))


def run(results_dir="results", draws=500, seed=0):
    res = json.load(open(os.path.join(results_dir, "results.json")))
    fr = res["frozen"]
    dd = pd.read_csv(os.path.join(results_dir, "deepdrid_predictions.csv"))
    dd = dd[dd.grade <= 4]
    cal, val = dd[dd.part == "training"], dd[dd.part == "validation"]
    assert not set(cal.patient_id) & set(val.patient_id), "calibration and evaluation patients must differ"
    out = dict(note="Added after seeing the external result; the frozen results stay the main results.",
               target_sensitivity=fr["target_sensitivity"], evaluation_photos=len(val),
               evaluation_patients=int(val.patient_id.nunique()), calibration_pool_patients=int(cal.patient_id.nunique()),
               draws=draws, frozen=workflow(val, fr["ref_threshold"], fr["poor_threshold"]), samples=[])
    rng = np.random.default_rng(seed)
    pats = cal.patient_id.unique()
    for n in SAMPLE_PATIENTS:
        if n > len(pats):                            # cannot draw more patients than the pool has
            continue
        rows, tries = [], 0
        while len(rows) < draws and tries < draws * 5:
            tries += 1
            sub = cal[cal.patient_id.isin(rng.choice(pats, n, replace=False))]
            if (sub.grade >= 2).sum() < 3:          # a clinic sample with almost no disease cannot set a cut-off
                continue
            t = ds.threshold_for_sensitivity((sub.grade >= 2).astype(int).values, sub.p_referable.values, fr["target_sensitivity"])
            rows.append(workflow(val, t, fr["poor_threshold"]))
        r = pd.DataFrame(rows)
        q = lambda c: {k: float(v) for k, v in zip(("p10", "median", "p90"), r[c].quantile([0.1, 0.5, 0.9]))}
        out["samples"].append(dict(patients=n, photos_median=int(round(cal.groupby("patient_id").size().median() * n)),
                                   draws=len(r), skipped_too_little_disease=tries - len(r),
                                   sensitivity=q("sensitivity"), specificity=q("specificity"), threshold=q("threshold"),
                                   referable_given_all_clear=q("referable_given_all_clear"),
                                   sight_threatening_given_all_clear=q("sight_threatening_given_all_clear"),
                                   REFER=q("REFER"), RETAKE=q("RETAKE"), RESCREEN=q("RESCREEN")))
    out["all_calibration_photos"] = workflow(
        val, ds.threshold_for_sensitivity((cal.grade >= 2).astype(int).values, cal.p_referable.values, fr["target_sensitivity"]),
        fr["poor_threshold"])
    json.dump(out, open(os.path.join(results_dir, "local_calibration.json"), "w"), indent=2)
    figure(out, os.path.join(results_dir, "local_calibration.png"))
    return out


def figure(out, path):
    from matplotlib.figure import Figure          # no pyplot: leaves the caller's backend (e.g. a notebook) alone
    s = out["samples"]
    x = [f"{d['patients']} patients\n(~{d['photos_median']} photos)" for d in s]
    fig = Figure(figsize=(11, 4.2), dpi=150)
    axes = fig.subplots(1, 2)
    for ax, key, title in ((axes[0], "sensitivity", "Sensitivity on 100 other patients"),
                           (axes[1], "specificity", "Specificity on 100 other patients")):
        med = [100 * d[key]["median"] for d in s]
        lo = [100 * (d[key]["median"] - d[key]["p10"]) for d in s]
        hi = [100 * (d[key]["p90"] - d[key]["median"]) for d in s]
        ax.errorbar(range(len(s)), med, yerr=[lo, hi], fmt="o", color="#2f6fdb", ms=8, capsize=5, lw=2,
                    label="cut-off re-set on a local sample\n(median, 10th to 90th percentile)")
        ax.axhline(100 * out["frozen"][key], color="#6e7781", ls="--", lw=1.5, label="frozen cut-off (set on Indian photos)")
        ax.set_xticks(range(len(s)), x, fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_ylabel("%", color="#4b5563")
        ax.set_title(title, loc="left", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e1e4e8")
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("DeepDRiD: re-setting the referral cut-off on labelled photos from the new hospitals "
                 "(added after seeing the external result)", fontsize=10, x=0.01, ha="left", color="#4b5563")
    fig.tight_layout()
    fig.savefig(path)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--results", default="results")
    a.add_argument("--draws", type=int, default=500)
    x = a.parse_args()
    o = run(x.results, x.draws)
    f = o["frozen"]
    print(f"frozen cut-off: sensitivity {f['sensitivity']:.1%}, specificity {f['specificity']:.1%}, "
          f"referable all-clears {f['referable_given_all_clear']}")
    for d in o["samples"]:
        print(f"{d['patients']:>3} patients: sensitivity {d['sensitivity']['median']:.1%} "
              f"({d['sensitivity']['p10']:.1%} to {d['sensitivity']['p90']:.1%}), specificity {d['specificity']['median']:.1%} "
              f"({d['specificity']['p10']:.1%} to {d['specificity']['p90']:.1%}), referable all-clears "
              f"{d['referable_given_all_clear']['median']:.0f}")
