"""End-to-end experiment: APTOS 2019 (India) for training, DeepDRiD (China) as the untouched external test.

Used by the notebook; can also run from the command line:
  python run_dr.py --aptos-csv .../train.csv --aptos-dir .../train_images --ddr-root DeepDRiD --out results
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from dataclasses import asdict, dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import drscreen as ds

INK, MUTED, GRID = "#1f2328", "#6a737d", "#e1e4e8"
C_MAIN, C_EXT, C_PROBE, C_COLOR = "#2f6fdb", "#7a3fd1", "#d9822b", "#8c959f"
C_GRAYBAR = "#6e7781"   # the "without" comparison: deliberately neutral


@dataclass
class RunConfig:
    aptos_csv: str
    aptos_dir: str
    ddr_root: str | None = None
    oiaddr_root: str | None = None       # v2 only: OIA-DDR (Li et al. 2019) photos added to training and validation
    out: str = "results"
    cache: str = "cache"
    size: int = 384
    arch: str = "efficientnet_b0"
    pretrained: bool = True
    epochs: int = 12
    lr: float = 3e-4
    batch_size: int = 32
    workers: int = 2
    target_sensitivity: float = 0.90     # fixed before any result was seen
    poor_threshold: float = 0.50         # balanced logistic regression, default cut
    dup_max_dist: int = 10
    seed: int = 0
    max_images: int | None = None        # for quick code checks only


# ------------------------------------------------------------------ data loading
def load_aptos(csv, img_dir):
    df = pd.read_csv(csv)
    ext = ".png"
    if df["id_code"].astype(str).str.contains(r"\.").any():
        ext = ""
    df["path"] = [os.path.join(img_dir, f"{i}{ext}") for i in df["id_code"].astype(str)]
    missing = (~df["path"].map(os.path.exists)).sum()
    if missing:
        alt = [p[:-4] + ".jpg" for p in df["path"]] if ext == ".png" else df["path"]
        df["path"] = [a if not os.path.exists(p) and os.path.exists(a) else p for p, a in zip(df["path"], alt)]
        missing = (~df["path"].map(os.path.exists)).sum()
    if missing:
        raise FileNotFoundError(f"{missing} APTOS images not found under {img_dir}")
    return df.rename(columns={"diagnosis": "grade"})[["id_code", "path", "grade"]]


IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def load_oiaddr(root, log=print):
    """OIA-DDR grading photos (Li et al. 2019; 147 hospitals, 42 camera models; CC BY-NC-SA 4.0), from whatever folder
    layout the copy has. Labels are read from, in order: '<image> <grade>' text files, CSV files with an image column
    and a grade column, or class folders named 0 to 5. Grade 5 (ungradable) is dropped. Our own split is used later."""
    import re
    index = {}
    for d, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(IMG_EXT):
                index.setdefault(f, os.path.join(d, f))
                index.setdefault(os.path.splitext(f)[0], os.path.join(d, f))
    labels = {}
    for d, _, fs in os.walk(root):
        for f in fs:
            p = os.path.join(d, f)
            if f.lower().endswith(".txt"):
                for line in open(p, errors="ignore"):
                    m = re.match(r"^\s*(\S+\.(?:jpg|jpeg|png|tif|tiff))\s+(\d)\s*$", line, re.I)
                    if m:
                        labels.setdefault(os.path.basename(m.group(1)), int(m.group(2)))
            elif f.lower().endswith(".csv"):
                try:
                    c = pd.read_csv(p)
                except Exception:  # noqa: BLE001
                    continue
                gcol = next((k for k in c.columns if str(k).lower() in ("diagnosis", "label", "grade", "level", "dr_grade", "class")), None)
                icol = next((k for k in c.columns if c[k].astype(str).map(lambda v: v in index).mean() > 0.5), None)
                if gcol is not None and icol is not None:
                    for i, g in zip(c[icol].astype(str), c[gcol]):
                        if str(g).strip().isdigit():
                            labels.setdefault(os.path.basename(index[i]) if i in index else i, int(g))
    how = "label files"
    if not labels:          # class folders: .../0/x.jpg ... .../5/y.jpg
        how = "class folders"
        for k, p in index.items():
            parent = os.path.basename(os.path.dirname(p))
            if parent.isdigit() and os.path.splitext(k)[1]:
                labels.setdefault(k, int(parent))
    rows = [dict(id_code=k, path=index[k], grade=g) for k, g in labels.items() if k in index]
    df = pd.DataFrame(rows, columns=["id_code", "path", "grade"]).drop_duplicates("path")
    n_all = len(df)
    df = df[df.grade.between(0, 4)].reset_index(drop=True)
    log(f"OIA-DDR: {n_all} labelled photos found under {root} (from {how}); {n_all - len(df)} ungradable dropped; "
        f"grades {df.grade.value_counts().sort_index().to_dict()}")
    if len(df) < 100:
        raise FileNotFoundError(f"Fewer than 100 labelled OIA-DDR photos found under {root}: check the dataset is attached")
    return df


def load_deepdrid(root, part):
    """part: 'training' or 'validation'. Returns one row per photo with eye-level grade, quality and source."""
    folder = os.path.join(root, "regular_fundus_images", f"regular-fundus-{part}")
    lab = pd.read_csv(os.path.join(folder, f"regular-fundus-{part}.csv"))
    src_csv = os.path.join(folder, f"regular-fundus-source-{part}.csv")
    if os.path.exists(src_csv):
        lab = lab.merge(pd.read_csv(src_csv)[["image_id", "Source"]], on="image_id", how="left")
    else:
        lab["Source"] = "unknown"
    lab["grade"] = lab["left_eye_DR_Level"].fillna(lab["right_eye_DR_Level"]).astype(int)
    lab["poor_quality"] = (lab["Overall quality"] == 0).astype(int)
    lab["path"] = [os.path.join(folder, "Images", str(p), f"{i}.jpg") for p, i in zip(lab["patient_id"], lab["image_id"])]
    lab = lab[lab["path"].map(os.path.exists)].reset_index(drop=True)
    lab["part"] = part
    return lab[["patient_id", "image_id", "path", "grade", "poor_quality", "Source", "part"]]


# ------------------------------------------------------------------ figures
def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def fig_roc(curves, path):
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=150)
    _style(ax)
    for label, y, p, thr, color in curves:
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr * 100, tpr * 100, color=color, lw=2, label=label)
        pred = p >= thr
        sens = (pred & (y == 1)).sum() / max(1, (y == 1).sum())
        fpr_pt = (pred & (y == 0)).sum() / max(1, (y == 0).sum())
        ax.plot([fpr_pt * 100], [sens * 100], "o", ms=7, color=color, markeredgecolor="white", markeredgewidth=1.5)
    ax.plot([0, 100], [0, 100], color=GRID, lw=1, ls="--")
    ax.set_xlabel("False alarms: healthy eyes sent to a specialist (%)", color=INK)
    ax.set_ylabel("Referable eyes caught (%)", color=INK)
    ax.set_title("Referable diabetic retinopathy: catches vs false alarms\n(dots = threshold frozen on validation)",
                 loc="left", fontsize=10.5, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.015), ncol=2, borderaxespad=0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_gallery(rows, path, title, size=160):
    rows = rows[:8]
    if len(rows) == 0:
        return
    fig, axes = plt.subplots(1, len(rows), figsize=(2.0 * len(rows), 2.5), dpi=150)
    axes = np.atleast_1d(axes)
    for ax, r in zip(axes, rows):
        ax.imshow(Image.open(r["path"]).convert("RGB").resize((size, size)))
        ax.set_title(r["caption"], fontsize=7, color=INK)
        ax.axis("off")
    fig.suptitle(title, fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def stress_test(model, gate, ref_thr, poor_thr, sets, cfg, log=print) -> pd.DataFrame:
    """Score the frozen tool on deliberately worse copies of held-out photos. Nothing is re-tuned."""
    from functools import partial
    rows = []
    for set_name, d in sets:
        y = (d["grade"].to_numpy() >= ds.REFERABLE_FROM).astype(int)
        for kind, strength, level in ds.stress_conditions():
            fn = None if kind == "clean" else partial(ds.degrade, kind=kind, level=level)
            P, E = ds.predict(model, ds.make_dataset(list(d["cached"]), None, cfg.size, False, degrade_fn=fn),
                              cfg.batch_size, workers=cfg.workers, with_embeddings=True)
            p_ref = ds.referable_prob(P)
            p_poor = gate.predict_proba(E)[:, 1] if gate is not None else np.zeros(len(d))
            rep = ds.binary_report(y, p_ref, ref_thr)
            w = ds.workflow_report(d["grade"], ds.decide(p_ref, p_poor, ref_thr, poor_thr))
            w0 = ds.workflow_report(d["grade"], ds.decide(p_ref, np.zeros(len(d)), ref_thr, poor_thr))
            rows.append(dict(photos=set_name, change=kind, strength=strength, level=level, n=len(d), referable=int(y.sum()),
                             AUC=rep["AUC"], sensitivity=rep["sensitivity"], specificity=rep["specificity"],
                             retake_share=w["RETAKE"] / max(1, w["photos"]),
                             referable_all_clear_with_check=w["referable_given_all_clear"],
                             referable_all_clear_without_check=w0["referable_given_all_clear"],
                             grade3_4_all_clear_with_check=w["sight_threatening_given_all_clear"],
                             grade3_4_all_clear_without_check=w0["sight_threatening_given_all_clear"]))
            log(f"  stress {set_name:>8} {kind:>15} {strength:>6}: sensitivity {rep['sensitivity']:.3f}, "
                f"specificity {rep['specificity']:.3f}, referable all-clears {w['referable_given_all_clear']} "
                f"with check / {w0['referable_given_all_clear']} without")
    return pd.DataFrame(rows)


def fig_stress(st: pd.DataFrame, path):
    """Per photo set: sensitivity and specificity (left), and referable eyes given an all-clear
    with vs without the quality check (right). Horizontal bars so every condition label is readable."""
    sets = list(dict.fromkeys(st["photos"]))
    fig, axes = plt.subplots(len(sets), 2, figsize=(12, 4.2 * len(sets)), dpi=150, squeeze=False)
    for i, name in enumerate(sets):
        d = st[st.photos == name].reset_index(drop=True)
        labels = ["original" if c == "clean" else f"{c.replace('_', ' ')}, {s}" for c, s in zip(d.change, d.strength)]
        y = np.arange(len(d))[::-1]
        h = 0.36
        ax = axes[i, 0]
        _style(ax)
        ax.grid(axis="y", visible=False)
        ax.barh(y + h / 2, d.sensitivity * 100, height=h, color=C_MAIN, label="sensitivity (referable eyes caught)")
        ax.barh(y - h / 2, d.specificity * 100, height=h, color=C_PROBE, label="specificity (healthy eyes cleared)")
        ax.set_yticks(y, labels, fontsize=8)
        ax.set_xlim(0, 100)
        ax.set_xlabel("%", color=MUTED, fontsize=9)
        ax.set_title(f"{name}: accuracy at the frozen threshold", loc="left", fontsize=10, color=INK, pad=24)
        ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.015), ncol=2, borderaxespad=0)
        ax = axes[i, 1]
        _style(ax)
        ax.grid(axis="y", visible=False)
        a, b = d.referable_all_clear_without_check, d.referable_all_clear_with_check
        ax.barh(y + h / 2, a, height=h, color=C_GRAYBAR, label="without quality check")
        ax.barh(y - h / 2, b, height=h, color=C_MAIN, label="with quality check")
        top = max(a.max(), b.max(), 1)
        for yy, va, vb in zip(y, a, b):
            ax.text(va + top * 0.01, yy + h / 2, str(int(va)), va="center", fontsize=7, color=MUTED)
            ax.text(vb + top * 0.01, yy - h / 2, str(int(vb)), va="center", fontsize=7, color=INK)
        ax.set_xlim(0, top * 1.12)
        ax.set_yticks(y, labels, fontsize=8)
        ax.set_xlabel(f"referable eyes given an all-clear (of {int(d.referable.iloc[0])})", color=MUTED, fontsize=9)
        ax.set_title(f"{name}: false all-clears", loc="left", fontsize=10, color=INK, pad=24)
        ax.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.015), ncol=2, borderaxespad=0)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_stress_examples(photo_path, size, path):
    conds = ds.stress_conditions()
    base = Image.open(photo_path).convert("RGB").resize((size, size))
    fig, axes = plt.subplots(1, len(conds), figsize=(1.7 * len(conds), 2.2), dpi=150)
    for ax, (kind, strength, level) in zip(axes, conds):
        ax.imshow(ds.degrade(base, kind, level))
        ax.set_title("original" if kind == "clean" else f"{kind.replace('_', ' ')}\n{strength}", fontsize=7, color=INK)
        ax.axis("off")
    fig.suptitle("The same photo under each stress-test change", fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------ main experiment
def run(cfg: RunConfig, log=print) -> dict:
    import torch
    t0 = time.time()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(cfg.out, exist_ok=True)
    out = {"config": asdict(cfg)}

    # 1. APTOS: cache, find near-duplicates, split by duplicate group
    ap = load_aptos(cfg.aptos_csv, cfg.aptos_dir)
    if cfg.max_images:
        ap = pd.concat([d.head(max(2, cfg.max_images // 5)) for _, d in ap.groupby("grade")]).reset_index(drop=True)
    ap["cached"] = ds.cache_images(list(ap["path"]), os.path.join(cfg.cache, f"aptos_{cfg.size}"), cfg.size, max(1, cfg.workers))
    groups, pairs = ds.duplicate_groups(list(ap["cached"]), cfg.dup_max_dist)
    ap["dup_group"] = groups
    conflicts = sum(1 for i, j, _ in pairs if ap.grade[i] != ap.grade[j])
    out["duplicates"] = dict(pairs=len(pairs), pairs_with_different_grades=conflicts,
                             images_in_groups=int((pd.Series(groups).value_counts() > 1).sum()))
    log(f"Near-duplicate pairs: {len(pairs)} ({conflicts} with different grades). Kept together when splitting.")
    ap["split"] = ds.group_split(ap["grade"].to_numpy(), groups, cfg.seed)
    ap.to_csv(os.path.join(cfg.out, "aptos_split.csv"), index=False)
    log(ap.groupby("split")["grade"].value_counts().unstack(fill_value=0).to_string())
    tr, va, te = (ap[ap.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    ybin = lambda d: (d["grade"].to_numpy() >= ds.REFERABLE_FROM).astype(int)
    ox = None
    if cfg.oiaddr_root:      # v2: more cameras in training and validation; APTOS test and DeepDRiD unchanged
        ox = load_oiaddr(cfg.oiaddr_root, log)
        if cfg.max_images:
            ox = pd.concat([d.head(max(2, cfg.max_images // 5)) for _, d in ox.groupby("grade")]).reset_index(drop=True)
        ox["cached"] = ds.cache_images(list(ox["path"]), os.path.join(cfg.cache, f"oiaddr_{cfg.size}"), cfg.size, max(1, cfg.workers))
        g2, p2 = ds.duplicate_groups(list(ox["cached"]), cfg.dup_max_dist)
        ox["dup_group"] = g2
        ox["split"] = ds.group_split(ox["grade"].to_numpy(), g2, cfg.seed)
        ox.to_csv(os.path.join(cfg.out, "oiaddr_split.csv"), index=False)
        out["oiaddr"] = dict(photos=len(ox), near_duplicate_pairs=len(p2),
                             split=ox.split.value_counts().to_dict(), grades=ox.grade.value_counts().sort_index().to_dict())
        log("OIA-DDR split:\n" + ox.groupby("split")["grade"].value_counts().unstack(fill_value=0).to_string())
        tr = pd.concat([tr.assign(source="APTOS"), ox[ox.split == "train"].assign(source="OIA-DDR")], ignore_index=True)
        va = pd.concat([va.assign(source="APTOS"), ox[ox.split == "val"].assign(source="OIA-DDR")], ignore_index=True)
        log(f"Training photos: {len(tr)} (APTOS + OIA-DDR); validation photos: {len(va)} (both sources)")

    # 2. Baselines
    results = {}
    Fc = {s: np.stack([ds.color_features(p) for p in d["cached"]]) for s, d in (("train", tr), ("val", va), ("test", te))}
    m = ds.fit_logreg(Fc["train"], ybin(tr))
    pv, pt = m.predict_proba(Fc["val"])[:, 1], m.predict_proba(Fc["test"])[:, 1]
    thr = ds.threshold_for_sensitivity(ybin(va), pv, cfg.target_sensitivity)
    results["colour_baseline"] = ds.binary_report(ybin(te), pt, thr, "Colour histogram + logistic regression")
    curves_te = [("Colour baseline (APTOS test)", ybin(te), pt, thr, C_COLOR)]

    probe_model = ds.build_model(cfg.arch, cfg.pretrained)
    emb = {}
    for s, d in (("train", tr), ("val", va), ("test", te)):
        _, emb[s] = ds.predict(probe_model, ds.make_dataset(list(d["cached"]), d["grade"].to_numpy(), cfg.size, False),
                               cfg.batch_size, workers=cfg.workers, with_embeddings=True)
    m = ds.fit_logreg(emb["train"], ybin(tr), C=0.1)
    pv, pt = m.predict_proba(emb["val"])[:, 1], m.predict_proba(emb["test"])[:, 1]
    thr = ds.threshold_for_sensitivity(ybin(va), pv, cfg.target_sensitivity)
    results["linear_probe"] = ds.binary_report(ybin(te), pt, thr, "Pretrained features, not fine-tuned + logistic regression")
    curves_te.append(("Linear probe (APTOS test)", ybin(te), pt, thr, C_PROBE))
    del probe_model
    for k in ("colour_baseline", "linear_probe"):
        log(f"{results[k]['model']}: AUC {results[k]['AUC']:.3f}, sensitivity {results[k]['sensitivity']:.3f}, specificity {results[k]['specificity']:.3f}")

    # 3. Fine-tuned model; threshold frozen on validation
    model = ds.build_model(cfg.arch, cfg.pretrained)
    hist = ds.train_model(model, ds.make_dataset(list(tr["cached"]), tr["grade"].to_numpy(), cfg.size, True),
                          ds.make_dataset(list(va["cached"]), va["grade"].to_numpy(), cfg.size, False),
                          va["grade"].to_numpy(), cfg.epochs, cfg.lr, cfg.batch_size, workers=cfg.workers,
                          out_path=os.path.join(cfg.out, "model_best.pt"), log=log)
    hist.to_csv(os.path.join(cfg.out, "training_history.csv"), index=False)
    Pv, Ev = ds.predict(model, ds.make_dataset(list(va["cached"]), None, cfg.size, False), cfg.batch_size, workers=cfg.workers, with_embeddings=True)
    ref_thr = ds.threshold_for_sensitivity(ybin(va), ds.referable_prob(Pv), cfg.target_sensitivity)
    out["frozen"] = asdict(ds.Frozen(cfg.arch, cfg.size, ref_thr, cfg.poor_threshold, cfg.target_sensitivity))
    log(f"FROZEN referral threshold {ref_thr:.3f} (validation sensitivity target {cfg.target_sensitivity:.0%})")

    # 4. APTOS held-out test (scored once, after freezing)
    Pt = ds.predict(model, ds.make_dataset(list(te["cached"]), None, cfg.size, False), cfg.batch_size, workers=cfg.workers)
    pt = ds.referable_prob(Pt)
    results["fine_tuned"] = ds.binary_report(ybin(te), pt, ref_thr, f"Fine-tuned {cfg.arch}")
    out["aptos_grades"] = ds.grade_report(te["grade"], Pt)
    out["aptos_flagged_by_grade"] = ds.flagged_by_grade(te["grade"], pt, ref_thr)
    curves_te.append((f"Fine-tuned (APTOS test, India)", ybin(te), pt, ref_thr, C_MAIN))
    te = te.assign(p_referable=pt, pred_grade=Pt.argmax(1))
    te.to_csv(os.path.join(cfg.out, "aptos_test_predictions.csv"), index=False)
    log(f"APTOS test: AUC {results['fine_tuned']['AUC']:.3f}, sensitivity {results['fine_tuned']['sensitivity']:.3f}, "
        f"specificity {results['fine_tuned']['specificity']:.3f}, missed {results['fine_tuned']['missed_FN']}")

    if ox is not None:      # OIA-DDR held-out test: new photos from the same hospitals as its training part
        ot = ox[ox.split == "test"].reset_index(drop=True)
        Po = ds.predict(model, ds.make_dataset(list(ot["cached"]), None, cfg.size, False), cfg.batch_size, workers=cfg.workers)
        ot["p_referable"] = ds.referable_prob(Po)
        results["oiaddr_test"] = ds.binary_report(ybin(ot), ot["p_referable"].to_numpy(), ref_thr, "Fine-tuned, OIA-DDR test (China)")
        ot.drop(columns=["path", "cached"]).to_csv(os.path.join(cfg.out, "oiaddr_test_predictions.csv"), index=False)
        curves_te.append(("Fine-tuned (OIA-DDR test, China)", ybin(ot), ot["p_referable"].to_numpy(), ref_thr, C_PROBE))
        log(f"OIA-DDR test: AUC {results['oiaddr_test']['AUC']:.3f}, sensitivity {results['oiaddr_test']['sensitivity']:.3f}, "
            f"specificity {results['oiaddr_test']['specificity']:.3f}")

    # 5. External: DeepDRiD (never used to train the DR model or to set its threshold)
    if cfg.ddr_root:
        dd = pd.concat([load_deepdrid(cfg.ddr_root, "training"), load_deepdrid(cfg.ddr_root, "validation")], ignore_index=True)
        dd["cached"] = ds.cache_images(list(dd["path"]), os.path.join(cfg.cache, f"ddr_{cfg.size}"), cfg.size, max(1, cfg.workers))
        Pd, Ed = ds.predict(model, ds.make_dataset(list(dd["cached"]), None, cfg.size, False), cfg.batch_size,
                            workers=cfg.workers, with_embeddings=True)
        dd["p_referable"] = ds.referable_prob(Pd)
        dd["near_copy_in_training"] = False
        if ox is not None:      # DeepDRiD must stay unseen: look for near-copies among the OIA-DDR photos used
            used = ox[ox.split.isin(["train", "val"])]
            Hd = np.stack([ds.dhash(p) for p in dd["cached"]]).astype(np.float32)
            Ho = np.stack([ds.dhash(p) for p in used["cached"]]).astype(np.float32)
            dist = Hd.shape[1] - (Hd @ Ho.T + (1 - Hd) @ (1 - Ho).T)
            dd["near_copy_in_training"] = (dist <= cfg.dup_max_dist).any(1)
            out["deepdrid_near_copies_in_oiaddr"] = int(dd["near_copy_in_training"].sum())
            log(f"DeepDRiD photos with a near-copy among the OIA-DDR photos used: {out['deepdrid_near_copies_in_oiaddr']}")
        graded = dd["grade"] <= 4
        g = dd[graded]
        rep = ds.binary_report(ybin(g), g["p_referable"].to_numpy(), ref_thr, "Fine-tuned, external DeepDRiD (China)")
        rep["AUC_CI_by_patient"] = _patient_bootstrap_auc(ybin(g), g["p_referable"].to_numpy(), g["patient_id"].to_numpy())
        results["external_all"] = rep
        if dd["near_copy_in_training"].any():
            gc = g[~g["near_copy_in_training"]]
            results["external_no_near_copies"] = ds.binary_report(ybin(gc), gc["p_referable"].to_numpy(), ref_thr,
                                                                  "External DeepDRiD, near-copies of training photos removed")
        for q, name in ((0, "good"), (1, "poor")):
            gq = g[g.poor_quality == q]
            if len(gq) and len(set(ybin(gq))) > 1:
                results[f"external_{name}_photos"] = ds.binary_report(ybin(gq), gq["p_referable"].to_numpy(), ref_thr,
                                                                       f"External, {name}-quality photos")
        curves_te.append(("Fine-tuned (DeepDRiD, China, external)", ybin(g), g["p_referable"].to_numpy(), ref_thr, C_EXT))
        out["external_by_source"] = [dict(source=s, photos=len(d), **{k: v for k, v in ds.binary_report(
            ybin(d), d["p_referable"].to_numpy(), ref_thr).items() if k in ("sensitivity", "specificity", "missed_FN")})
            for s, d in g.groupby("Source") if len(d) >= 20 and len(set(ybin(d))) > 1]

        # 6. Quality gate: trained on DeepDRiD training photos' quality labels, tested on DeepDRiD validation
        trq, teq = dd.part.to_numpy() == "training", dd.part.to_numpy() == "validation"
        gate = ds.fit_logreg(Ed[trq], dd.loc[trq, "poor_quality"].to_numpy(), C=0.1)
        dd["p_poor"] = gate.predict_proba(Ed)[:, 1]
        from sklearn.metrics import roc_auc_score
        v = dd[teq]
        out["quality_gate"] = dict(AUC_detecting_poor_photos=float(roc_auc_score(v.poor_quality, v.p_poor)),
                                   flagged_share=float((v.p_poor >= cfg.poor_threshold).mean()),
                                   poor_photos_flagged=float((v.p_poor[v.poor_quality == 1] >= cfg.poor_threshold).mean()),
                                   good_photos_flagged=float((v.p_poor[v.poor_quality == 0] >= cfg.poor_threshold).mean()))
        vg = v[v.grade <= 4]
        with_gate = ds.decide(vg.p_referable, vg.p_poor, ref_thr, cfg.poor_threshold)
        no_gate = ds.decide(vg.p_referable, np.zeros(len(vg)), ref_thr, cfg.poor_threshold)
        out["workflow"] = dict(with_quality_gate=ds.workflow_report(vg.grade, with_gate),
                               without_quality_gate=ds.workflow_report(vg.grade, no_gate),
                               ungradable_photos=int((v.grade == 5).sum()),
                               ungradable_sent_to_retake_or_refer=int((ds.decide(v[v.grade == 5].p_referable, v[v.grade == 5].p_poor,
                                                                                ref_thr, cfg.poor_threshold) != "RESCREEN").sum()))
        dd["decision"] = ds.decide(dd.p_referable, dd.p_poor, ref_thr, cfg.poor_threshold)
        dd.to_csv(os.path.join(cfg.out, "deepdrid_predictions.csv"), index=False)
        with open(os.path.join(cfg.out, "quality_gate.pkl"), "wb") as f:
            pickle.dump(gate, f)
        log(f"External DeepDRiD: AUC {rep['AUC']:.3f}, sensitivity {rep['sensitivity']:.3f}, specificity {rep['specificity']:.3f}")
        log(f"Workflow on DeepDRiD validation: {out['workflow']}")

    # 7. Tables and figures
    summary = pd.DataFrame([{k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()
                             if k in ("model", "n", "referable", "AUC", "sensitivity", "specificity", "PPV", "NPV",
                                      "missed_FN", "false_alarms_FP", "referred_share")} for r in results.values()])
    summary.to_csv(os.path.join(cfg.out, "summary.csv"), index=False)
    fig_roc(curves_te, os.path.join(cfg.out, "roc.png"))
    misses = te[(te.grade >= ds.REFERABLE_FROM) & (te.p_referable < ref_thr)].sort_values("p_referable")
    fig_gallery([dict(path=r.cached, caption=f"grade {r.grade} ({ds.GRADES[r.grade]})\nscore {r.p_referable:.2f}")
                 for r in misses.itertuples()], os.path.join(cfg.out, "missed_cases.png"),
                "Referable eyes the model missed (APTOS test) — look for why")
    if cfg.ddr_root:   # external misses: DeepDRiD photos (CC BY-SA 4.0), safe to show publicly with attribution
        mx = dd[(dd.grade >= ds.REFERABLE_FROM) & (dd.decision == "RESCREEN")].sort_values("p_referable")
        fig_gallery([dict(path=r.cached, caption=f"grade {r.grade} ({ds.GRADES[r.grade]})\nscore {r.p_referable:.2f}, "
                                                 f"{'poor' if r.poor_quality else 'good'} photo")
                     for r in mx.itertuples()], os.path.join(cfg.out, "missed_external.png"),
                    "Referable eyes given an all-clear on external DeepDRiD photos (CC BY-SA 4.0)")
    fps = te[(te.grade < ds.REFERABLE_FROM) & (te.p_referable >= ref_thr)].sort_values("p_referable", ascending=False)
    fig_gallery([dict(path=r.cached, caption=f"grade {r.grade} ({ds.GRADES[r.grade]})\nscore {r.p_referable:.2f}")
                 for r in fps.itertuples()], os.path.join(cfg.out, "false_alarms.png"),
                "Eyes sent to a specialist that did not need it (APTOS test)")
    # 8. Speed on the CPU alone (no graphics card), from the original photo file to a decision.
    #    Optional: if it fails, the run still finishes and the write-up says speed was not measured.
    try:
        if cfg.ddr_root:
            bench_paths, bench_gate = list(dd.loc[dd.part == "validation", "path"]), gate
        else:
            bench_paths, bench_gate = list(te["path"]), None
        cs = ds.cpu_benchmark(model, bench_gate, bench_paths, cfg.size, ref_thr, cfg.poor_threshold)
        cs["model_file_MB"] = os.path.getsize(os.path.join(cfg.out, "model_best.pt")) / 1e6
        out["cpu_speed"] = cs
        log(f"CPU only: {cs['median_seconds_per_photo']:.2f} s per photo (median of {cs['photos_timed']}), "
            f"heat map {cs['heatmap_median_seconds']:.2f} s, {cs['cpu_threads']} threads, {cs['cpu']}, "
            f"model file {cs['model_file_MB']:.1f} MB")
    except Exception as e:  # never lose a finished run over a speed test
        out["cpu_speed"] = dict(error=repr(e))
        log(f"CPU speed test skipped: {e!r}")

    # 9. Stress test: the frozen tool on deliberately worse copies of held-out photos (planned before results).
    #    Optional: if it fails, the run still finishes and the write-up says it was not run.
    try:
        sets = [("APTOS test", te)]
        if cfg.ddr_root:
            sets.append(("DeepDRiD validation", dd[(dd.part == "validation") & (dd.grade <= 4)].reset_index(drop=True)))
        log("Stress test (worse photos):")
        st = stress_test(model, gate if cfg.ddr_root else None, ref_thr, cfg.poor_threshold, sets, cfg, log)
        st.to_csv(os.path.join(cfg.out, "stress_test.csv"), index=False)
        out["stress_test"] = st.to_dict("records")
        fig_stress(st, os.path.join(cfg.out, "stress_test.png"))
        ex = sets[-1][1]
        ex = ex[ex.grade >= ds.REFERABLE_FROM] if (ex.grade >= ds.REFERABLE_FROM).any() else ex
        fig_stress_examples(ex["cached"].iloc[0], cfg.size, os.path.join(cfg.out, "stress_examples.png"))
    except Exception as e:  # never lose a finished run over the stress test
        out["stress_test_error"] = repr(e)
        log(f"Stress test skipped: {e!r}")

    out["results"] = results
    out["seconds"] = time.time() - t0
    with open(os.path.join(cfg.out, "results.json"), "w") as f:
        json.dump(out, f, indent=2, default=lambda o: o.to_dict("records") if isinstance(o, pd.DataFrame) else (o.tolist() if hasattr(o, "tolist") else str(o)))
    log(f"Done in {out['seconds'] / 60:.1f} min. Results in {cfg.out}/")
    out["summary"] = summary
    out["model"] = model
    return out


def _patient_bootstrap_auc(y, p, patients, n=1000, seed=0):
    """AUC interval resampling whole patients (each has up to 4 photos), so it is not falsely narrow."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    pats = np.unique(patients)
    idx = {q: np.where(patients == q)[0] for q in pats}
    vals = []
    for _ in range(n):
        pick = np.concatenate([idx[q] for q in rng.choice(pats, len(pats))])
        if len(set(y[pick])) > 1:
            vals.append(roc_auc_score(y[pick], p[pick]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--aptos-csv", required=True)
    a.add_argument("--aptos-dir", required=True)
    a.add_argument("--ddr-root", default=None, help="DeepDRiD (external test)")
    a.add_argument("--oiaddr-root", default=None, help="v2 only: OIA-DDR photos added to training and validation")
    a.add_argument("--out", default="results")
    a.add_argument("--epochs", type=int, default=12)
    a.add_argument("--size", type=int, default=384)
    a.add_argument("--no-pretrained", action="store_true")
    a.add_argument("--max-images", type=int, default=None)
    a.add_argument("--workers", type=int, default=2)
    x = a.parse_args()
    r = run(RunConfig(aptos_csv=x.aptos_csv, aptos_dir=x.aptos_dir, ddr_root=x.ddr_root, oiaddr_root=x.oiaddr_root,
                      out=x.out, size=x.size, epochs=x.epochs, pretrained=not x.no_pretrained,
                      max_images=x.max_images, workers=x.workers))
    print(r["summary"].to_string(index=False))
