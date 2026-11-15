"""Roshni: diabetic-eye screening for a primary-care diabetes clinic in Pakistan.

One retinal photo in, one of three next steps out:
  REFER     - referable diabetic retinopathy suspected (moderate NPDR or worse)
  RETAKE    - photo not good enough to give an all-clear; take it again
  RESCREEN  - no referable disease seen on a good photo; screen again in 12 months

Safety rules built into the code:
  * A poor photo can never produce an all-clear, but it never blocks a referral either.
  * The referral threshold is chosen on the validation split to reach a sensitivity target
    fixed in advance (90%), then frozen before any test image is scored.
  * Near-duplicate photos are grouped before splitting, so a copy of a test image can never
    sit in training (a common way to get results that are too good to be true).
  * The external test set (DeepDRiD, China) is never used for training or for any threshold.

Prototype for a hackathon. Not medical advice and not a diagnosis.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image

GRADES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
REFERABLE_FROM = 2           # moderate NPDR or worse = "more-than-mild DR", the usual referral line
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ============================================================ preprocessing
def crop_fundus(img: Image.Image, tol: int = 10) -> Image.Image:
    """Cut away the black border around the round fundus image."""
    a = np.asarray(img.convert("L"))
    mask = a > tol
    if mask.sum() < 0.05 * a.size:          # almost all dark: leave it (the quality gate will judge it)
        return img
    ys, xs = np.where(mask)
    return img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def prep_image(path_or_img, size: int = 384) -> Image.Image:
    img = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    img = crop_fundus(img.convert("RGB"))
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (0, 0, 0))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((size, size), Image.BILINEAR)


def _cache_one(args):
    src, dst, size = args
    if not os.path.exists(dst):
        try:
            prep_image(src, size).save(dst, quality=95)
        except Exception as e:                # a broken file should not stop the run
            return f"FAILED {src}: {e}"
    return None


def cache_images(srcs, out_dir, size=384, workers=4):
    """Crop, pad and resize every image once, so training reads small JPEGs."""
    from multiprocessing import Pool
    os.makedirs(out_dir, exist_ok=True)
    dsts = [os.path.join(out_dir, os.path.splitext(os.path.basename(s))[0] + ".jpg") for s in srcs]
    jobs = [(s, d, size) for s, d in zip(srcs, dsts)]
    t0 = time.time()
    with Pool(workers) as pool:
        errs = [e for e in pool.imap(_cache_one, jobs, chunksize=8) if e]
    print(f"Cached {len(dsts)} images to {out_dir} in {time.time() - t0:.0f}s; {len(errs)} failed")
    for e in errs[:5]:
        print(" ", e)
    return dsts


# ============================================================ near-duplicates
def dhash(path, n: int = 16) -> np.ndarray:
    a = np.asarray(Image.open(path).convert("L").resize((n + 1, n), Image.BILINEAR), dtype=np.float32)
    return (a[:, 1:] > a[:, :-1]).ravel()


def duplicate_groups(paths, max_dist: int = 10):
    """Group photos whose 256-bit difference hashes are within `max_dist` bits.
    Returns (group id per image, list of near-duplicate pairs)."""
    H = np.stack([dhash(p) for p in paths]).astype(np.float32)
    n_bits = H.shape[1]
    parent = list(range(len(paths)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = []
    for s in range(0, len(H), 1024):
        blk = H[s:s + 1024]
        same = blk @ H.T + (1 - blk) @ (1 - H).T            # bits that agree
        dist = n_bits - same
        ii, jj = np.where(dist <= max_dist)
        for i, j in zip(ii + s, jj):
            if i < j:
                pairs.append((int(i), int(j), int(dist[i - s, j])))
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    groups = np.array([find(i) for i in range(len(paths))])
    return groups, pairs


def group_split(labels, groups, seed: int = 0, n_splits: int = 7):
    """Seven stratified folds: one test, one validation, five training. Near-duplicate groups stay whole, so the
    actual shares can differ from 1/7 each (the real split is reported from aptos_split.csv)."""
    from sklearn.model_selection import StratifiedGroupKFold
    sgk = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.zeros(len(labels), int)
    for k, (_, idx) in enumerate(sgk.split(np.zeros(len(labels)), labels, groups)):
        folds[idx] = k
    split = np.where(folds == 0, "test", np.where(folds == 1, "val", "train"))
    return split


# ============================================================ simple baseline features
def color_features(path) -> np.ndarray:
    """16-bin histograms per colour channel over the fundus (black border ignored) + mean/std."""
    a = np.asarray(Image.open(path).convert("RGB").resize((128, 128)), dtype=np.float32)
    m = a.sum(axis=2) > 30
    if m.sum() < 50:
        m = np.ones(a.shape[:2], bool)
    feats = []
    for ch in range(3):
        v = a[..., ch][m]
        h, _ = np.histogram(v, bins=16, range=(0, 255))
        feats += list(h / max(1, h.sum())) + [v.mean() / 255, v.std() / 255]
    return np.array(feats, np.float32)


# ============================================================ torch parts
def _torch():
    import torch
    return torch


def get_transforms(size: int, train: bool):
    from torchvision import transforms as T
    if train:
        return T.Compose([T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomRotation(25),
                          T.ColorJitter(0.2, 0.2, 0.1), T.ToTensor(), T.Normalize(MEAN, STD)])
    return T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


def make_dataset(paths, labels, size, train, degrade_fn=None):
    """degrade_fn (optional) makes each photo worse in a fixed way before scoring; used only by the stress test."""
    from torch.utils.data import Dataset
    tf = get_transforms(size, train)

    class _DS(Dataset):
        def __len__(self):
            return len(paths)

        def __getitem__(self, i):
            img = Image.open(paths[i]).convert("RGB")
            if img.size != (size, size):
                img = img.resize((size, size))
            if degrade_fn is not None:
                img = degrade_fn(img)
            return tf(img), int(labels[i]) if labels is not None else -1
    return _DS()


def build_model(arch: str = "efficientnet_b0", pretrained: bool = True, n_classes: int = 5):
    import torch.nn as nn
    import torchvision
    if arch == "efficientnet_b0":
        m = torchvision.models.efficientnet_b0(weights="IMAGENET1K_V1" if pretrained else None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    elif arch == "resnet50":
        m = torchvision.models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    else:
        raise ValueError(arch)
    m.arch = arch
    return m


def _embed(model, x):
    """Penultimate-layer features (used by the linear-probe baseline and the quality gate)."""
    torch = _torch()
    if getattr(model, "arch", "") == "resnet50":
        f = torch.nn.Sequential(*list(model.children())[:-1])(x)
    else:
        f = model.avgpool(model.features(x))
    return torch.flatten(f, 1)


def predict(model, ds, batch_size=32, device=None, workers=2, with_embeddings=False):
    torch = _torch()
    from torch.utils.data import DataLoader
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    P, E = [], []
    with torch.no_grad():
        for x, _ in dl:
            x = x.to(device)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                if with_embeddings:
                    e = _embed(model, x)
                    logits = model.classifier(e) if hasattr(model, "classifier") else model.fc(e)
                    E.append(e.float().cpu().numpy())
                else:
                    logits = model(x)
            P.append(torch.softmax(logits.float(), 1).cpu().numpy())
    P = np.concatenate(P)
    return (P, np.concatenate(E)) if with_embeddings else P


def referable_prob(P: np.ndarray) -> np.ndarray:
    return P[:, REFERABLE_FROM:].sum(axis=1)


def train_model(model, train_ds, val_ds, val_grades, epochs=12, lr=3e-4, batch_size=32,
                device=None, workers=2, out_path="model_best.pt", log=print):
    """Train; keep the epoch with the best validation AUC for referable DR (validation only)."""
    torch = _torch()
    from torch.utils.data import DataLoader
    from sklearn.metrics import roc_auc_score
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * max(1, len(dl)))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    yv = (np.asarray(val_grades) >= REFERABLE_FROM).astype(int)
    best, hist = -1.0, []
    for ep in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device == "cuda"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(y)
            n += len(y)
        pv = referable_prob(predict(model, val_ds, batch_size, device, workers))
        auc = roc_auc_score(yv, pv) if len(set(yv)) > 1 else float("nan")
        hist.append(dict(epoch=ep + 1, train_loss=tot / max(1, n), val_auc=auc, seconds=time.time() - t0))
        log(f"epoch {ep + 1}/{epochs}: loss {tot / max(1, n):.3f}, validation AUC (referable) {auc:.3f}, {time.time() - t0:.0f}s")
        if auc > best:
            best = auc
            torch.save(model.state_dict(), out_path)
    model.load_state_dict(torch.load(out_path, map_location=device))
    return pd.DataFrame(hist)


# ============================================================ thresholds and metrics
def threshold_for_sensitivity(y_bin, p, target=0.90) -> float:
    """Highest threshold whose sensitivity on these (validation) cases is at least `target`."""
    y_bin, p = np.asarray(y_bin), np.asarray(p)
    pos = np.sort(p[y_bin == 1])[::-1]
    if len(pos) == 0:
        return 0.5
    k = int(np.ceil(target * len(pos))) - 1            # need at least k+1 positives above the line
    return float(pos[k])


def bootstrap_auc(y, p, n=1000, seed=0):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) > 1:
            vals.append(roc_auc_score(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def binary_report(y_bin, p, thr, name="") -> dict:
    from sklearn.metrics import roc_auc_score
    y_bin, p = np.asarray(y_bin).astype(int), np.asarray(p)
    pred = p >= thr
    tp, fn = int((pred & (y_bin == 1)).sum()), int((~pred & (y_bin == 1)).sum())
    fp, tn = int((pred & (y_bin == 0)).sum()), int((~pred & (y_bin == 0)).sum())
    auc = roc_auc_score(y_bin, p) if len(set(y_bin)) > 1 else float("nan")
    return dict(model=name, n=len(y_bin), referable=int(y_bin.sum()), threshold=float(thr),
                AUC=auc, AUC_CI=bootstrap_auc(y_bin, p) if len(set(y_bin)) > 1 else (np.nan, np.nan),
                sensitivity=tp / max(1, tp + fn), sensitivity_CI=wilson(tp, tp + fn),
                specificity=tn / max(1, tn + fp), specificity_CI=wilson(tn, tn + fp),
                PPV=tp / max(1, tp + fp), NPV=tn / max(1, tn + fn),
                missed_FN=fn, false_alarms_FP=fp, referred_share=float(pred.mean()))


def grade_report(grades, P) -> dict:
    from sklearn.metrics import cohen_kappa_score, confusion_matrix
    g = np.asarray(grades).astype(int)
    pred = P.argmax(1)
    pr = referable_prob(P)
    per_grade = {GRADES[k]: dict(n=int((g == k).sum())) for k in range(5)}
    return dict(QWK=float(cohen_kappa_score(g, pred, weights="quadratic")),
                confusion=confusion_matrix(g, pred, labels=list(range(5))).tolist(),
                per_grade=per_grade, mean_referable_prob_by_grade=[float(pr[g == k].mean()) if (g == k).any() else None
                                                                   for k in range(5)])


def flagged_by_grade(grades, p, thr) -> pd.DataFrame:
    g = np.asarray(grades).astype(int)
    rows = []
    for k in range(5):
        m = g == k
        rows.append(dict(grade=f"{k} {GRADES[k]}", photos=int(m.sum()),
                         sent_to_specialist=int((p[m] >= thr).sum()),
                         share_sent=float((p[m] >= thr).mean()) if m.any() else np.nan))
    return pd.DataFrame(rows)


def fit_logreg(X, y, C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))
    return m.fit(X, y)


# ============================================================ the clinic decision
def decide(p_ref, p_poor, ref_thr, poor_thr):
    """Refer whenever referable DR is suspected (a poor photo never blocks a referral).
    Otherwise a poor photo gets RETAKE instead of an all-clear."""
    p_ref, p_poor = np.asarray(p_ref), np.asarray(p_poor)
    return np.where(p_ref >= ref_thr, "REFER", np.where(p_poor >= poor_thr, "RETAKE", "RESCREEN"))


def workflow_report(grades, decisions) -> dict:
    g = np.asarray(grades).astype(int)
    d = np.asarray(decisions)
    ref = g >= REFERABLE_FROM
    return dict(photos=len(g),
                REFER=int((d == "REFER").sum()), RETAKE=int((d == "RETAKE").sum()), RESCREEN=int((d == "RESCREEN").sum()),
                referable_given_all_clear=int((ref & (d == "RESCREEN")).sum()),
                referable_caught_by_retake=int((ref & (d == "RETAKE")).sum()),
                sight_threatening_given_all_clear=int(((g >= 3) & (d == "RESCREEN")).sum()))


# ============================================================ stress test: worse photos
# Planned before any result was seen. Each change is applied to the 384 px photo, at two fixed strengths,
# to mimic what cheaper cameras, rushed photography or sending photos by phone do to image quality.
# It is a stress test of the frozen tool, not a validation on real low-cost cameras.
DEGRADATIONS = {
    #  name               what it imitates                                   mild   strong
    "blur":            ("out of focus or camera shake (Gaussian blur, px)",  1.5,   3.0),
    "low_resolution":  ("a cheaper, lower-resolution sensor (shrunk to px)", 192,   96),
    "underexposed":    ("too dark: poor lighting or a small pupil (brightness x)", 0.6, 0.35),
    "compressed":      ("sent over a messaging app (JPEG quality)",          30,    8),
}


def degrade(img: Image.Image, kind: str, level) -> Image.Image:
    """Make a photo worse in one fixed, repeatable way (no randomness)."""
    import io
    from PIL import ImageEnhance, ImageFilter
    img = img.convert("RGB")
    if kind == "clean":
        return img
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(float(level)))
    if kind == "low_resolution":
        w, h = img.size
        return img.resize((int(level), int(level)), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    if kind == "underexposed":
        return ImageEnhance.Brightness(img).enhance(float(level))
    if kind == "compressed":
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=int(level))
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    raise ValueError(kind)


def stress_conditions():
    """[(change, strength, level)], starting with the original photos."""
    out = [("clean", "none", None)]
    for kind, (_, mild, strong) in DEGRADATIONS.items():
        out += [(kind, "mild", mild), (kind, "strong", strong)]
    return out


# ============================================================ one photo, as the clinic app runs it
def triage_one(model, gate, img, size, ref_thr, poor_thr, tf=None):
    """Screen one photo exactly as the clinic app does: open, crop, resize, score, quality check, decide.
    img is a PIL image or a file path. Returns (decision, p_referable, p_poor, grade_probabilities, embedding)."""
    torch = _torch()
    model.eval()
    tf = tf or get_transforms(size, False)
    dev = next(model.parameters()).device
    x = tf(prep_image(img, size)).unsqueeze(0).to(dev)
    with torch.no_grad():
        e = _embed(model, x)
        head = model.classifier if hasattr(model, "classifier") else model.fc
        P = torch.softmax(head(e), 1).cpu().numpy()
    e = e.cpu().numpy()
    p_ref = float(referable_prob(P)[0])
    p_poor = float(gate.predict_proba(e)[0, 1]) if gate is not None else 0.0
    return str(decide([p_ref], [p_poor], ref_thr, poor_thr)[0]), p_ref, p_poor, P[0], e


def _cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    import platform
    return platform.processor() or platform.machine() or "unknown"


def cpu_benchmark(model, gate, paths, size, ref_thr, poor_thr, n=20, heatmaps=5) -> dict:
    """Time the clinic step on the CPU alone (no graphics card), starting from the original photo file.
    The first photo is a warm-up and is not timed."""
    import copy
    torch = _torch()
    m = copy.deepcopy(model).cpu().eval()
    paths = list(paths)[: n + 1]
    if len(paths) < 2:
        raise ValueError("need at least 2 photos to time")
    tf = get_transforms(size, False)
    triage_one(m, gate, paths[0], size, ref_thr, poor_thr, tf)
    times = []
    for p in paths[1:]:
        t = time.perf_counter()
        triage_one(m, gate, p, size, ref_thr, poor_thr, tf)
        times.append(time.perf_counter() - t)
    hm = []
    for p in paths[1: 1 + heatmaps]:
        t = time.perf_counter()
        gradcam(m, Image.open(p).convert("RGB"), size, device="cpu")
        hm.append(time.perf_counter() - t)
    return dict(photos_timed=len(times), median_seconds_per_photo=float(np.median(times)),
                slowest_seconds_per_photo=float(np.max(times)),
                heatmap_median_seconds=float(np.median(hm)) if hm else float("nan"),
                cpu=_cpu_name(), cpu_threads=int(torch.get_num_threads()),
                parameters=int(sum(q.numel() for q in m.parameters())))


# ============================================================ explanation
def gradcam(model, pil_img, size=384, device=None):
    """Heat map of where the model looked when scoring referable DR (Grad-CAM on the last conv block), drawn on the photo."""
    cam = gradcam_map(model, pil_img, size, device)
    cam_img = Image.fromarray(np.uint8(cam * 255)).resize((size, size), Image.BILINEAR)
    base = prep_image(pil_img, size)
    heat = np.asarray(cam_img, dtype=np.float32) / 255
    rgb = np.asarray(base, dtype=np.float32)
    overlay = rgb * (1 - 0.45 * heat[..., None]) + np.array([255, 40, 0]) * (0.45 * heat[..., None])
    return Image.fromarray(np.uint8(np.clip(overlay, 0, 255)))


def gradcam_map(model, pil_img, size=384, device=None):
    """The raw Grad-CAM map (last conv block, scaled to 0..1) for the referable score."""
    torch = _torch()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    x = get_transforms(size, False)(prep_image(pil_img, size)).unsqueeze(0).to(device)
    layer = model.features[-1] if hasattr(model, "features") else model.layer4
    store = {}
    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("g", go[0]))
    try:
        logits = model(x)
        score = torch.logsumexp(logits[:, REFERABLE_FROM:], 1) - torch.logsumexp(logits, 1)
        model.zero_grad()
        score.sum().backward()
        w = store["g"].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((w * store["a"]).sum(1))[0].detach().cpu().numpy()
    finally:
        h1.remove()
        h2.remove()
    cam = cam - cam.min()
    return cam / (cam.max() + 1e-8)


# ============================================================ report text
# Languages for the note. A language is offered to users only once a native speaker on the team has checked it,
# except the defaults for the Pakistan setting, which the submission checker refuses to ship unchecked.
# To add a language later (for example Uzbek, checked by Laziza): add an entry here with checked=False, add its
# sentence to every entry in REPORT (tests fail until all four are there), have the reviewer approve the wording,
# then set checked=True. Nothing else in the code changes.
LANGS = {
    "en": dict(name="English", dir="ltr", reviewer="whole team", checked=True),
    "ur": dict(name="اردو", dir="rtl", reviewer="Zainab", checked=False),   # set True after Zainab's check (task Z2)
    "hi": dict(name="हिन्दी", dir="ltr", reviewer="Varun", checked=False),   # hidden until Varun checks it
}
# Shown by default for the Pakistan clinic setting.
DEFAULT_LANGS = ("en", "ur")


def shown_langs():
    """Languages the app offers: every checked language, plus the defaults (which must be checked before submission)."""
    return [c for c, v in LANGS.items() if v["checked"] or c in DEFAULT_LANGS]


# Same meaning in every language: the action, why, and that the doctor confirms it.
REPORT = {
    "REFER": {
        "en": "Referable diabetic retinopathy suspected (moderate or worse). Recommended next step, for the doctor to confirm: refer to an eye specialist.",
        "hi": "मधुमेह से आँख के पर्दे (रेटिना) की मध्यम या उससे गंभीर बीमारी (डायबिटिक रेटिनोपैथी) की आशंका है। अनुशंसित अगला कदम, डॉक्टर की पुष्टि के बाद: नेत्र विशेषज्ञ के पास भेजें।",
        "ur": "ذیابیطس کی وجہ سے آنکھ کے پردے (ریٹینا) کی درمیانی یا زیادہ بیماری کا شبہ ہے۔ تجویز کردہ اگلا قدم، ڈاکٹر کی تصدیق کے بعد: آنکھوں کے ماہر کے پاس بھیجیں۔",
    },
    "RETAKE": {
        "en": "The photo is not clear enough to rule out disease. Recommended next step: retake the photo; if it is still unclear, refer to an eye specialist.",
        "hi": "तस्वीर इतनी साफ़ नहीं है कि बीमारी को नकारा जा सके। अनुशंसित अगला कदम: तस्वीर दोबारा लें; अगर फिर भी साफ़ न हो, तो नेत्र विशेषज्ञ के पास भेजें।",
        "ur": "تصویر اتنی واضح نہیں کہ بیماری کو رد کیا جا سکے۔ تجویز کردہ اگلا قدم: تصویر دوبارہ لیں؛ اگر پھر بھی واضح نہ ہو تو آنکھوں کے ماہر کے پاس بھیجیں۔",
    },
    "RESCREEN": {
        "en": "No referable diabetic retinopathy seen on this photo. Recommended next step: routine eye screening again in 12 months.",
        "hi": "इस तस्वीर में ऐसी डायबिटिक रेटिनोपैथी नहीं दिखी जिसके लिए विशेषज्ञ के पास भेजना ज़रूरी हो। अनुशंसित अगला कदम: 12 महीने बाद फिर से नियमित आँखों की जाँच।",
        "ur": "اس تصویر میں ریفر کرنے کے قابل ذیابیطس کی آنکھ کی بیماری نظر نہیں آئی۔ تجویز کردہ اگلا قدم: 12 ماہ بعد دوبارہ معمول کا معائنہ۔",
    },
    "DISCLAIMER": {
        "en": "Prototype for a student hackathon. Not medical advice and not a diagnosis.",
        "hi": "यह छात्रों के हैकाथॉन का एक प्रोटोटाइप है। यह चिकित्सीय सलाह या निदान नहीं है।",
        "ur": "یہ طلبہ کے ہیکاتھون کا تجرباتی نمونہ ہے۔ یہ طبی مشورہ یا تشخیص نہیں ہے۔",
    },
}


def slip_html(decision: str, langs=DEFAULT_LANGS) -> str:
    """The printable note for the doctor and patient, one block per language (Urdu right-to-left)."""
    blocks = []
    for code in langs:
        name, direction = LANGS[code]["name"], LANGS[code]["dir"]
        blocks.append(f"<div dir='{direction}' lang='{code}' style='margin:6px 0;line-height:1.5'>"
                      f"<b>{name}:</b> {REPORT[decision][code]}<br>"
                      f"<span style='opacity:0.7;font-size:0.9em'>{REPORT['DISCLAIMER'][code]}</span></div>")
    return "".join(blocks)


@dataclass
class Frozen:
    """Everything the clinic tool needs, fixed before the test sets were scored."""
    arch: str
    size: int
    ref_threshold: float
    poor_threshold: float
    target_sensitivity: float
