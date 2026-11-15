import os
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import drscreen as ds  # noqa: E402


def test_threshold_reaches_target_sensitivity():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(0.3 * y + rng.normal(0.35, 0.2, 500), 0, 1)
    for target in (0.8, 0.9, 0.95):
        t = ds.threshold_for_sensitivity(y, p, target)
        sens = (p[y == 1] >= t).mean()
        assert sens >= target
        # and it is the highest such threshold: nudging it up drops below target
        assert (p[y == 1] >= t + 1e-9).mean() < target or t == p[y == 1].min()


def test_decide_never_blocks_a_referral_and_never_clears_a_bad_photo():
    d = ds.decide(p_ref=[0.9, 0.9, 0.1, 0.1], p_poor=[0.9, 0.1, 0.9, 0.1], ref_thr=0.5, poor_thr=0.5)
    assert d.tolist() == ["REFER", "REFER", "RETAKE", "RESCREEN"]


def test_workflow_counts():
    w = ds.workflow_report([0, 2, 3, 4, 1], ["RESCREEN", "RESCREEN", "RETAKE", "REFER", "REFER"])
    assert w["referable_given_all_clear"] == 1 and w["referable_caught_by_retake"] == 1
    assert w["sight_threatening_given_all_clear"] == 0


def test_binary_report_counts():
    r = ds.binary_report([1, 1, 0, 0, 1], [0.9, 0.2, 0.8, 0.1, 0.7], 0.5)
    assert (r["missed_FN"], r["false_alarms_FP"]) == (1, 1)
    assert r["sensitivity"] == pytest.approx(2 / 3) and r["specificity"] == pytest.approx(1 / 2)


def test_wilson_interval_contains_estimate():
    lo, hi = ds.wilson(45, 50)
    assert lo < 0.9 < hi and 0 <= lo and hi <= 1


def test_referable_prob_sums_moderate_and_worse():
    P = np.array([[0.5, 0.2, 0.1, 0.1, 0.1]])
    assert ds.referable_prob(P)[0] == pytest.approx(0.3)


def test_duplicates_grouped_and_split_kept_together(tmp_path):
    rng = np.random.default_rng(1)
    paths = []
    for k in range(40):
        a = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
        p = tmp_path / f"img{k}.jpg"
        Image.fromarray(a).save(p)
        paths.append(str(p))
    # make image 40 a resized, recompressed copy of image 3
    Image.open(paths[3]).resize((80, 80)).save(tmp_path / "copy.jpg", quality=70)
    paths.append(str(tmp_path / "copy.jpg"))
    groups, pairs = ds.duplicate_groups(paths, max_dist=20)
    assert groups[3] == groups[40]
    labels = np.array([k % 5 for k in range(41)])
    split = ds.group_split(labels, groups, seed=0)
    assert split[3] == split[40]


def test_prep_handles_black_and_odd_shapes():
    assert ds.prep_image(Image.new("RGB", (300, 200)), 64).size == (64, 64)
    img = Image.new("RGB", (400, 300), (0, 0, 0))
    img.paste((180, 80, 40), (100, 50, 300, 250))
    out = ds.prep_image(img, 64)
    assert out.size == (64, 64) and np.asarray(out).mean() > 50   # border cropped away


def test_reports_exist_in_every_language():
    for k in ("REFER", "RETAKE", "RESCREEN", "DISCLAIMER"):
        for code in ds.LANGS:
            assert ds.REPORT[k][code].strip(), (k, code)
        assert set(ds.REPORT[k]) == set(ds.LANGS)
    html = ds.slip_html("RETAKE")  # default for the Pakistan setting: English and Urdu
    assert "lang='en'" in html and "lang='ur'" in html and "dir='rtl'" in html and "lang='hi'" not in html
    assert "ru" not in ds.LANGS                       # removed: no native Russian speaker on the team
    for code, v in ds.LANGS.items():                  # every language names a reviewer and a checked flag
        assert v["reviewer"] and isinstance(v["checked"], bool) and v["dir"] in ("ltr", "rtl")
    # unchecked languages are hidden unless they are a default (defaults are enforced by check_submission.py)
    assert set(ds.shown_langs()) == {c for c, v in ds.LANGS.items() if v["checked"]} | set(ds.DEFAULT_LANGS)
    assert all("12" in ds.REPORT["RESCREEN"][c] for c in ds.LANGS)


def test_fill_report_formats_and_refuses_gaps():
    import fill_report as fr
    assert fr.fmt(0.91234, "pct") == "91.2%" and fr.fmt(0.9, "pct0") == "90%"
    assert fr.fmt([0.81, 0.86], "ci") == "0.810 to 0.860" and fr.fmt(1234, "int") == "1,234"
    R = {"a": {"b": [{"c": 0.5}]}, "table": {}}
    text, missing = fr.fill("x [FROM_RESULTS_JSON: a.b.0.c | .2f] y [FROM_RESULTS_JSON: a.zz]", R)
    assert "0.50" in text and missing and "a.zz" in missing[0]


def _fake_fundus(path, w=300, h=260, seed=0):  # noqa: E302
    rng = np.random.default_rng(seed)
    a = np.zeros((h, w, 3), np.uint8)
    yy, xx = np.mgrid[:h, :w]
    disc = (yy - h / 2) ** 2 + (xx - w / 2) ** 2 < (min(h, w) / 2.2) ** 2
    a[disc] = (rng.integers(120, 200), rng.integers(40, 90), rng.integers(10, 40))
    Image.fromarray(a).save(path)
    return str(path)


def test_triage_one_matches_decide_and_cpu_benchmark_runs(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    torch.manual_seed(0)
    model = ds.build_model("efficientnet_b0", pretrained=False)
    paths = [_fake_fundus(tmp_path / f"{i}.jpg", seed=i) for i in range(4)]
    rng = np.random.default_rng(0)
    gate = ds.fit_logreg(rng.normal(size=(40, 1280)), np.r_[np.zeros(20), np.ones(20)], C=0.1)
    d, p_ref, p_poor, P, e = ds.triage_one(model, gate, paths[0], 64, 0.5, 0.5)
    assert d == ds.decide([p_ref], [p_poor], 0.5, 0.5)[0]
    assert abs(P.sum() - 1) < 1e-5 and e.shape == (1, 1280) and 0 <= p_poor <= 1
    assert abs(p_ref - P[2:].sum()) < 1e-6
    # a referral threshold of 0 must always refer, whatever the photo quality
    assert ds.triage_one(model, gate, paths[1], 64, 0.0, 0.0)[0] == "REFER"
    cs = ds.cpu_benchmark(model, gate, paths, 64, 0.5, 0.5, n=3, heatmaps=1)
    assert cs["photos_timed"] == 3 and cs["median_seconds_per_photo"] > 0 and cs["cpu_threads"] >= 1
    assert cs["parameters"] > 1_000_000


def test_degradations_are_fixed_and_change_the_photo():
    img = Image.open(_fake_fundus("/tmp/_deg.jpg", 384, 384)).convert("RGB")
    conds = ds.stress_conditions()
    assert conds[0] == ("clean", "none", None) and len(conds) == 1 + 2 * len(ds.DEGRADATIONS)
    for kind, strength, level in conds[1:]:
        a, b = ds.degrade(img, kind, level), ds.degrade(img, kind, level)
        assert a.size == img.size and np.array_equal(np.asarray(a), np.asarray(b))   # repeatable
        assert not np.array_equal(np.asarray(a), np.asarray(img)), (kind, strength)  # actually worse
    dark = np.asarray(ds.degrade(img, "underexposed", 0.35)).mean()
    assert dark < np.asarray(img).mean()


def test_web_resize_and_rules_match_python(tmp_path):
    import shutil
    import subprocess
    if shutil.which("node") is None:
        pytest.skip("node not installed")
    rng = np.random.default_rng(0)
    cases = []
    for sw, sh, dw, dh in [(1800, 1800, 384, 384), (777, 777, 384, 384), (300, 300, 384, 384), (1001, 640, 96, 200)]:
        a = rng.integers(0, 256, (sh, sw, 3), dtype=np.uint8)
        a.tofile(f"/tmp/rs_{sw}x{sh}.raw")
        np.asarray(Image.fromarray(a).resize((dw, dh), Image.BILINEAR)).tofile(f"/tmp/rs_{sw}x{sh}_to_{dw}x{dh}.raw")
        cases.append(f"{sw},{sh},{dw},{dh}")
    here = os.path.dirname(os.path.abspath(__file__))
    r = subprocess.run(["node", os.path.join(here, "test_web.mjs")] + cases, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_team_inputs_fill_markers_and_optional_blocks(tmp_path):
    import fill_report as fr
    p = tmp_path / "TEAM_INPUTS.md"
    p.write_text("# x\n<!-- instructions -->\n## where_it_fails\nPhoto A was blurred.\n\n## problem\n<!-- only a comment -->\n"
                 "## adoption\nA hypothesis.\n", encoding="utf-8")
    team = fr.read_team(str(p))
    assert team == {"where_it_fails": "Photo A was blurred.", "adoption": "A hypothesis."}   # empty sections are ignored
    tpl = ("[TEAM_TO_WRITE(where_it_fails): say why] [TEAM_TO_WRITE(ai_tools): list]\n"
           "<!-- team:problem -->Default problem.<!-- /team:problem -->\n<!-- team:adoption --><!-- /team:adoption -->")
    out, used = fr.apply_team(tpl, team)
    assert "Photo A was blurred." in out and "[TEAM_TO_WRITE(ai_tools)" in out      # filled vs still waiting
    assert "Default problem." in out and "A hypothesis." in out and "<!-- team" not in out
    assert used == {"where_it_fails", "adoption"}


def test_local_calibration_keeps_patients_apart_and_scores_workflow(tmp_path):
    import json
    import pandas as pd
    import local_calibration as lcm
    rng = np.random.default_rng(1)
    rows = []
    for part, pats in (("training", range(0, 60)), ("validation", range(100, 130))):
        for pid in pats:
            for k in range(4):
                g = int(rng.integers(0, 5))
                rows.append(dict(patient_id=pid, image_id=f"{pid}_{k}", grade=g, part=part, Source="x", poor_quality=0,
                                 p_referable=float(np.clip(0.2 * g + rng.normal(0, 0.1), 0, 1)), p_poor=0.1))
    pd.DataFrame(rows).to_csv(tmp_path / "deepdrid_predictions.csv", index=False)
    json.dump({"frozen": {"ref_threshold": 0.9, "poor_threshold": 0.5, "target_sensitivity": 0.9}}, open(tmp_path / "results.json", "w"))
    out = lcm.run(str(tmp_path), draws=20)
    assert out["evaluation_patients"] == 30
    assert out["frozen"]["threshold"] == 0.9
    s = out["samples"][0]
    assert 0 <= s["sensitivity"]["median"] <= 1 and s["draws"] > 0
    assert s["sensitivity"]["median"] >= out["frozen"]["sensitivity"]   # a lower, locally set cut-off catches more
    assert (tmp_path / "local_calibration.png").exists()


def test_oiaddr_loader_reads_txt_csv_and_class_folders(tmp_path):
    import pandas as pd
    from PIL import Image
    import run_dr
    img = Image.new("RGB", (32, 32), (120, 60, 30))
    (tmp_path / "A" / "train").mkdir(parents=True)
    (tmp_path / "B").mkdir()
    lines = []
    for i in range(120):
        img.save(tmp_path / "A" / "train" / f"p{i}.jpg")
        lines.append(f"p{i}.jpg {i % 6}")
        d = tmp_path / "C" / str(i % 6)
        d.mkdir(parents=True, exist_ok=True)
        img.save(d / f"q{i}.jpg")
        img.save(tmp_path / "B" / f"r{i}.jpg")
    (tmp_path / "A" / "train.txt").write_text("\n".join(lines))
    pd.DataFrame({"id_code": [f"r{i}" for i in range(120)], "diagnosis": [i % 6 for i in range(120)]}).to_csv(tmp_path / "B" / "labels.csv", index=False)
    for sub in ("A", "B", "C"):
        df = run_dr.load_oiaddr(str(tmp_path / sub), log=lambda *a: None)
        assert len(df) == 100 and df.grade.between(0, 4).all()    # grade 5 (ungradable) dropped
