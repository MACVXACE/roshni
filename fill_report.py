"""Fill README.md and VIDEO_SCRIPT.md from the run's own output files, so no number is typed by hand.

  python fill_report.py                      # reads results/, writes README.md and VIDEO_SCRIPT.md
  python fill_report.py --results quick_check --allow-quick --out-dir /tmp/test   # testing only

Tokens in the templates look like  [FROM_RESULTS_JSON: results.fine_tuned.AUC | .3f]
  path   dotted path into results.json (list items by number), or derived.* / table.* computed here
  format .3f  .2f  pct (91.2%)  pct0 (91%)  int  ci (0.812 to 0.861)  cipct  str
Text written by people comes from TEAM_INPUTS.md (one "## key" section each), never from results:
  [TEAM_TO_WRITE(key): instruction]          required; filled from the "## key" section, listed if still empty
  <!-- team:key -->default<!-- /team:key -->   optional; the "## key" section replaces the default text
If any results token cannot be filled, nothing is written.
"""
import argparse
import json
import math
import os
import re
import sys

import pandas as pd

TOKEN = re.compile(r"\[FROM_RESULTS_JSON:\s*([^\]|]+?)\s*(?:\|\s*([^\]]+?)\s*)?\]")
TEAM = re.compile(r"\[TEAM_TO_WRITE(?:\((\w+)\))?:[^\]]*\]")
BLOCK = re.compile(r"<!-- team:(\w+) -->(.*?)<!-- /team:\1 -->", re.S)
# word limits for the team's pieces (from the official template and our plan); over-long text is reported, not cut
LIMITS = {"problem": 80, "where_it_fails": 55, "what_we_learned": 50, "next_steps": 60, "adoption": 120,
          "video_miss_reason": 12}
HEADINGS = {"adoption": "**Who would test it first (a hypothesis, not a finding)**\n\n"}


def read_team(path):
    """{key: text} from TEAM_INPUTS.md. HTML comments (the instructions) and empty sections are ignored."""
    if not path or not os.path.exists(path):
        return {}
    text = re.sub(r"<!--.*?-->", "", open(path, encoding="utf-8").read(), flags=re.S)
    parts = re.split(r"^## (\w+)\s*$", text, flags=re.M)
    return {k: v.strip() for k, v in zip(parts[1::2], parts[2::2]) if v.strip()}


def apply_team(text, team):
    used = set()

    def block(m):
        key, default = m.group(1), m.group(2)
        if key in team:
            used.add(key)
            return HEADINGS.get(key, "") + team[key]
        return default

    def marker(m):
        key = m.group(1)
        if key and key in team:
            used.add(key)
            return team[key]
        return m.group(0)
    return TEAM.sub(marker, BLOCK.sub(block, text)), used


class Missing(Exception):
    pass


def get(data, path):
    cur = data
    for part in path.split("."):
        if isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            raise Missing(path)
    if cur is None or (isinstance(cur, float) and math.isnan(cur)):
        raise Missing(path + " (empty)")
    return cur


def fmt(v, spec):
    if spec in (None, "", "str"):
        return f"{v:.3f}" if isinstance(v, float) else str(v)
    if spec == "pct":
        return f"{100 * float(v):.1f}%"
    if spec == "pct0":
        return f"{100 * float(v):.0f}%"
    if spec == "int":
        return f"{int(round(float(v))):,}"
    if spec in ("ci", "cipct"):
        a, b = (float(x) for x in v)
        if any(math.isnan(x) for x in (a, b)):
            raise Missing("interval is empty")
        return f"{100 * a:.1f}% to {100 * b:.1f}%" if spec == "cipct" else f"{a:.3f} to {b:.3f}"
    return format(float(v), spec)


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def build(res_dir):
    with open(os.path.join(res_dir, "results.json")) as f:
        R = json.load(f)
    r = R["results"]
    d = {}
    d["aptos_test_n"] = r["fine_tuned"]["n"]
    if "seconds" in R:
        d["run_minutes"] = round(R["seconds"] / 60)
    sp_p = os.path.join(res_dir, "aptos_split.csv")
    if os.path.exists(sp_p):
        c = pd.read_csv(sp_p)["split"].value_counts()
        d["split_counts"] = (f"{c.get('train', 0):,} training, {c.get('val', 0):,} validation and "
                             f"{c.get('test', 0):,} test photos")
    d["aptos_test_referable"] = r["fine_tuned"]["referable"]
    hist_p = os.path.join(res_dir, "training_history.csv")
    if os.path.exists(hist_p):
        h = pd.read_csv(hist_p)
        d["epochs_run"] = len(h)
        d["best_epoch"] = int(h.loc[h["val_auc"].idxmax(), "epoch"])
    fl = {row["grade"][0]: row for row in R.get("aptos_flagged_by_grade", [])}
    if "3" in fl and "4" in fl:
        d["severe_sent"], d["severe_total"] = fl["3"]["sent_to_specialist"], fl["3"]["photos"]
        d["prolif_sent"], d["prolif_total"] = fl["4"]["sent_to_specialist"], fl["4"]["photos"]
    if "external_all" in r:
        d["ext_photos"] = r["external_all"]["n"]
        dd_p = os.path.join(res_dir, "deepdrid_predictions.csv")
        if os.path.exists(dd_p):
            dd = pd.read_csv(dd_p)
            d["ext_patients"] = int(dd.loc[dd["grade"] <= 4, "patient_id"].nunique())
    if "workflow" in R:
        w, wo = R["workflow"]["with_quality_gate"], R["workflow"]["without_quality_gate"]
        d["val_photos"] = w["photos"]
        a, b = wo["referable_given_all_clear"], w["referable_given_all_clear"]
        if b > a:
            raise RuntimeError("Quality check increased all-clears for referable eyes; that should be impossible. Check the run.")
        d["gate_sentence"] = (f"cut referable eyes given an all-clear from {a} to {b}, at the cost of {w['RETAKE']} retakes"
                              if b < a else
                              f"did not change the number of referable eyes given an all-clear ({b}), and asked for {w['RETAKE']} retakes")
        d["gate_short"] = (f"cut false all-clears from {a} to {b}, for {w['RETAKE']} retakes" if b < a else
                           f"did not reduce false all-clears ({b}), and asked for {w['RETAKE']} retakes")
    cs = R.get("cpu_speed") or {}
    if "median_seconds_per_photo" in cs:
        d["cpu_short"] = f"about {cs['median_seconds_per_photo']:.2f} s per photo on a CPU, with no graphics card"
        d["cpu_sentence"] = (f"on the CPU alone ({cs['cpu']}, {cs['cpu_threads']} threads, no graphics card), going from the "
                             f"original photo file to a decision took a median of {cs['median_seconds_per_photo']:.2f} s "
                             f"(slowest {cs['slowest_seconds_per_photo']:.2f} s, {cs['photos_timed']} photos); the heat map "
                             f"adds about {cs['heatmap_median_seconds']:.1f} s. The model file is {cs['model_file_MB']:.0f} MB "
                             f"({cs['parameters'] / 1e6:.1f} million parameters)")
    else:
        d["cpu_short"] = "speed on a CPU not measured in this run"
        d["cpu_sentence"] = "speed on a CPU was not measured in this run"
    st = pd.DataFrame(R.get("stress_test") or [])
    if len(st):
        name = "DeepDRiD validation" if (st.photos == "DeepDRiD validation").any() else "APTOS test"
        x = st[st.photos == name]
        clean, changed = x[x.change == "clean"].iloc[0], x[x.change != "clean"]
        worst = changed.loc[changed.sensitivity.idxmin()]
        wspec = changed.loc[changed.specificity.idxmin()]
        lab = lambda c: f"{c.change.replace('_', ' ')}, {c.strength}"
        w_lo, w_hi = int(changed.referable_all_clear_with_check.min()), int(changed.referable_all_clear_with_check.max())
        wo_hi = int(changed.referable_all_clear_without_check.max())
        d["stress_sentence"] = (
            f"on the {int(clean.n)} {name} photos, sensitivity was {100 * clean.sensitivity:.1f}% on the originals and "
            f"{100 * worst.sensitivity:.1f}% at worst ({lab(worst)}). With the quality check, referable eyes given an all-clear "
            f"were {int(clean.referable_all_clear_with_check)} on the originals and between {w_lo} and {w_hi} across the "
            f"{len(changed)} changes, against up to {wo_hi} without it; the check sent up to "
            f"{100 * changed.retake_share.max():.0f}% of photos to a retake. Strong compression failed the other way: "
            f"specificity fell to {100 * wspec.specificity:.1f}% ({lab(wspec)}), so many healthy eyes would be referred")
        d["stress_brief"] = (f"sensitivity on the {name} photos fell from {100 * clean.sensitivity:.1f}% to "
                             f"{100 * worst.sensitivity:.1f}% ({lab(worst)}), yet with the quality check false all-clears "
                             f"stayed at {w_hi} or fewer ({int(clean.referable_all_clear_with_check)} on the originals), against "
                             f"up to {wo_hi} without it; strong compression instead cut specificity to {100 * wspec.specificity:.1f}%")
        d["stress_short"] = (f"On deliberately worse photos: at most {w_hi} false all-clears with the check, "
                             f"up to {wo_hi} without it.")
        T_stress = table([[("original" if c.change == "clean" else f"{c.change.replace('_', ' ')}, {c.strength}"), c.photos,
                           f"{100 * c.sensitivity:.1f}%", f"{100 * c.specificity:.1f}%", f"{100 * c.retake_share:.0f}%",
                           int(c.referable_all_clear_without_check), int(c.referable_all_clear_with_check)]
                          for c in st.itertuples()],
                         ["Photos", "Set", "Sensitivity", "Specificity", "Sent to retake",
                          "Referable all-clears, no check", "Referable all-clears, with check"])
    else:
        d["stress_sentence"] = "the stress test was not run in this run"
        d["stress_short"] = ""
        d["stress_brief"] = "not run in this run"
        T_stress = "_Stress test not run in this run._"
    ft, cb, lp = r["fine_tuned"], r["colour_baseline"], r["linear_probe"]
    d["baseline_sentence"] = (f"with an AUC of {ft['AUC']:.2f}, against {cb['AUC']:.2f} for a colour baseline "
                              f"and {lp['AUC']:.2f} for untuned features")

    T = {}
    rows = []
    for k in ("colour_baseline", "linear_probe", "fine_tuned", "external_all"):
        if k in r:
            x = r[k]
            rows.append([x["model"], f"{x['AUC']:.3f} ({x['AUC_CI'][0]:.3f} to {x['AUC_CI'][1]:.3f})",
                         f"{100 * x['sensitivity']:.1f}%", f"{100 * x['specificity']:.1f}%",
                         x["missed_FN"], x["false_alarms_FP"]])
    T["main"] = table(rows, ["Model", "AUC (95% CI)", "Sensitivity", "Specificity", "Missed (FN)", "False alarms (FP)"])
    if R.get("aptos_flagged_by_grade"):
        T["by_grade"] = table([[g["grade"], g["photos"], g["sent_to_specialist"], f"{100 * g['share_sent']:.0f}%"]
                               for g in R["aptos_flagged_by_grade"]], ["Grade", "Photos", "Sent to specialist", "Share sent"])
    if R.get("external_by_source"):
        T["by_source"] = table([[s["source"], s["photos"], f"{100 * s['sensitivity']:.1f}%", f"{100 * s['specificity']:.1f}%",
                                 s["missed_FN"]] for s in R["external_by_source"]],
                               ["Source", "Photos", "Sensitivity", "Specificity", "Missed (FN)"])
    if "workflow" in R:
        w, wo = R["workflow"]["with_quality_gate"], R["workflow"]["without_quality_gate"]
        names = [("REFER", "Refer"), ("RETAKE", "Retake"), ("RESCREEN", "Rescreen (all-clear)"),
                 ("referable_given_all_clear", "Referable eyes given an all-clear"),
                 ("sight_threatening_given_all_clear", "Grade 3 to 4 eyes given an all-clear")]
        T["workflow"] = table([[label, wo[k], w[k]] for k, label in names], ["", "Without quality check", "With quality check"])
    T["stress"] = T_stress
    R["derived"], R["table"] = d, T
    return R


def checks(R, res_dir, allow_quick):
    c = R["config"]
    problems = []
    if not allow_quick and (c.get("max_images") or not c.get("pretrained", True) or c.get("epochs", 0) < 5):
        problems.append("These results come from a quick or test run (max_images set, no pretrained weights, or under 5 epochs). "
                        "Use the full run, or pass --allow-quick for testing only.")
    sp = os.path.join(res_dir, "summary.csv")
    if os.path.exists(sp):
        s = pd.read_csv(sp)
        by_name = {v["model"]: v for v in R["results"].values()}
        for _, row in s.iterrows():
            v = by_name.get(row["model"])
            for k in ("AUC", "sensitivity", "specificity"):
                if v is not None and k in row and abs(float(row[k]) - float(v[k])) > 0.0011:
                    problems.append(f"summary.csv and results.json disagree on {row['model']} {k}")
    for k, v in R["results"].items():
        for m in ("AUC", "sensitivity", "specificity"):
            if not 0 <= float(v[m]) <= 1:
                problems.append(f"{k} {m} out of range: {v[m]}")
    return problems


def gates(R):
    r = R["results"]
    ft, cb, lp = r["fine_tuned"], r["colour_baseline"], r["linear_probe"]
    print("\nDiagnostics (report them whatever they say; they are not a reason to change project):")
    print(f"  A. Fine-tuned beats both baselines on AUC: {ft['AUC'] > max(cb['AUC'], lp['AUC'])} "
          f"({ft['AUC']:.3f} vs {cb['AUC']:.3f}, {lp['AUC']:.3f}); on specificity: {ft['specificity'] > max(cb['specificity'], lp['specificity'])}")
    if "external_all" in r:
        e = r["external_all"]
        print(f"  B. External AUC {e['AUC']:.3f}, patient-level 95% CI {e['AUC_CI_by_patient'][0]:.3f} to {e['AUC_CI_by_patient'][1]:.3f}")
    if "workflow" in R:
        w, wo = R["workflow"]["with_quality_gate"], R["workflow"]["without_quality_gate"]
        print(f"  C. Referable all-clears without/with quality check: {wo['referable_given_all_clear']} / {w['referable_given_all_clear']}; "
              f"ungradable (grade 5) photos in this set: {R['workflow']['ungradable_photos']}")


def fill(text, R):
    missing = []

    def sub(m):
        path, spec = m.group(1).strip(), (m.group(2) or "").strip() or None
        try:
            if path.startswith("table."):
                return R["table"][path.split(".", 1)[1]]
            return fmt(get(R, path), spec)
        except (Missing, KeyError, TypeError, ValueError) as e:
            missing.append(f"{path} ({e})")
            return m.group(0)
    return TOKEN.sub(sub, text), missing


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--results", default="results")
    a.add_argument("--allow-quick", action="store_true")
    a.add_argument("--out-dir", default=".")
    a.add_argument("--team", default="TEAM_INPUTS.md", help="the team's written pieces (optional)")
    x = a.parse_args()
    team = read_team(x.team)
    for k, v in team.items():
        n = len(v.split())
        if k in LIMITS and n > LIMITS[k]:
            print(f"Note: team piece '{k}' has {n} words; the limit is {LIMITS[k]}. Please shorten it.")
    R = build(x.results)
    problems = checks(R, x.results, x.allow_quick)
    if problems:
        print("STOPPED, nothing written:\n  " + "\n  ".join(problems))
        sys.exit(1)
    outputs, all_missing = {}, []
    for tpl, out in (("README.template.md", "README.md"), ("VIDEO_SCRIPT.template.md", "VIDEO_SCRIPT.md")):
        text, used = apply_team(open(tpl, encoding="utf-8").read(), team)
        text, missing = fill(text, R)
        outputs[out] = text
        all_missing += [f"{tpl}: {m}" for m in missing]
    if all_missing:
        print("STOPPED, nothing written. These tokens could not be filled:\n  " + "\n  ".join(all_missing))
        sys.exit(1)
    os.makedirs(x.out_dir, exist_ok=True)
    for out, text in outputs.items():
        with open(os.path.join(x.out_dir, out), "w", encoding="utf-8") as f:
            f.write(text)
        left = TEAM.findall(text)
        print(f"Wrote {out}" + (f"; still for the team to write: {len(left)}" if left else ""))
        for t in left:
            print("   ", t[:110])
    words = len(re.sub(r"<!--.*?-->|\*\(.*?\)\*|#.*", "", outputs["VIDEO_SCRIPT.md"]).split())
    print(f"Video script: about {words} spoken words (aim for 260 to 300).")
    gates(R)


if __name__ == "__main__":
    main()
