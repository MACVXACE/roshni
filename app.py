"""Roshni demo (optional Gradio version; the main demo is the browser version in web/): upload a retinal photo,
get REFER / RETAKE / RESCREEN with a note in English and Urdu.

  python app.py --results results [--share]

Prototype for a student hackathon. Not medical advice and not a diagnosis.
"""
import argparse
import glob
import json
import os
import pickle

import gradio as gr
import numpy as np
import pandas as pd
import torch

import drscreen as ds

COLORS = {"REFER": "#b42318", "RETAKE": "#b54708", "RESCREEN": "#067647"}
TITLES = {"REFER": "Refer to an eye specialist", "RETAKE": "Retake the photo", "RESCREEN": "Rescreen in 12 months"}


def load_bundle(results_dir):
    with open(os.path.join(results_dir, "results.json")) as f:
        res = json.load(f)
    fr = res["frozen"]
    model = ds.build_model(fr["arch"], pretrained=False)
    model.load_state_dict(torch.load(os.path.join(results_dir, "model_best.pt"), map_location="cpu"))
    model.eval()
    gate = None
    gp = os.path.join(results_dir, "quality_gate.pkl")
    if os.path.exists(gp):
        with open(gp, "rb") as f:
            gate = pickle.load(f)
    summary = pd.read_csv(os.path.join(results_dir, "summary.csv")) if os.path.exists(os.path.join(results_dir, "summary.csv")) else None
    return model, gate, fr, summary, res


def build_app(results_dir: str, examples_dir: str | None = None) -> gr.Blocks:
    model, gate, fr, summary, res = load_bundle(results_dir)
    size, ref_thr, poor_thr = fr["size"], fr["ref_threshold"], fr["poor_threshold"]
    tf = ds.get_transforms(size, False)

    def screen(img, langs=ds.DEFAULT_LANGS):
        langs = [c for c in ds.shown_langs() if c in (langs or [])] or ["en"]
        if img is None:
            return None, "Upload a retinal photo first.", None, ""
        d, p_ref, p_poor, P, _ = ds.triage_one(model, gate, img, size, ref_thr, poor_thr, tf)
        P = P[None, :]
        grades = {f"{k} {g}": float(P[0, k]) for k, g in enumerate(ds.GRADES)}
        card = (f"<div style='border-left:6px solid {COLORS[d]};padding:10px 14px;border-radius:6px;"
                f"background:rgba(0,0,0,0.03)'><div style='font-size:22px;font-weight:700;color:{COLORS[d]}'>"
                f"{TITLES[d]}</div><div style='margin-top:6px'>Referable-disease score {p_ref:.2f} "
                f"(refer at {ref_thr:.2f} or above) · Poor-photo score "
                + (f"{p_poor:.2f} (retake at {poor_thr:.2f} or above)" if gate is not None else "not available")
                + "</div></div>")
        note = ("<div style='font-weight:600;margin-bottom:4px'>Note for the doctor and patient</div>"
                + ds.slip_html(d, langs))
        try:
            heat = ds.gradcam(model, img, size, device="cpu")
        except Exception:
            heat = None
        return heat, card, grades, note

    ex = []
    if examples_dir and os.path.isdir(examples_dir):
        ex = sorted(glob.glob(os.path.join(examples_dir, "*.jpg")) + glob.glob(os.path.join(examples_dir, "*.png")))[:8]

    with gr.Blocks(title="Roshni: diabetic eye screening") as demo:
        gr.Markdown("# Roshni: offline diabetic eye triage for primary-care diabetes clinics in Pakistan\n"
                    "For a clinic that can take retinal photos but has no eye specialist on site. Upload one photo taken at a "
                    "diabetes check-up; the tool suggests one next step for the doctor to confirm: **refer**, **retake the photo**, "
                    "or **rescreen in 12 months**. A photo judged too poor to read gets a retake instead of an all-clear, and never blocks a referral.  \n"
                    f"_Thresholds were frozen before testing (referral threshold set for {fr['target_sensitivity']:.0%} sensitivity on validation). "
                    "Prototype for a student hackathon; not medical advice and not a diagnosis._")
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Retinal photo")
                langs = gr.CheckboxGroup([(ds.LANGS[k]["name"], k) for k in ds.shown_langs()], value=list(ds.DEFAULT_LANGS),
                                         label="Languages on the note")
                btn = gr.Button("Screen this photo", variant="primary")
                if ex:
                    gr.Examples(ex, inp, label="Example photos")
            with gr.Column():
                card = gr.HTML()
                grades = gr.Label(num_top_classes=5, label="Estimated severity")
                note = gr.HTML()
                heat = gr.Image(label="Where the model looked (Grad-CAM)", interactive=False)
        btn.click(screen, [inp, langs], [heat, card, grades, note])
        if summary is not None:
            gr.Markdown("## How it did on photos it never saw")
            gr.Dataframe(summary, interactive=False, wrap=True)
        if "workflow" in res:
            w = res["workflow"]["with_quality_gate"]
            wo = res["workflow"]["without_quality_gate"]
            gr.Markdown(f"On {w['photos']} external photos (DeepDRiD validation): {w['REFER']} refer, {w['RETAKE']} retake, "
                        f"{w['RESCREEN']} rescreen. Referable photos given an all-clear: **{w['referable_given_all_clear']}** "
                        f"with the photo-quality check vs **{wo['referable_given_all_clear']}** without it.")
    return demo


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--results", default="results")
    a.add_argument("--examples", default=None)
    a.add_argument("--share", action="store_true")
    x = a.parse_args()
    build_app(x.results, x.examples).launch(share=x.share)
