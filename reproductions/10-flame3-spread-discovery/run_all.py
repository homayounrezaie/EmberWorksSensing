#!/usr/bin/env python3
"""Run the whole FLAME 3 Hanna Hammock pipeline:
download -> ingest -> fronts -> landscape (model inputs: thermal fuel moisture, 3DEP terrain,
LANDFIRE fuels / canopy) -> moisture (satellite live fuel moisture, FWI codes, checks of the thermal
moisture) -> fit/forecast/score (+ ablation) -> export JSON + web/index.html -> QA figures.

Every stage is skipped when its outputs already exist (downloads are cached
per file), so an interrupted run resumes where it stopped.  Use --force to
redo stages, e.g. `python run_all.py --force fronts evaluate export`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.config import Config  # noqa: E402
from pipeline.sites import SITE  # noqa: E402

STAGES = ["download", "ingest", "fronts", "landscape", "moisture", "evaluate", "export", "qa"]


def outputs(cfg: Config):
    i = cfg.interim_dir
    return {
        "download": [cfg.raw_dir / f".download_complete_{cfg.site}"],
        "ingest": [i / "frames.npz", i / "grid.json"],
        "fronts": [i / "masks.npz", i / "front_points.csv", i / "wind_intervals.csv"],
        "landscape": [i / "landscape.npz", i / "landscape.json", i / "moisture_t.npz"],
        "moisture": [i / "moisture.json", i / "fwi_ktlh.csv"],
        "evaluate": [i / "results_default.json", i / "metrics_default.csv", i / "ablation.csv"],
        "export": [cfg.web_dir / "data" / SITE["data"], cfg.web_dir / SITE["page"]],
        "qa": [cfg.qa_dir / "step5_metrics.png"],
    }


def run_stage(name: str, cfg: Config):
    from pipeline import flame3
    if name == "download":
        flame3.download(cfg)
    elif name == "ingest":
        flame3.ingest(cfg)
    elif name == "fronts":
        flame3.report(flame3.build(cfg))
    elif name == "landscape":
        from pipeline import landscape
        landscape.download(cfg)
        landscape.report(landscape.build(cfg))
        landscape.moisture_series(cfg)
    elif name == "moisture":
        from pipeline import moisture
        moisture.download(cfg)
        moisture.report(moisture.build(cfg))
    elif name == "evaluate":
        evaluate(cfg)
    elif name == "export":
        from pipeline import export
        export.build(cfg)
        print(export.build_html(cfg))
    elif name == "qa":
        make_qa(cfg)


def evaluate(cfg: Config):
    from pipeline import evaluate as EV
    from pipeline.discover import terms_for
    r = EV.fit_split(cfg, cfg.train_fire_passes, verbose=True)
    mt = r["metrics"]
    mt.to_csv(cfg.interim_dir / "metrics_default.csv", index=False)
    cols = ["iou", "iou_new", "front_mean_m", "front_p90_m", "arrival_err_s"]
    summary = dict(
        k=r["k"], train_intervals=r["train"], test_intervals=r["test"], lam=r["lam"],
        terms=[t[1] for t in terms_for(cfg.terms)], beta=[float(b) for b in r["beta"]], active=[bool(a) for a in r["active"]],
        bootstrap_inclusion=[float(v) for v in r["boot"].inclusion], r2_train=r["r2_train"], r2_test=r["r2_test"],
        n_train=r["n_train"], n_test=r["n_test"], lambda_criterion=cfg.lambda_criterion, sign_priors=cfg.sign_priors,
        reference=dict(U20=r["U_ref"], U10=r["U10_ref"], wd_from=r["wd_ref"], FRFD=r["I_ref"]), ref_inputs=r["ref_inputs"],
        point_skill=r["point_skill"], observed=r["observed"], significance=r["significance"],
        test_means=mt[mt.split == "test"].groupby("model")[cols].mean().to_dict("index"),
        explanations=r["explanations"])
    (cfg.interim_dir / "results_default.json").write_text(json.dumps(summary, indent=1, default=float))
    ab = EV.ablation(cfg)
    print("ablation (discovered equation, test means):\n", ab.drop(columns="equation").round(3).to_string(index=False))


def make_qa(cfg: Config):
    import numpy as np
    import pandas as pd
    from pipeline import evaluate as EV, plots

    m = np.load(cfg.interim_dir / "masks.npz")
    F = pd.read_csv(cfg.interim_dir / "front_points.csv")
    ov = pd.read_csv(cfg.interim_dir / "overpasses.csv")
    tmin = (ov.t_unit_mean.values - ov.t_unit_mean[0]) / 60
    sens = pd.read_csv(cfg.interim_dir / "threshold_sensitivity.csv")
    cfg.qa_dir.mkdir(parents=True, exist_ok=True)
    plots.fig_threshold_sensitivity(sens, tmin, cfg.thr_delta, cfg.thr_active, unit="°C", delta_label="ever > {:g} °C",
                                    delta_title="Burnt area vs. 'hot' threshold").savefig(cfg.qa_dir / "step2_threshold_sensitivity.png", dpi=90)
    plots.fig_front_qa(m, F, [24, 27, 30, 33], plots.crop_box(m["unit"]), grid_res=cfg.grid_res_m,
                       img_lo=20, img_hi=600, ros_max=0.2).savefig(cfg.qa_dir / "step2_fronts.png", dpi=60)
    plots.fig_ros_vs_theta(F[F.pass_id >= cfg.first_snapshot]).savefig(cfg.qa_dir / "step2_ros_vs_theta.png", dpi=90)
    mt = pd.read_csv(cfg.interim_dir / "metrics_default.csv")
    plots.fig_metrics(mt, cfg.first_snapshot + cfg.train_fire_passes - 1, EV.MODEL_NAMES).savefig(cfg.qa_dir / "step5_metrics.png", dpi=90)
    from pipeline import landscape
    plots.fig_inputs(landscape.load(cfg), m["unit"]).savefig(cfg.qa_dir / "step3_inputs.png", dpi=80)
    print("QA figures ->", cfg.qa_dir)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", nargs="*", default=[], choices=STAGES + ["all"], help="stages to redo")
    a = ap.parse_args()
    force = set(STAGES) if "all" in a.force else set(a.force)
    cfg = Config()
    t_all = time.time()
    print(f"===== FLAME 3 {cfg.site} ({cfg.date})", flush=True)
    for st in STAGES:
        done = all(Path(p).exists() for p in outputs(cfg)[st])
        if done and st not in force:
            print(f"[{st}] up to date — skipped")
            continue
        t0 = time.time()
        print(f"[{st}] running…", flush=True)
        run_stage(st, cfg)
        print(f"[{st}] done in {time.time() - t0:.0f} s", flush=True)
    print(f"all stages finished in {time.time() - t_all:.0f} s")


if __name__ == "__main__":
    main()
