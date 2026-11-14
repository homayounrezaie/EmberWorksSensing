"""QA figures (matplotlib).  Colours follow the web page: discovered = teal,
FlamMap MTT = orange, the other physics models in their own hues, no-spread = grey."""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

C = dict(physics="#eb6834", discovered="#1baf7a", calibrated="#2a78d6", baseline="#898781",
         ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7", surface="#fcfcfb")
# one colour per forecast model (fig_metrics)
MODEL_C = dict(discovered="#1baf7a", flammap="#eb6834", physics="#2a78d6", fbp="#8a5cd1", persist="#b8860b", baseline="#898781")


def style():
    plt.rcParams.update({
        "figure.facecolor": C["surface"], "axes.facecolor": C["surface"], "savefig.facecolor": C["surface"],
        "axes.edgecolor": C["axis"], "axes.labelcolor": C["ink2"], "axes.titlecolor": C["ink"],
        "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "xtick.color": C["muted"], "ytick.color": C["muted"],
        "font.family": "sans-serif", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold",
        "lines.linewidth": 2, "legend.frameon": False})


def crop_box(U, margin=20):
    r, c = np.where(U)
    return (max(r.min() - margin, 0), min(r.max() + margin, U.shape[0]),
            max(c.min() - margin, 0), min(c.max() + margin, U.shape[1]))


def fig_threshold_sensitivity(sens: pd.DataFrame, tmin: np.ndarray, base_delta=250, base_active=2000, unit="W/m²",
                              delta_label="Δ > {:g} W/m²", delta_title="Burnt area vs. rise-over-baseline threshold"):
    style()
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    panels = [(axs[0], "delta", base_delta, delta_label, "burnt_ha", "cumulative burnt area (ha)", delta_title),
              (axs[1], "active", base_active, "> {:g} " + unit, "active_ha", "actively flaming area (ha)",
               "Flaming area vs. active-fire threshold")]
    for ax, kind, base, unit, col_name, ylab, title in panels:
        d = sens[sens.kind == kind]
        thrs = sorted(d.thr.unique())
        blues = plt.cm.Blues(np.linspace(0.35, 0.9, len(thrs)))
        for col, t in zip(blues, thrs):
            s = d[d.thr == t].sort_values("pass_id")
            ax.plot(tmin[s.pass_id], s[col_name], color=C["ink"] if t == base else col,
                    lw=2.4 if t == base else 1.2, label=unit.format(t) + ("  (default)" if t == base else ""))
            if kind == "delta" and t in (thrs[0], base, thrs[-1]):  # direct-label extremes + default only
                ax.annotate(f"{t:g}", (tmin[s.pass_id].max(), s[col_name].iloc[-1]), xytext=(4, 0),
                            textcoords="offset points", fontsize=8, color=C["ink2"], va="center")
        ax.set_xlabel("minutes after first image"); ax.set_ylabel(ylab); ax.set_title(title)
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def fig_front_qa(m, F: pd.DataFrame, pass_ids, box, grid_res=3.0, img_lo=300, img_hi=3e4, ros_max=0.3):
    """LWIR at t, burnt(t) and burnt(t+1) outlines, front points by ROS / flag."""
    style()
    r0, r1, c0, c1 = box
    n = len(pass_ids)
    ncol = 2 if n > 1 else 1
    nrow = int(np.ceil(n / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(8 * ncol, 7.2 * nrow), constrained_layout=True)
    axs = np.atleast_1d(axs).ravel()
    for ax, i in zip(axs, pass_ids):
        E, b0, b1 = m["E"][i], m["burnt"][i], m["burnt"][i + 1]
        ax.imshow(np.clip(E[r0:r1, c0:c1], img_lo, img_hi), cmap="inferno", norm=LogNorm(img_lo, img_hi), interpolation="nearest")
        ax.contour(b0[r0:r1, c0:c1], [0.5], colors="white", linewidths=0.8)
        ax.contour(b1[r0:r1, c0:c1], [0.5], colors="#7fdbff", linewidths=0.8)
        f = F[F.pass_id == i]
        bad = f[f.active & ~f.clean]
        ax.scatter(bad.col - c0, bad.row - r0, s=6, c="#bbbbbb", marker="x", lw=0.6, label="active, excluded")
        cl = f[f.clean]
        sc = ax.scatter(cl.col - c0, cl.row - r0, s=8, c=cl.ros, cmap="viridis", vmin=0, vmax=ros_max, label="clean (ROS)")
        ax.set_title(f"pass {i}→{i + 1}: clean {len(cl)}, excluded {len(bad)}")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.colorbar(sc, ax=axs, shrink=0.7, label="ROS (m/s)")
    axs[0].legend(loc="lower left", fontsize=8, markerscale=3, labelcolor="white")
    return fig


def fig_ros_vs_theta(F: pd.DataFrame):
    style()
    c = F[F.clean]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
    ax = axs[0]
    ax.scatter(c.theta, c.ros, s=2, alpha=0.15, color=C["ink2"], rasterized=True)
    bins = np.arange(-180, 181, 20)
    mid = 0.5 * (bins[1:] + bins[:-1])
    g = c.groupby(pd.cut(c.theta, bins), observed=False).ros
    ax.plot(mid, g.median().values, color=C["ink"], lw=2, label="median")
    ax.plot(mid, g.quantile(0.9).values, color=C["ink"], lw=1, ls="--", label="90th percentile")
    ax.set_xlabel("θ: front normal relative to wind (deg, 0 = head, ±180 = back)")
    ax.set_ylabel("ROS (m/s)"); ax.set_ylim(0, 0.5); ax.set_xticks(range(-180, 181, 60))
    ax.set_title(f"Measured ROS vs. angle to wind (clean points, n={len(c)})"); ax.legend()
    ax = axs[1]
    for ph, col in [("ignition", C["calibrated"]), ("burnout", C["physics"])]:
        d = c[c.phase == ph]
        gg = d.groupby(pd.cut(np.cos(np.radians(d.theta)), np.linspace(-1, 1, 11)), observed=False).ros
        x = np.linspace(-0.9, 0.9, 10)
        ax.plot(x, gg.mean().values, "-o", color=col, ms=4, lw=1.5, label=f"{ph} phase (n={len(d)})")
    ax.set_xlabel("cos θ"); ax.set_ylabel("mean ROS (m/s)")
    ax.set_title("Mean ROS vs. cos θ, by burn phase"); ax.legend()
    fig.tight_layout()
    return fig


def fig_coef_stability(boot: pd.DataFrame, beta, active, by_split: pd.DataFrame | None = None):
    """Left: block-bootstrap inclusion frequency and 5-95 % coefficient range
    (default split).  Right: coefficients across train/test splits."""
    style()
    ncol = 2 if by_split is not None else 1
    fig, axs = plt.subplots(1, ncol, figsize=(6.5 * ncol, 4.6))
    axs = np.atleast_1d(axs)
    ax = axs[0]
    y = np.arange(len(boot))[::-1]
    ax.barh(y, boot.inclusion, color=[C["discovered"] if a else C["axis"] for a in active], height=0.6)
    ax.axvline(0.6, color=C["ink2"], lw=1, ls="--")
    ax.set_yticks(y, boot.term); ax.set_xlim(0, 1.05); ax.set_xlabel("share of bootstrap refits keeping the term")
    ax.set_title("Term stability (block bootstrap, training data)")
    for yy, inc, b, a, lo, hi in zip(y, boot.inclusion, beta, active, boot.p05, boot.p95):
        txt = f"{b:+.4f}  [{lo:+.3f}, {hi:+.3f}]" if a else "dropped"
        ax.text(min(inc, 1) + 0.02 if inc < 0.55 else 0.02, yy, txt, va="center", fontsize=8,
                color=C["ink"] if inc >= 0.55 else C["ink2"])
    if by_split is not None:
        ax = axs[1]
        terms = [t for t in by_split.columns if t != "k" and (by_split[t] != 0).any()]
        cols = [C["discovered"], C["physics"], C["calibrated"], "#e87ba4", "#6250d6", "#eda100", "#008300", "#e34948"]
        for t, col in zip(terms, cols):
            ax.plot(by_split.k, by_split[t], "-o", ms=4, color=col, lw=1.6, label=t)
        ax.axhline(0, color=C["axis"], lw=1)
        ax.set_xlabel("fire overpasses used for training (k)"); ax.set_ylabel("coefficient (original units)")
        ax.set_title("Kept coefficients across train/test splits"); ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def fig_metrics(mt: pd.DataFrame, test_start: int, names: dict | None = None):
    """Per-interval scores of every model; the shaded band is the training period."""
    style()
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.2))
    names = names or {}
    models = [mm for mm in MODEL_C if mm in set(mt.model)]
    for ax, (key, lab) in zip(axs, [("iou_new", "New-area IoU (higher is better)"), ("front_mean_m", "Mean front error, m (lower is better)"),
                                    ("arrival_err_s", "Arrival-time error, s (lower is better)")]):
        ax.axvspan(mt.pass_id.min() + 1 - 0.5, test_start + 0.5, color="#f3f2ee", zorder=0)
        for model in models:
            d = mt[mt.model == model].sort_values("pass_id")
            ax.plot(d.pass_id + 1, d[key], "-o", ms=3.5, color=MODEL_C[model], ls="--" if model == "baseline" else "-",
                    lw=2.2 if model == "discovered" else 1.4, label=names.get(model, model))
        ax.set_title(lab); ax.set_xlabel("forecast target (snapshot index)")
        ax.text(mt.pass_id.min() + 1, ax.get_ylim()[1], " training", va="top", fontsize=8, color=C["muted"])
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_inputs(L: dict, unit: np.ndarray):
    """Model-input layers on the analysis grid (pipeline/landscape.py)."""
    style()
    fig, ax = plt.subplots(1, 5, figsize=(22, 4.6))
    out = ~unit.astype(bool)

    def show(a, d, title, cmap, **kw):
        im = a.imshow(np.ma.masked_where(out, d), cmap=cmap, **kw)
        a.set_title(title); a.set_xticks([]); a.set_yticks([]); fig.colorbar(im, ax=a, shrink=0.8)

    show(ax[0], L["T_fuel"], "pre-fire surface temperature (°C)", "inferno")
    show(ax[1], L["m1"], "1-h dead fuel moisture (fraction)", "Blues",
         vmin=np.percentile(L["m1"][unit], 5), vmax=np.percentile(L["m1"][unit], 99))
    show(ax[2], L["elev"], "elevation (m), arrows = upslope", "terrain")
    yy, xx = np.mgrid[5:L["elev"].shape[0]:12, 5:L["elev"].shape[1]:12]
    t, up = L["tan_slope"][yy, xx], np.radians(L["up_az"][yy, xx])
    ax[2].quiver(xx, yy, t * np.sin(up), t * np.cos(up), color="k", scale=1.2)
    names = L["meta"]["fuel_names"]
    show(ax[3], L["fuel"].astype(float), "LANDFIRE fuel model: " + ", ".join(f"{i}={n}" for i, n in enumerate(names)), "Set2",
         vmin=0, vmax=max(len(names) - 1, 1))
    show(ax[4], L["waf"], "wind adjustment factor (canopy)", "viridis")
    fig.tight_layout()
    return fig
