"""Fit, forecast and score for one train/test split (Hanna Hammock).

Split: the first `k` snapshots are training data (the intervals between them
supply ROS samples; the discovered equation and its threshold see nothing
else).  Every later interval is a test forecast: from the observed burnt
area + active front at snapshot i, predict snapshot i+1.  Every model gets
the same front, the same detected ignitions and the same spatial inputs
(pipeline/landscape.py): 1-h fuel moisture from the pre-fire thermal image,
3DEP slope / aspect, LANDFIRE fuel model and canopy (wind reduction), and
the airport wind.

Models
  discovered   sparse-regression (STLSQ) normal spread rate -> Huygens wavelets;
               candidate terms include moisture, slope, midflame wind, Rothermel R0
               and the previous local spread rate
  physics      Rothermel, per-cell ellipse (LANDFIRE fuel model, moisture, canopy-reduced
               wind and slope as vectors) -> Huygens wavelets (FARSITE method)
  flammap      the same per-cell Rothermel ellipses -> minimum travel time (FlamMap MTT method)
  fbp          Canadian FBP C-7, per-cell (FFMC from the moisture map, 10 m wind + slope-
               equivalent wind) -> Huygens wavelets
  persist      each front point keeps the spread rate observed at its location in the previous
               interval (isotropic) -- a data-only nowcasting baseline (takes no inputs)
  baseline     no spread (detected ignitions only)
"""
from __future__ import annotations

import json
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from . import discover as D
from . import forecast as FC
from . import landscape as LS
from . import moisture as MO
from . import physics as PH
from .config import Config
from .plots import crop_box

R_CAP = 1.0  # m/s, cap on any model's normal spread rate (keeps wavelets bounded)
LAMBDAS = np.round(np.r_[0, np.geomspace(0.001, 0.05, 24)], 5)
MODELS = ["discovered", "physics", "flammap", "fbp", "persist", "baseline"]
STATIC = ["physics", "flammap", "fbp", "persist", "baseline"]  # independent of the train/test split
MODEL_NAMES = {"discovered": "Discovered equation", "physics": "Rothermel + Huygens (FARSITE-style)",
               "flammap": "FlamMap MTT (Rothermel)", "fbp": "Canadian FBP C-7", "persist": "Persistence",
               "baseline": "No spread"}
U10_TO_U20 = 1.0 / 1.15  # 10 m open wind -> 20-ft wind (NWCG convention)


def load(cfg: Config):
    m = np.load(cfg.interim_dir / "masks.npz")
    F = pd.read_csv(cfg.interim_dir / "front_points.csv", low_memory=False)
    W = pd.read_csv(cfg.interim_dir / "wind_intervals.csv").set_index("pass_id")
    W["U20"] = W.U_open10m * U10_TO_U20
    ign = pd.read_csv(cfg.interim_dir / "ignition_points.csv")
    ov = pd.read_csv(cfg.interim_dir / "overpasses.csv")
    F = add_persistence(F, json.loads((cfg.interim_dir / "grid.json").read_text())["res"])
    m = dict(m)
    ctx = context(cfg, m)
    F = add_inputs(F, ctx, W)
    m["ctx"] = ctx
    return m, F[F.pass_id >= cfg.first_snapshot].copy(), W, ign, ov  # persistence uses snapshot first-1


def context(cfg: Config, m) -> dict:
    """Landscape layers cropped to the forecast box + the Rothermel lookup tables."""
    L = LS.load(cfg)
    box = crop_box(m["unit"])
    r0, r1, c0, c1 = box
    Lc = {k: L[k][r0:r1, c0:c1] for k in ("m1", "tan_slope", "up_az", "waf", "fuel", "cc")}
    # fuel surface temperature before the fire, quantised exactly as shipped to the page (20 + 0.1 q, C)
    Lc["tpre"] = 20.0 + 0.1 * np.clip(np.round((L["T_fuel"][r0:r1, c0:c1] - 20.0) / 0.1), 0, 255)
    mo = MO.load(cfg)   # satellite live fuel moisture + weather BUI (pipeline/moisture.py); defaults if not run
    m_herb = mo["live"]["herb"] if mo else PH.M_HERB
    m_wood = mo["live"]["wood"] if mo else PH.M_WOOD
    bui = mo["checks"]["bui"] if mo else None
    tab = PH.rothermel_tables(L["meta"]["fuel_names"], LS.QUANT["m1"], m_herb, m_wood)
    # fine dead fuel moisture of every snapshot (landscape.moisture_series); the pre-fire map if not built
    m1_t = LS.moisture_maps(cfg, Lc["m1"].shape) if (cfg.interim_dir / "moisture_t.npz").exists() else None
    return dict(L=Lc, box=box, tab=tab, meta=L["meta"], fields={}, m_herb=m_herb, m_wood=m_wood, bui=bui, m1_t=m1_t)


def moisture_at(ctx, snap):
    """Moisture map used for forecasts made from snapshot `snap`."""
    m = ctx.get("m1_t")
    if m is None:
        return ctx["L"]["m1"]
    return m[int(np.clip(snap, 0, len(m) - 1))]


def add_inputs(F: pd.DataFrame, ctx, W) -> pd.DataFrame:
    """Per-front-point inputs sampled at the point (nearest cell, the web page's rounding)."""
    F = F.copy()
    r0, _, c0, _ = ctx["box"]
    rows, cols = F.row.values - r0, F.col.values - c0
    L = ctx["L"]
    for col, key in (("tan_s", "tan_slope"), ("up_az", "up_az"), ("waf", "waf"), ("fuel_idx", "fuel"), ("Ts", "tpre")):
        F[col] = LS.sample(L, rows, cols, key)
    F["m1"] = 0.0
    for p, idx in F.groupby("pass_id").indices.items():   # moisture of the snapshot the point belongs to
        F.iloc[idx, F.columns.get_loc("m1")] = LS.sample(dict(m1=moisture_at(ctx, p)), rows[idx], cols[idx], "m1")
    q = np.clip(np.round((F.m1.values - LS.QUANT["m1"][0]) / LS.QUANT["m1"][1]), 0, 255).astype(int)
    F["R0"] = ctx["tab"]["R0"][F.fuel_idx.values.astype(int), q]
    F["Umid"] = F.waf.values * W.U20.reindex(F.pass_id.clip(upper=W.index.max())).values
    F["Rr"] = predict_samples("physics", F, ctx, W)   # every input through Rothermel's physics, at the front normal
    F["Rf"] = predict_samples("fbp", F, ctx, W)       # ... and through FBP's
    return F


def fields(ctx, model, U20, U10, wd_from, snap):
    """Per-cell ellipses of a physics model for one weather state and snapshot moisture map (cached)."""
    fam = "fbp" if model == "fbp" else "rothermel"
    snap = int(snap) if ctx.get("m1_t") is not None else -1
    key = (fam, round(float(U20), 6), round(float(U10), 6), round(float(wd_from), 4), snap)
    if key not in ctx["fields"]:
        wind_to = (wd_from + 180.0) % 360.0
        L = dict(ctx["L"], m1=moisture_at(ctx, snap))
        if fam == "fbp":
            f = PH.fbp_field(L, U10, wind_to, bui=ctx.get("bui"))
        else:
            f = PH.rothermel_field(L, ctx["tab"], U20, wind_to, LS.QUANT["m1"])
        f["inv"] = None
        ctx["fields"][key] = f
    return ctx["fields"][key]


def interval_fields(ctx, model, i, W):
    w = min(max(int(i), int(W.index.min())), int(W.index.max()))   # the last snapshot has no next interval
    return fields(ctx, model, W.loc[w, "U20"], W.loc[w, "U_open10m"], W.loc[w, "wd_from"], i)


def add_persistence(F: pd.DataFrame, res: float, radius_m=3.0, k=5) -> pd.DataFrame:
    """Column P: the spread rate observed in the previous interval near each front point (median of the
    5 nearest active points within 3 m; 0 if none) -- the persistence model.
    Column Pm: the same from the interval before that (within 6 m) -- the discovered equation's memory
    term.  The previous interval shares this snapshot's front position with the interval being
    forecast, so position noise makes those two rates anti-correlated; one interval earlier shares none."""
    F = F.copy()
    F["P"] = _memory(F, res, lag=1, radius_m=radius_m, k=k)
    F["Pm"] = _memory(F, res, lag=2, radius_m=2 * radius_m, k=k)
    return F


def _memory(F, res, lag, radius_m, k):
    out = pd.Series(0.0, index=F.index)
    for p, g in F.groupby("pass_id"):
        prev = F[(F.pass_id == p - lag) & F.active & F.ros.notna()]
        if prev.empty:
            continue
        tree = cKDTree(prev[["row", "col"]].values)
        d, j = tree.query(g[["row", "col"]].values, k=min(k, len(prev)))
        d, j = np.atleast_2d(d.T).T, np.atleast_2d(j.T).T
        ros = prev.ros.values[j]
        ros[d > radius_m / res] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows -> 0
            out.loc[g.index] = np.nan_to_num(np.nanmedian(ros, axis=1), nan=0.0)
    return out.values


def split_passes(n_pre, n_pass, k):
    train = list(range(n_pre, n_pre + k - 1))            # interval start snapshots used for training
    test = list(range(n_pre + k - 1, n_pass - 1))        # forecast start snapshots
    return train, test


def interval_sources(i, F, ign, W, ctx):
    r0, _, c0, _ = ctx["box"]
    f = F[(F.pass_id == i) & F.active]
    t0, t1 = W.loc[i, "t0"], W.loc[i, "t1"]
    g = ign[(ign.t_utc > t0) & (ign.t_utc <= t1)].copy()
    g["key"] = np.round(g.row * 2).astype(int) * 100000 + np.round(g.col * 2).astype(int)  # one source per half pixel
    g = g.sort_values("t_utc").drop_duplicates("key")
    prev = F[(F.pass_id == i - 1) & F.active & F.ros.notna()]
    P_ign = float(prev.ros.median()) if len(prev) else 0.0  # persistence: new ignitions spread at the last median rate
    return dict(front=dict(row=f.row.values - r0, col=f.col.values - c0, I=f.frfd.values, k=f.kappa.values, P=f.P.values, Pm=f.Pm.values),
                ign=dict(row=g.row.values - r0, col=g.col.values - c0, tau0=g.t_utc.values - t0), P_ign=P_ign)


def source_inputs(src, ctx, U20, I_ref, i):
    """Per-source feature dict for the discovered equation (front points, then ignitions), snapshot i."""
    L, tab = ctx["L"], ctx["tab"]
    rows = np.r_[src["front"]["row"], src["ign"]["row"]]
    cols = np.r_[src["front"]["col"], src["ign"]["col"]]
    ni = len(src["ign"]["row"])
    m1 = LS.sample(dict(m1=moisture_at(ctx, i)), rows, cols, "m1")
    q = np.clip(np.round((m1 - LS.QUANT["m1"][0]) / LS.QUANT["m1"][1]), 0, 255).astype(int)
    fu = LS.sample(L, rows, cols, "fuel").astype(int)
    return dict(U=LS.sample(L, rows, cols, "waf") * U20, I=np.r_[src["front"]["I"], np.full(ni, I_ref)],
                k=np.r_[src["front"]["k"], np.zeros(ni)], m=m1, tan=LS.sample(L, rows, cols, "tan_slope"),
                up=LS.sample(L, rows, cols, "up_az"), R0=tab["R0"][fu, q], P=np.r_[src["front"]["Pm"], np.full(ni, src["P_ign"])],
                T=LS.sample(L, rows, cols, "tpre"))


def source_physics(src, ctx, i, W):
    """(N, NDIR) Rothermel and FBP normal speeds of each source's own cell (library terms Rr, Rf)."""
    cells = source_cells(src, ctx)
    return {k: np.minimum(PH.field_normal_speed(interval_fields(ctx, mdl, i, W), cells, FC.DIRS), R_CAP)
            for k, mdl in (("Rr", "physics"), ("Rf", "fbp"))}


def source_cells(src, ctx):
    H, W = ctx["L"]["m1"].shape
    rows = np.r_[src["front"]["row"], src["ign"]["row"]]
    cols = np.r_[src["front"]["col"], src["ign"]["col"]]
    r = np.clip(np.floor(rows + 0.5).astype(int), 0, H - 1)
    c = np.clip(np.floor(cols + 0.5).astype(int), 0, W - 1)
    return r * W + c


def normal_speeds(model, i, src, W, ctx, beta=None, I_ref=None, keys=D.DEFAULT_KEYS):
    """(N, NDIR) normal speeds for front + ignition sources of interval i."""
    wind_to = (W.loc[i, "wd_from"] + 180.0) % 360.0
    nf, ni = len(src["front"]["row"]), len(src["ign"]["row"])
    if model == "baseline":
        return np.zeros((nf + ni, FC.NDIR))
    if model in ("physics", "fbp"):
        fld = interval_fields(ctx, model, i, W)
        return np.minimum(PH.field_normal_speed(fld, source_cells(src, ctx), FC.DIRS), R_CAP)
    if model == "persist":
        P = np.minimum(np.r_[src["front"]["P"], np.full(ni, src["P_ign"])], R_CAP)
        return np.repeat(P[:, None], FC.NDIR, axis=1)
    th = np.radians(FC.DIRS - wind_to)[None, :]
    f = {k: np.asarray(v)[:, None] for k, v in source_inputs(src, ctx, W.loc[i, "U20"], I_ref, i).items()}
    f.update(source_physics(src, ctx, i, W))
    return np.minimum(D.normal_speed(beta, th, f, keys, wind_to), R_CAP)


def run_forecast(model, i, src, W, burnt_c, fuel_c, ctx, beta=None, I_ref=None, res=0.5,
                 keys=D.DEFAULT_KEYS, ign_r0=0.5):
    dt = W.loc[i, "dt"]
    nf = len(src["front"]["row"])
    rows = np.r_[src["front"]["row"], src["ign"]["row"]]
    cols = np.r_[src["front"]["col"], src["ign"]["col"]]
    tau0 = np.r_[np.zeros(nf), src["ign"]["tau0"]]
    if model == "flammap":
        fld = interval_fields(ctx, model, i, W)
        if fld["inv"] is None:
            fld["inv"] = FC.mtt_inverse_speeds(fld, res)
        AT = FC.mtt_arrival(rows, cols, tau0, fld["inv"], fuel_c & ~burnt_c[i], burnt_c[i].shape, 2 * dt, res)
    else:
        r = FC.wulff_radius(normal_speeds(model, i, src, W, ctx, beta, I_ref, keys))
        r0 = np.r_[np.zeros(nf), np.full(len(src["ign"]["row"]), ign_r0)]
        if model == "baseline":  # only the ignitions (zero-size wavelets)
            rows, cols, tau0, r0, r = rows[nf:], cols[nf:], tau0[nf:], r0[nf:], r[nf:]
        AT = FC.arrival_time(rows, cols, tau0, r, r0, burnt_c[i].shape, 2 * dt, res)
    return FC.forecast_mask(burnt_c[i], fuel_c, AT, dt), AT


# ---------------------------------------------------------------------------
# point-wise predictions at the measured ROS samples
# ---------------------------------------------------------------------------
ANGLE_CLASSES = dict(head=(0, 30), flank=(60, 120), back=(150, 180.01))


def predict_samples(model, Fs: pd.DataFrame, ctx, W, beta=None, keys=D.DEFAULT_KEYS):
    """Each model's normal spread rate at the sampled front points (normal azimuth phi)."""
    if model == "discovered":
        A, _ = D.design(Fs, keys)
        return np.minimum(np.maximum(A @ beta, 0.0), R_CAP)
    if model == "persist":
        return Fs.P.values.copy()
    if model == "baseline":
        return np.zeros(len(Fs))
    r0, _, c0, _ = ctx["box"]
    H, Wd = ctx["L"]["m1"].shape
    out = np.zeros(len(Fs))
    for p, idx in Fs.groupby("pass_id").indices.items():
        g = Fs.iloc[idx]
        fld = interval_fields(ctx, model, p, W)
        r = np.clip(np.floor(g.row.values - r0 + 0.5).astype(int), 0, H - 1)
        c = np.clip(np.floor(g.col.values - c0 + 0.5).astype(int), 0, Wd - 1)
        cell = r * Wd + c
        th = np.radians(g.phi.values - fld["head_az"].ravel()[cell])
        a, b, cc = (fld[k].ravel()[cell] for k in ("a", "b", "c"))
        out[idx] = cc * np.cos(th) + np.sqrt((a * np.cos(th)) ** 2 + (b * np.sin(th)) ** 2)
    return np.minimum(out, R_CAP)


def class_medians(theta_deg, ros) -> dict:
    """Head / flank / back medians by angle between front normal and wind; L/B with the
    ellipse geometry used for every model, (head + back) / (2 * flank)."""
    a = np.abs(theta_deg)
    med = {}
    for k, (lo, hi) in ANGLE_CLASSES.items():
        sel = (a >= lo) & (a <= hi)
        med[k] = float(np.median(ros[sel])) if sel.any() else np.nan
        med["n_" + k] = int(sel.sum())
    med["lb"] = float((med["head"] + med["back"]) / (2 * med["flank"])) if med["flank"] > 0 else np.nan
    return med


def r2_score(y, pred):
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()) if len(y) > 1 else np.nan


def fit_split(cfg: Config, k: int, m=None, F=None, W=None, ign=None, verbose=True, forecasts=True, cache=None):
    if m is None:
        m, F, W, ign, _ = load(cfg)
    ctx = m["ctx"]
    n_pre, n_pass = cfg.first_snapshot, m["burnt"].shape[0]
    train, test = split_passes(n_pre, n_pass, k)
    keys = cfg.terms
    res = json.loads((cfg.interim_dir / "grid.json").read_text())["res"]
    Fc = F[F.clean]
    B = D.block_stats(Fc, keys)
    Btr, Bte = B[B.pass_id.isin(train)], B[B.pass_id.isin(test)]
    signs = D.signs_for(keys, cfg.sign_priors)
    forced = D.forced_for(keys, cfg.forced_terms)
    pen = D.pen_for(keys, cfg.unpenalized_terms)
    Ftr, Fte = Fc[Fc.pass_id.isin(train)], Fc[Fc.pass_id.isin(test)]
    I_ref = float(Ftr.frfd.median())
    ridge = cfg.ridge
    if forced.all():   # every term kept: choose the shrinkage instead of the threshold
        lam, cv = 0.0, None
        ridge, cv = choose_ridge_forecast(cfg, m, F, W, ign, Btr, train, signs, I_ref, keys, forced, pen)
    elif cfg.lambda_criterion == "forecast":
        lam, cv = choose_lambda_forecast(cfg, m, F, W, ign, Btr, train, signs, I_ref, keys, forced)
    else:
        lam, cv = D.choose_lambda(Btr, LAMBDAS, cfg.ridge, signs, forced)
    if cfg.stlsq_threshold is not None:
        lam = float(cfg.stlsq_threshold)
    AtA, Aty, yty, n = D.sum_stats(Btr)
    beta, xi, active = D.stlsq(AtA, Aty, n, lam, ridge, signs=signs, forced=forced, pen=pen)
    r2_tr, rmse_tr = D.r2(beta, AtA, Aty, yty, n)
    r2_te, rmse_te = D.r2(beta, *D.sum_stats(Bte)) if len(Bte) else (np.nan, np.nan)
    boot = D.bootstrap(Btr, lam, cfg.n_bootstrap, ridge, keys=keys, signs=signs, forced=forced, pen=pen)

    # point-wise skill and the head / flank / back comparison at the measured front points
    pts = {}
    for model in ["discovered", "physics", "fbp", "persist"]:
        pts[model] = dict(r2_train=r2_score(Ftr.ros.values, predict_samples(model, Ftr, ctx, W, beta, keys)),
                          r2_test=r2_score(Fte.ros.values, predict_samples(model, Fte, ctx, W, beta, keys)),
                          **class_medians(Fc.theta.values, predict_samples(model, Fc, ctx, W, beta, keys)))
    pts["flammap"] = pts["physics"]
    observed = dict(**class_medians(Fc.theta.values, Fc.ros.values), n=int(len(Fc)))

    Wtr = W.loc[train]
    U_ref = float(np.average(Wtr.U20, weights=Wtr.dt))
    U10_ref = float(np.average(Wtr.U_open10m, weights=Wtr.dt))
    wd_ref = float(np.degrees(np.arctan2(np.average(np.sin(np.radians(Wtr.wd_from)), weights=Wtr.dt),
                                         np.average(np.cos(np.radians(Wtr.wd_from)), weights=Wtr.dt))) % 360)
    ref_inputs = {kk: float(np.median(Ftr[c])) for kk, c in
                  (("U", "Umid"), ("I", "frfd"), ("m", "m1"), ("tan", "tan_s"), ("R0", "R0"), ("P", "Pm"))}
    result = dict(k=k, keys=keys, res=res, train=train, test=test, lam=lam, ridge=ridge, cv=cv, beta=beta, xi=xi, active=active, boot=boot,
                  r2_train=r2_tr, r2_test=r2_te, rmse_train=rmse_tr, rmse_test=rmse_te,
                  n_train=int(n), n_test=int(Fte.shape[0]), I_ref=I_ref, U_ref=U_ref, U10_ref=U10_ref, wd_ref=wd_ref,
                  point_skill=pts, observed=observed, ref_inputs=ref_inputs, B=B,
                  explanations=D.explain(beta, active, boot.inclusion.values, dict(U=ref_inputs["U"], I=I_ref), keys,
                                         "midflame wind"))
    if forecasts:
        box = ctx["box"]
        r0_, r1_, c0_, c1_ = box
        burnt_c = m["burnt"][:, r0_:r1_, c0_:c1_]
        fuel_c = m["unit"][r0_:r1_, c0_:c1_]
        cache = {} if cache is None else cache
        rows, preds = [], {}
        for i in range(n_pre, n_pass - 1):
            src = interval_sources(i, F, ign, W, ctx)
            for model in MODELS:
                if model in STATIC and (i, model) in cache:
                    pred, mt = cache[(i, model)]
                else:
                    pred, AT = run_forecast(model, i, src, W, burnt_c, fuel_c, ctx, beta, I_ref, res, keys, cfg.ign_r0_m)
                    mt = FC.metrics(pred, burnt_c[i + 1], burnt_c[i], fuel_c, AT, W.loc[i, "dt"], res)
                    if model in STATIC:
                        cache[(i, model)] = (pred, mt)
                rows.append(dict(pass_id=i, model=model, split="train" if i in train else "test",
                                 phase="ignition" if len(src["ign"]["row"]) else "spread", **mt))
                preds[(i, model)] = pred
        result["metrics"] = pd.DataFrame(rows)
        result["preds"] = preds
        result["box"] = box
        result["significance"] = significance(result["metrics"])
    if verbose:
        report(result)
    return result


def choose_ridge_forecast(cfg, m, F, W, ign, Btr, train, signs, I_ref, keys, forced, pen=None):
    """All terms kept: choose the ridge shrinkage by leave-one-interval-out forecast skill (new-area IoU) on the
    training intervals; paired 1-SE rule towards stronger shrinkage."""
    ctx = m["ctx"]
    r0, r1, c0, c1 = ctx["box"]
    burnt_c, fuel_c = m["burnt"][:, r0:r1, c0:c1], m["unit"][r0:r1, c0:c1]
    res = json.loads((cfg.interim_dir / "grid.json").read_text())["res"]
    grid = np.asarray(cfg.ridge_grid, float)
    S = np.full((len(grid), len(train)), np.nan)
    for j, p in enumerate(train):
        AtA, Aty, _, n = D.sum_stats(Btr[Btr.pass_id != p])
        src = interval_sources(p, F, ign, W, ctx)
        for i, rg in enumerate(grid):
            beta, _, _ = D.stlsq(AtA, Aty, n, 0.0, rg, signs=signs, forced=forced, pen=pen)
            pred, AT = run_forecast("discovered", p, src, W, burnt_c, fuel_c, ctx, beta, I_ref, res, keys, cfg.ign_r0_m)
            S[i, j] = FC.metrics(pred, burnt_c[p + 1], burnt_c[p], fuel_c, AT, W.loc[p, "dt"], res)["iou_new"]
    S = S[:, np.isfinite(S).all(0)]
    mean = S.mean(1)
    best = int(np.argmax(mean))
    diff = S[best] - S
    se = diff.std(1, ddof=1) / np.sqrt(S.shape[1])
    cand = np.where(diff.mean(1) <= se)[0]
    return float(grid[cand.max()]), pd.DataFrame(dict(lam=grid, cv_mse=-mean, cv_se_paired=se))


def choose_lambda_forecast(cfg, m, F, W, ign, Btr, train, signs, I_ref, keys, forced=None):
    """Leave-one-interval-out over the TRAINING intervals, scored by what the model is judged on:
    for each held-out interval, fit on the others at every lambda and forecast the held-out
    interval; score = new-area IoU.  Paired 1-SE rule towards the simpler (larger) lambda."""
    ctx = m["ctx"]
    r0, r1, c0, c1 = ctx["box"]
    burnt_c, fuel_c = m["burnt"][:, r0:r1, c0:c1], m["unit"][r0:r1, c0:c1]
    res = json.loads((cfg.interim_dir / "grid.json").read_text())["res"]
    S = np.full((len(LAMBDAS), len(train)), np.nan)
    for j, p in enumerate(train):
        AtA, Aty, _, n = D.sum_stats(Btr[Btr.pass_id != p])
        src = interval_sources(p, F, ign, W, ctx)
        seen = {}
        for i, lam in enumerate(LAMBDAS):
            beta, _, _ = D.stlsq(AtA, Aty, n, lam, cfg.ridge, signs=signs, forced=forced)
            key = tuple(np.round(beta, 12))
            if key not in seen:
                pred, AT = run_forecast("discovered", p, src, W, burnt_c, fuel_c, ctx, beta, I_ref, res, keys, cfg.ign_r0_m)
                seen[key] = FC.metrics(pred, burnt_c[p + 1], burnt_c[p], fuel_c, AT, W.loc[p, "dt"], res)["iou_new"]
            S[i, j] = seen[key]
    ok = np.isfinite(S).all(0)
    S = S[:, ok]
    mean = S.mean(1)
    best = int(np.argmax(mean))
    diff = S[best] - S
    se = diff.std(1, ddof=1) / np.sqrt(S.shape[1])
    cand = np.where(diff.mean(1) <= se)[0]
    lam = float(LAMBDAS[cand.max()])
    return lam, pd.DataFrame(dict(lam=LAMBDAS, cv_mse=-mean, cv_se_paired=se))


def significance(mt: pd.DataFrame, key="iou_new") -> dict:
    """Paired Wilcoxon signed-rank test, discovered vs every other model, over the test intervals
    (two-sided; the per-interval scores are paired because every model forecasts the same interval)."""
    from scipy.stats import wilcoxon
    t = mt[mt.split == "test"].pivot(index="pass_id", columns="model", values=key)
    out = {}
    for model in MODELS[1:]:
        d = (t["discovered"] - t[model]).dropna()
        d = d[d != 0]
        out[model] = dict(n=int(len(d)), median_diff=float(d.median()) if len(d) else np.nan,
                          p=float(wilcoxon(d).pvalue) if len(d) >= 5 else np.nan)
    return out


def report(r):
    print(f"split k={r['k']}: train intervals {r['train'][0]}..{r['train'][-1]}, test {r['test'][0]}..{r['test'][-1]}")
    print(f"lambda (CV, 1-SE) = {r['lam']}; n_train={r['n_train']} n_test={r['n_test']}")
    for (key, disp), b, a, inc in zip(D.terms_for(r["keys"]), r["beta"], r["active"], r["boot"].inclusion):
        print(f"   {disp:12s} {'%+.5f' % b if a else 'dropped':>9s}   bootstrap inclusion {inc:.0%}")
    print(f"discovered R² train {r['r2_train']:.3f} test {r['r2_test']:.3f}")
    for k, v in r["point_skill"].items():
        print(f"   {k:10s} R² train {v['r2_train']:6.3f} test {v['r2_test']:6.3f} | head {v['head']:.4f} flank {v['flank']:.4f} "
              f"back {v['back']:.4f} L/B {v['lb']:.2f}")
    o = r["observed"]
    print(f"   observed   head {o['head']:.4f} flank {o['flank']:.4f} back {o['back']:.4f} L/B {o['lb']:.2f} (n={o['n']})")
    if "metrics" in r:
        mt = r["metrics"]
        cols = ["iou", "iou_new", "front_mean_m", "front_p90_m", "arrival_err_s"]
        print("TEST means\n", mt[mt.split == "test"].groupby("model")[cols].mean().round(3).loc[MODELS])
        print("Wilcoxon (discovered vs model, new-area IoU):",
              {k: f"Δ{v['median_diff']:+.3f} p={v['p']:.3f}" for k, v in r["significance"].items()})


BASE_TERMS = ("1", "cos", "sin", "cos2", "I", "Icos", "k", "kcos")   # library before the model inputs were added
from .config import ALL_TERMS
ABLATION = [  # (name, library, λ criterion, sign priors, forced terms, unpenalised terms)
    ("A. Front terms only (θ, FRFD, κ); sparse, λ by ROS error", BASE_TERMS, "mse", False, (), ()),
    ("B. + every input as a candidate (16 terms); sparse, λ by forecast skill", tuple(ALL_TERMS), "forecast", True, (), ()),
    ("C. all 16 terms kept, ridge by forecast skill", tuple(ALL_TERMS), "forecast", True, tuple(ALL_TERMS), ()),
    ("D. direction-aware library, all kept, inputs shrunk, cos θ / cos² θ free (final)", None, "forecast", True, None, ("cos", "cos2")),
]


def ablation(cfg: Config, k=None, out=None):
    """Adds one accuracy idea at a time and scores the discovered equation on the same test snapshots.
    The final configuration (D) was fixed before looking at test scores; this table is diagnostic."""
    from dataclasses import replace
    k = k or cfg.train_fire_passes
    m, F, W, ign, _ = load(cfg)
    cache, rows = {}, []
    for name, terms, crit, sg, forced, free in ABLATION:
        c = replace(cfg, term_keys=terms, lambda_criterion=crit, sign_priors=sg,
                    forced_terms=tuple(c_ for c_ in (terms or cfg.terms)) if forced is None else forced, unpenalized_terms=free)
        r = fit_split(c, k, m, F, W, ign, verbose=False, forecasts=True, cache=cache)
        t = r["metrics"][(r["metrics"].split == "test") & (r["metrics"].model == "discovered")]
        eq = " + ".join(f"{b:.4g}·{d}" for (_, d), b, a in zip(D.terms_for(c.terms), r["beta"], r["active"]) if a)
        rows.append(dict(variant=name, lam=r["lam"], train_cv_iou=float(-r["cv"].cv_mse.min()) if crit == "forecast" else np.nan,
                         iou_new=t.iou_new.mean(), front_mean_m=t.front_mean_m.mean(),
                         front_p90_m=t.front_p90_m.mean(), arrival_err_s=t.arrival_err_s.mean(), r2_test=r["r2_test"],
                         terms_kept=int(r["active"].sum()), equation=eq))
    df = pd.DataFrame(rows)
    df.to_csv(out or cfg.interim_dir / "ablation.csv", index=False)
    return df


if __name__ == "__main__":
    cfg = Config()
    fit_split(cfg, int(sys.argv[1]) if len(sys.argv) > 1 else cfg.train_fire_passes)
