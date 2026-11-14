"""Export everything the web page needs to web/data/<data>.json and build
the self-contained web/index.html (JSON inlined so it works from file://).

The browser re-runs STLSQ (from per-block normal equations), the bootstrap,
every model's forecast and the metrics; the Python results for every split
are shipped too so the page can be checked against them
(tests/check_js_core.js).
"""
from __future__ import annotations

import base64
import io
import json
import re
import sys

import numpy as np
import pandas as pd
from PIL import Image

from . import discover as D
from . import evaluate as EV
from . import forecast as FC
from . import landscape as LS
from . import moisture as MO
from . import physics as PH
from .config import ROOT, SITES, Config
from .sites import SITE

# models offered in the page's physics-model picker (the discovered equation is always shown)
PICKER = [
    dict(key="flammap", name="FlamMap MTT (Rothermel)", engine="mtt", field="rothermel"),
    dict(key="physics", name="Rothermel + Huygens (FARSITE-style)", engine="huygens", field="rothermel"),
    dict(key="fbp", name="Canadian FBP C-7", engine="huygens", field="fbp"),
    dict(key="persist", name="Persistence (last observed rate)", engine="persist", field=None),
]


def b64u8(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode()


def rle8(a: np.ndarray) -> list:
    """uint8 run-length code [v0, n0, v1, n1, ...] (for the blocky LANDFIRE layers)."""
    f = np.asarray(a, np.uint8).ravel()
    idx = np.r_[0, np.flatnonzero(np.diff(f)) + 1]
    runs = np.diff(np.r_[idx, f.size])
    return np.column_stack([f[idx], runs]).ravel().tolist()


def rle(mask: np.ndarray):
    f = mask.ravel().astype(np.int8)
    idx = np.flatnonzero(np.diff(f)) + 1
    runs = np.diff(np.r_[0, idx, f.size])
    return [int(f[0])] + runs.tolist()


def jpeg_b64(E: np.ndarray, lo, hi, q=88):
    v = np.clip(255 * (np.log10(np.clip(np.nan_to_num(E, nan=lo), lo, hi)) - np.log10(lo)) /
                (np.log10(hi) - np.log10(lo)), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(v, mode="L").save(buf, format="JPEG", quality=q)
    return base64.b64encode(buf.getvalue()).decode()


def inferno_lut():
    import matplotlib
    cm = matplotlib.colormaps["inferno"]
    return "".join("%02x%02x%02x" % tuple(int(255 * c) for c in cm(i / 255)[:3]) for i in range(256))


def _r(x, nd):
    return [round(float(v), nd) for v in np.asarray(x)]


def build(cfg: Config, verbose=True):
    keys = cfg.terms
    k_list = SITES[cfg.site]["k_list"]
    m, F, W, ign, ov = EV.load(cfg)
    ctx = m["ctx"]
    n_pre, n_pass = cfg.first_snapshot, m["burnt"].shape[0]
    box = ctx["box"]
    r0, r1, c0, c1 = box
    grid = json.loads((cfg.interim_dir / "grid.json").read_text())
    a, _, x0, _, e, y1 = grid["transform"]
    tm = ov.t_unit_mean.values

    intervals = []
    for i in range(n_pre, n_pass - 1):
        src = EV.interval_sources(i, F, ign, W, ctx)
        intervals.append(dict(
            i=i, t0=round(W.loc[i, "t0"], 1), dt=round(W.loc[i, "dt"], 1), U10=float(W.loc[i, "U_open10m"]),
            U20=float(W.loc[i, "U20"]), wd_from=float(W.loc[i, "wd_from"]),
            phase="ignition" if len(src["ign"]["row"]) else "spread",
            front=dict(r=_r(src["front"]["row"], 2), c=_r(src["front"]["col"], 2), I=_r(src["front"]["I"], 2),
                       k=_r(src["front"]["k"], 4), P=_r(src["front"]["P"], 4), Pm=_r(src["front"]["Pm"], 4)),
            ign=dict(r=_r(src["ign"]["row"], 2), c=_r(src["ign"]["col"], 2), tau0=_r(src["ign"]["tau0"], 1)),
            P_ign=round(src["P_ign"], 4)))

    B = D.block_stats(F[F.clean], keys)
    iu = np.triu_indices(len(keys))
    blocks = [dict(p=int(r.pass_id), n=int(r.n), A=[float("%.7g" % v) for v in r.AtA[iu]],
                   b=[float("%.7g" % v) for v in r.Aty], yy=float("%.7g" % r.yty)) for r in B.itertuples()]

    splits, py_metrics, cache = {}, [], {}
    for k in k_list:
        r = EV.fit_split(cfg, k, m, F, W, ign, verbose=False, forecasts=True, cache=cache)
        mt = r["metrics"]
        py_metrics.append(mt.assign(k=k))
        splits[k] = dict(
            k=k, train=r["train"], test=r["test"], lam_cv=r["lam"], ridge=r["ridge"], lam_criterion=cfg.lambda_criterion,
            all_terms=bool(len(cfg.forced_terms) == len(keys)),
            refs=dict(U=r["U_ref"], U10=r["U10_ref"], wd_from=r["wd_ref"], I=r["I_ref"]), ref_inputs=r["ref_inputs"],
            point_skill=r["point_skill"], significance=r["significance"],
            r2=dict(disc_train=r["r2_train"], disc_test=r["r2_test"]),
            py=dict(beta=_r(r["beta"], 6), active=[bool(v) for v in r["active"]], inclusion=_r(r["boot"].inclusion, 3),
                    metrics={mod: {c: _r(g.sort_values("pass_id")[c], 4) for c in
                                   ["iou", "iou_new", "front_mean_m", "front_p90_m", "arrival_err_s"]}
                             for mod, g in mt.groupby("model")}),
            cv=dict(lam=_r(r["cv"].lam, 5), mse=_r(r["cv"].cv_mse, 6)) if r["cv"] is not None else None)
        if verbose:
            t = mt[mt.split == "test"].groupby("model").iou_new.mean()
            print(f"  split k={k}: lam={r['lam']}, kept {int(r['active'].sum())} terms, test new-IoU "
                  + ", ".join(f"{mm} {t[mm]:.3f}" for mm in EV.MODELS), file=sys.stderr)
    pd.concat(py_metrics).to_csv(cfg.interim_dir / "metrics_all_splits.csv", index=False)

    burnt_c = m["burnt"][:, r0:r1, c0:c1]
    col = SITE["color"]
    observed = r["observed"]
    Lc, tab, lmeta = ctx["L"], ctx["tab"], ctx["meta"]
    Lfull = LS.load(cfg)
    elev = Lfull["elev"][r0:r1, c0:c1]
    tpre = Lfull["T_fuel"][r0:r1, c0:c1]
    e_lo, e_st = float(np.floor(elev.min() * 10) / 10), 0.05
    layers = dict(quant={k: list(v) for k, v in LS.QUANT.items()}, fuel_names=lmeta["fuel_names"], fuel_codes=lmeta["fuel_codes"],
                  m1=b64u8(LS.to_uint8("m1", Lc["m1"])), tan_slope=b64u8(LS.to_uint8("tan_slope", Lc["tan_slope"])),
                  up_az=b64u8(LS.to_uint8("up_az", Lc["up_az"])), waf=rle8(LS.to_uint8("waf", Lc["waf"])),
                  cc=rle8(LS.to_uint8("cc", Lc["cc"])), fuel=rle8(Lc["fuel"]),
                  elev=b64u8(np.clip(np.round((elev - e_lo) / e_st), 0, 255)), elev_q=[e_lo, e_st],
                  tpre=b64u8(np.clip(np.round((tpre - 20.0) / 0.1), 0, 255)), tpre_q=[20.0, 0.1])
    rothermel = dict(fuels=tab["fuels"], R0=[[float("%.9g" % v) for v in row] for row in tab["R0"]],
                     IR=[[float("%.9g" % v) for v in row] for row in tab["IR"]],
                     **{k: [float(v) for v in tab[k]] for k in ("C", "B", "E", "ratio", "beta")},
                     m_herb=ctx["m_herb"], m_wood=ctx["m_wood"], m10_add=PH.M10_ADD, m100_add=PH.M100_ADD)
    fbp = dict(fuel="C-7", **{k: PH.FBP_FUELS["C-7"][k] for k in ("a", "b", "c", "q", "bui0")}, slope_max=PH.FBP_SLOPE_MAX_PCT,
               bui=ctx["bui"])
    moisture = MO.load(cfg)
    if moisture:
        moisture = dict(live=moisture["live"], checks=moisture["checks"])
        live_map = live_layer(cfg, ctx)
        if live_map is not None:
            layers["ndii"] = b64u8(live_map[0]); layers["ndii_q"] = live_map[1]
    from .flame3 import ONSITE_WX
    inputs = dict(stats=lmeta["stats"], fuel_share=lmeta["fuel_share"], T_air=lmeta["T_air"], RH_air=lmeta["RH_air"],
                  U10=float(W.U_open10m.iloc[0]), U20=float(W.U20.iloc[0]), wd_from=float(W.wd_from.iloc[0]),
                  onsite_wind=ONSITE_WX["wind_mph"] * PH.MPH, sources=lmeta["sources"])
    meta = dict(
        site=cfg.site, date=cfg.date, res=float(a), box=[int(v) for v in box], H=int(r1 - r0), W=int(c1 - c0),
        x0=float(x0 + c0 * a), y0=float(y1 + r0 * e), crs=f"EPSG:{cfg.epsg}",
        n_pre=n_pre, n_pass=n_pass, t_utc=_r(tm, 1), frames=ov.frames_on_unit.astype(str).tolist(),
        default_k=cfg.train_fire_passes, k_list=list(k_list), k_range=[min(k_list), max(k_list)],
        lam_max=0.05, ridge=cfg.ridge, R_CAP=EV.R_CAP, IGN_R0=cfg.ign_r0_m, NDIR=FC.NDIR,
        terms=[dict(key=k, disp=d, sign=int(sg), forced=bool(fo), pen=float(pw)) for (k, d), sg, fo, pw in
               zip(D.terms_for(keys), D.signs_for(keys, cfg.sign_priors), D.forced_for(keys, cfg.forced_terms),
                   D.pen_for(keys, cfg.unpenalized_terms))],
        models=PICKER, default_model="flammap", model_names=EV.MODEL_NAMES,
        layers=layers, rothermel=rothermel, fbp=fbp, inputs=inputs, moisture=moisture,
        color=dict(lo=col["lo"], hi=col["hi"], unit=col["unit"], ticks=col["ticks"], lut=inferno_lut()),
        labels=SITE["labels"], sources=SITE["sources"], observed=observed)
    mt = cfg.interim_dir / "moisture_t.npz"
    if mt.exists():   # fine dead fuel moisture of every shown snapshot (1 m blocks, same quantisation)
        z = np.load(mt)
        layers["m1_t"] = dict(h2=int(z["q"].shape[1]), w2=int(z["q"].shape[2]), first=int(n_pre),
                              maps=[b64u8(z["q"][p]) for p in range(n_pre, n_pass)])
        if "tq" in z.files:   # the surface temperature each moisture map comes from (display only)
            layers["t_t"] = dict(enc=LS.T_TQ, maps=[b64u8(z["tq"][p]) for p in range(n_pre, n_pass)])
    bmj = LS.raw_dir(cfg) / "basemap_esri_world_topo.jpg"
    if bmj.exists():
        info = json.loads((LS.raw_dir(cfg) / "basemap_esri_world_topo.json").read_text())
        meta["basemap"] = dict(jpg=base64.b64encode(bmj.read_bytes()).decode(), bounds=info["bounds"], name=info["name"],
                               attribution=info["attribution"], service=LS.BASEMAP_URL)
        dem3 = LS.raw_dir(cfg) / "dem_3dep_3m_basemap_extent.tif"
        if dem3.exists():   # relief shading + 3D view: elevation in cm above z0, uint16 little-endian
            import rasterio
            with rasterio.open(dem3) as d:
                z = d.read(1).astype(float)
            z[(z < -1000) | ~np.isfinite(z)] = np.nan
            z = np.where(np.isnan(z), np.nanmean(z), z)
            z0 = float(np.floor(z.min()))
            q = np.clip(np.round((z - z0) * 100), 0, 65535).astype("<u2")
            meta["terrain3d"] = dict(bounds=info["bounds"], w=int(z.shape[1]), h=int(z.shape[0]), z0=z0, step=0.01,
                                     b64=base64.b64encode(q.tobytes()).decode(), res=LS.TERRAIN3D_RES_M,
                                     source="USGS 3DEP 1 m lidar DEM, resampled to 3 m")
    wxf = MO.raw_dir(cfg) / "KTLH_20230228_full.csv"
    if wxf.exists():   # weather at the site on the burn day: the crew's note + every KTLH report
        from .flame3 import ONSITE_NOTES
        w = pd.read_csv(wxf, na_values="M")
        num = lambda v, f=lambda x: x, nd=1: (round(float(f(v)), nd) if pd.notna(v) and str(v) not in ("T",) else None)
        obs = []
        for rr in w.itertuples():
            t = pd.Timestamp(rr.valid)
            obs.append(dict(t=int(t.hour * 3600 + t.minute * 60), time=t.strftime("%H:%M"), T=num(rr.tmpf, lambda x: (x - 32) / 1.8),
                            Td=num(rr.dwpf, lambda x: (x - 32) / 1.8), RH=num(rr.relh, nd=0), wd=num(rr.drct, nd=0),
                            ws=num(rr.sknt, lambda x: x * 0.514444), gust=num(rr.gust, lambda x: x * 0.514444),
                            p=num(rr.alti, lambda x: x * 33.8639, 0), vis=num(rr.vsby, lambda x: x * 1.609344, 0),
                            sky=rr.skyc1 if pd.notna(rr.skyc1) else None, skyl=num(rr.skyl1, lambda x: x * 0.3048, 0),
                            wx=rr.wxcodes if pd.notna(rr.wxcodes) else None))
        mo = MO.load(cfg)
        meta["weather"] = dict(station="KTLH (Tallahassee airport, 30 km S)", date="28 Feb 2023", obs=obs,
                               onsite=[dict(n, T=round((n["T_F"] - 32) / 1.8, 1), ws=round(n["wind_mph"] * PH.MPH, 1)) for n in ONSITE_NOTES],
                               fwi=dict(FFMC=round(mo["checks"]["ffmc"], 1), BUI=round(mo["checks"]["bui"], 1)) if mo else None)
    vf = np.load(cfg.interim_dir / "frames.npz")["valid_frac"][r0:r1, c0:c1]
    meta["footprint"] = rle(vf >= 0.9)   # outside it the thermal base map is transparent
    data = dict(meta=meta, intervals=intervals, blocks=blocks, splits={str(k): v for k, v in splits.items()},
                fuel=rle(m["unit"][r0:r1, c0:c1]),
                # snapshots before first_snapshot are not shown: ship empty placeholders to keep absolute indexing
                burnt=[rle(b) if p >= n_pre else [0, int(b.size)] for p, b in enumerate(burnt_c)],
                images=[jpeg_b64(E[r0:r1, c0:c1], col["lo"], col["hi"]) if p >= n_pre else "" for p, E in enumerate(m["E"])])
    out = cfg.web_dir / "data"
    out.mkdir(parents=True, exist_ok=True)
    (out / SITE["data"]).write_text(json.dumps(_clean(data), separators=(",", ":"), allow_nan=False))
    return data


def live_layer(cfg, ctx):
    """Sentinel-2 NDII of the last clear pre-burn scene on the forecast grid (nearest 20 m pixel), uint8."""
    p = MO.raw_dir(cfg) / "s2_last_clear.npz"
    if not p.exists():
        return None
    import rasterio
    from rasterio.warp import Resampling, reproject
    z = np.load(p, allow_pickle=True)
    g = json.loads((cfg.interim_dir / "grid.json").read_text())
    r0, r1, c0, c1 = ctx["box"]
    a, _, x0, _, e, y1 = g["transform"]
    dst = np.full((r1 - r0, c1 - c0), np.nan)
    reproject(z["ndii"].astype(float), dst, src_transform=rasterio.Affine(*z["transform"]), src_crs=str(z["crs"]),
              dst_transform=rasterio.Affine(a, 0, x0 + c0 * a, 0, e, y1 + r0 * e), dst_crs=g["crs"],
              resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan)
    lo, st = -0.2, 0.004
    return np.clip(np.round((np.nan_to_num(dst, nan=lo) - lo) / st), 0, 255).astype(np.uint8), [lo, st]


def _clean(o):
    """NaN/inf -> None and numpy scalars -> Python, recursively."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return float(o) if np.isfinite(o) else None
    return o


def build_html(cfg: Config):
    tpl = (ROOT / "pipeline" / "web_template.html").read_text()
    for name in SITE.get("hide", []):  # drop optional page sections
        tpl = re.sub(rf"\s*<!--SECTION:{name}-->.*?<!--/SECTION:{name}-->", "", tpl, flags=re.S)
    for k in ("title", "description", "h1", "subtitle", "map_caption", "eqvars", "panel_caption", "steps",
              "limitations", "methods", "data_heading", "inputs_caption", "footer"):
        tpl = tpl.replace("{{" + k + "}}", SITE.get(k, ""))
    js = (cfg.web_dir / "data" / SITE["data"]).read_text()
    html = tpl.replace("/*__DEMO_DATA__*/null", js.replace("</", "<\\/"))
    out = cfg.web_dir / SITE["page"]
    out.write_text(html)
    return out


if __name__ == "__main__":
    cfg = Config()
    build(cfg)
    print(build_html(cfg))
