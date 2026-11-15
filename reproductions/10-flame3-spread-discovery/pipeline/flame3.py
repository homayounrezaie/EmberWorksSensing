"""Second site: FLAME 3 "Hanna Hammock" nadir thermal plot (Hopkins et al. 2024).

A UAV (Autel EVO II 640T, radiometric LWIR) hovered at ~122 m over a fixed,
GCP-georeferenced ~100 m plot during the Tall Timbers prescribed burn of
28 Feb 2023 and recorded a radiometric thermal frame every 5 s.  The dataset
ships stabilised + georeferenced frames in degrees Celsius
("geo_thermal_tiff_celsius").  This module produces the same interim
artefacts as the RxCADRE stages (masks.npz, front_points.csv,
wind_intervals.csv, ignition_points.csv, overpasses.csv, grid.json) so that
evaluate.py / export.py / the web page work unchanged.

Stages: download -> ingest (reproject frames to a north-up grid) -> build
(snapshots, burnt / active masks, data-driven ignition detection, wind,
fronts + ROS samples).
"""
from __future__ import annotations

import io
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.transform import Affine
from rasterio.warp import Resampling, reproject
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import morphology

from .config import Config, SITES
from .remote_zip import HTTPRangeFile, extract_members

KAGGLE = "https://www.kaggle.com/api/v1/datasets/download/brycehopkins/flame-3-nadir-thermal-plot-subset"
CAMERA_CLOCK_OFFSET_H = 4  # README: "time stamps on the images are all 4 hours behind the actual time"
SIGMA = 5.670374419e-8


def raw_dir(cfg: Config) -> Path:
    return cfg.raw_dir / "flame3"


def plot_name(cfg: Config) -> str:
    return f"plot {SITES[cfg.site]['plot']}"


# ----------------------------------------------------------------------------
# download
# ----------------------------------------------------------------------------
def download(cfg: Config) -> None:
    out = raw_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    listing = out / "zip_listing.json"
    if not listing.exists():
        zf = zipfile.ZipFile(HTTPRangeFile(KAGGLE, block=1 << 20))
        json.dump([(i.filename, i.file_size, i.compress_size, i.header_offset) for i in zf.infolist()], open(listing, "w"))
    L = json.load(open(listing))
    plot = plot_name(cfg)
    want = [n for n, s, c, o in L if not n.endswith("/") and (
        (f"/{plot}/duringburn/geo_thermal_tiff_celsius" in n) or "/GCPs/" in n or "/metadata/" in n or n.endswith("README.rtf"))]
    print(f"[download] FLAME 3 {plot}: {len(want)} members", file=sys.stderr)
    extract_members(KAGGLE, want, out)

    # frame acquisition times: EXIF of the raw thermal JPGs (first 48 KB of each member)
    times = out / f"frame_times_{plot.replace(' ', '')}.json"
    if not times.exists():
        jp = sorted((o, n) for n, s, c, o in L if f"/{plot}/duringburn/raw_thermal_jpg/" in n and n.lower().endswith(".jpg"))
        zf = zipfile.ZipFile(HTTPRangeFile(KAGGLE, block=64 << 10))
        from PIL import Image
        rows = []
        for _, name in jp:
            with zf.open(name) as m:
                head = m.read(48 << 10)
            ex = Image.open(io.BytesIO(head + b"\xff\xd9")).getexif()
            rows.append(dict(frame=int(re.search(r"IRX_(\d+)", name).group(1)), exif_time=ex.get(306)))
        json.dump(rows, open(times, "w"))

    # Tallahassee (KTLH) METAR, 30 km S of the burn: the only wind-direction record for the afternoon
    metar = out / "KTLH_20230228.csv"
    if not metar.exists():
        r = requests.get("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py", timeout=60, params=dict(
            station="TLH", data=["drct", "sknt", "gust", "tmpf", "relh"], year1=2023, month1=2, day1=28,
            year2=2023, month2=3, day2=1, tz="America/New_York", format="onlycomma", latlon="yes", missing="M",
            report_type=[3, 4]), headers={"User-Agent": "rxcadre-spread-pipeline/0.1"})
        r.raise_for_status()
        metar.write_text(r.text)
    (cfg.raw_dir / f".download_complete_{cfg.site}").write_text("ok\n")


# ----------------------------------------------------------------------------
# ingest: reproject every frame onto one north-up grid
# ----------------------------------------------------------------------------
def _frame_files(cfg: Config):
    base = raw_dir(cfg) / "Hanna Hammock" / plot_name(cfg) / "duringburn" / "geo_thermal_tiff_celsius"
    fs = sorted(base.rglob("*.TIFF"), key=lambda p: int(re.search(r"IRX_(\d+)", p.name).group(1)))
    return fs


def ingest(cfg: Config) -> dict:
    out = cfg.interim_dir
    out.mkdir(parents=True, exist_ok=True)
    fs = _frame_files(cfg)
    tt = pd.DataFrame(json.load(open(raw_dir(cfg) / f"frame_times_{plot_name(cfg).replace(' ', '')}.json")))
    tt["t_local"] = pd.to_datetime(tt.exif_time, format="%Y:%m:%d %H:%M:%S") + pd.Timedelta(hours=CAMERA_CLOCK_OFFSET_H)
    tmap = dict(zip(tt.frame, tt.t_local))
    res = cfg.grid_res_m
    # target grid = bounding box of all frame footprints, snapped to res
    xs, ys = [], []
    for f in fs:
        with rasterio.open(f) as ds:
            for c, r in [(0, 0), (ds.width, 0), (0, ds.height), (ds.width, ds.height)]:
                x, y = ds.transform * (c, r); xs.append(x); ys.append(y)
            crs = ds.crs
    x0, x1 = np.floor(min(xs) / res) * res, np.ceil(max(xs) / res) * res
    y0, y1 = np.floor(min(ys) / res) * res, np.ceil(max(ys) / res) * res
    W, H = int(round((x1 - x0) / res)), int(round((y1 - y0) / res))
    dst_tr = Affine(res, 0, x0, 0, -res, y1)
    n = len(fs)
    Tmax = np.full((n, H, W), np.nan, np.float32)
    Tavg = np.full((n, H, W), np.nan, np.float32)
    frames, times = [], []
    for k, f in enumerate(fs):
        fr = int(re.search(r"IRX_(\d+)", f.name).group(1))
        with rasterio.open(f) as ds:
            a = ds.read(1).astype(np.float32)
            a[(a <= -270) | ~np.isfinite(a)] = np.nan  # DN 0 (stabilisation border) -> -275 C
            for dst, rs in [(Tmax[k], Resampling.max), (Tavg[k], Resampling.average)]:
                reproject(a, dst, src_transform=ds.transform, src_crs=ds.crs, dst_transform=dst_tr, dst_crs=crs,
                          src_nodata=np.nan, dst_nodata=np.nan, resampling=rs)
        frames.append(fr)
        t = tmap[fr]
        times.append((t - t.normalize()).total_seconds())
    for A in (Tmax, Tavg):
        A[A < 5] = np.nan  # resampling fills outside-footprint cells with 0 — physically impossible here (~29 C ambient)
    valid = np.isfinite(Tavg)
    np.savez_compressed(out / "frames.npz", Tmax=Tmax, Tavg=Tavg, times=np.array(times), frames=np.array(frames),
                        valid_frac=valid.mean(0).astype(np.float32))
    grid = dict(transform=list(dst_tr)[:6], width=W, height=H, crs=str(crs), res=res)
    (out / "grid.json").write_text(json.dumps(grid, indent=1))
    dts = np.diff(times)
    print(f"[ingest] {n} frames {frames[0]}-{frames[-1]}, {pd.Timestamp(tmap[frames[0]]).strftime('%H:%M:%S')}-"
          f"{pd.Timestamp(tmap[frames[-1]]).strftime('%H:%M:%S')} local; dt median {np.median(dts):.1f} s, gaps "
          f"{[(int(frames[i + 1]), int(d)) for i, d in enumerate(dts) if d > 10]}; grid {W}x{H} @ {res} m", file=sys.stderr)
    return dict(Tmax=Tmax, Tavg=Tavg, times=np.array(times), frames=np.array(frames), grid=grid)


# ----------------------------------------------------------------------------
# build: snapshots, masks, ignitions, wind, fronts  (same artefacts as RxCADRE)
# ----------------------------------------------------------------------------
# Wind.  The only on-site record is the burn crew's hand note (Hammock Wx 2/28,
# metadata/Hammock Wx.jpeg): 13:00 81 F 58 % 4 mph; 14:00 84 F 54 % 4 mph;
# 15:00 84 F 52 % 1 mph (speed only, in-stand eye level).  Direction comes from
# Tallahassee airport (KTLH, 30 km S, 10 m open): 240 deg at 15:53 and 16:53.
ONSITE_WX = dict(T_F=84.0, RH=52.0, wind_mph=1.0, time="15:00")
# the crew's whole weather note (metadata/Hammock Wx.jpeg), local time EST
ONSITE_NOTES = [dict(time="13:00", t=46800, T_F=81.0, RH=58.0, wind_mph=4.0, sky="broken clouds"),
                dict(time="14:00", t=50400, T_F=84.0, RH=54.0, wind_mph=4.0, sky="broken clouds"),
                dict(time="15:00", t=54000, T_F=84.0, RH=52.0, wind_mph=1.0, sky="broken clouds")]
def site_air(cfg=None):
    """Air temperature (deg C) and relative humidity (%) during the burn: the crew's 15:00 on-site reading."""
    return (ONSITE_WX["T_F"] - 32) / 1.8, ONSITE_WX["RH"]


SITE_GAS_T_BG = 30.0  # deg C, unburnt background (median frame temperature ~29-30 C)


def wind_from_metar(cfg: Config, t0: float, t1: float):
    m = pd.read_csv(raw_dir(cfg) / "KTLH_20230228.csv", na_values="M")
    m["sod"] = (pd.to_datetime(m.valid) - pd.to_datetime(m.valid).dt.normalize()).dt.total_seconds()
    w = m[(m.sod >= t0 - 3600) & (m.sod <= t1 + 3600)]
    rad = np.radians(w.drct)
    d = float(np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360)
    return d, float(w.sknt.mean() * 0.514444), w


def radiant_flux(T_c):
    """Brightness temperature (deg C) -> radiant exitance above background (W/m², emissivity 1).
    A proxy for FRFD: the camera's ~8-14 um brightness temperature saturates near 600 C."""
    Tk = np.nan_to_num(T_c, nan=SITE_GAS_T_BG) + 273.15
    return np.maximum(SIGMA * (Tk ** 4 - (SITE_GAS_T_BG + 273.15) ** 4), 0.0)


def build(cfg: Config) -> dict:
    from .fronts import extract_front, measure_advance
    out = cfg.interim_dir
    d = np.load(out / "frames.npz")
    Tmax, Tavg, times, frames = d["Tmax"], d["Tavg"], d["times"], d["frames"]
    grid = json.loads((out / "grid.json").read_text())
    res = grid["res"]
    a, _, x0, _, e, y1 = grid["transform"]
    H, W = Tmax.shape[1:]
    valid = ndi.binary_erosion(d["valid_frac"] >= 0.98, iterations=2)

    # --- burning / burnt ----------------------------------------------------
    hot = np.nan_to_num(Tmax) > cfg.thr_delta
    hot = np.array([morphology.remove_small_objects(h, 3) for h in hot])
    first = np.where(hot.any(0), hot.argmax(0), -1)
    arrival = np.where(first >= 0, times[np.maximum(first, 0)], np.nan)
    burnt0 = ((np.nan_to_num(Tavg[0]) > cfg.thr_burnt0) | hot[0]) & valid
    burnt0 = morphology.remove_small_holes(morphology.remove_small_objects(burnt0, cfg.min_blob_px), cfg.max_hole_px)
    ever = burnt0 | np.isfinite(arrival)
    # tree crowns: never hot, but enclosed by burnt ground -> ground unobservable, not "unburnt fuel"
    enclosed = ndi.binary_fill_holes(ndi.binary_closing(ever, iterations=2))
    crown = enclosed & ~ever & (np.nanmax(Tmax, 0) < cfg.thr_burnt0)
    crown = ndi.binary_dilation(morphology.remove_small_objects(crown, 4), iterations=1)
    fuel = valid & ~crown

    # --- snapshots ----------------------------------------------------------
    ts = np.arange(times[0], times[-1] + 1e-6, cfg.snapshot_s)
    fidx = np.array([int(np.argmin(np.abs(times - t))) for t in ts])
    fidx = np.unique(fidx)
    n = len(fidx)
    burnt = np.zeros((n, H, W), bool)
    active = np.zeros((n, H, W), bool)
    acc = np.zeros((H, W), bool)
    for s, k in enumerate(fidx):
        b = (burnt0 | (np.nan_to_num(arrival, nan=np.inf) <= times[k])) & fuel
        b = morphology.remove_small_objects(b, cfg.min_blob_px)
        acc = morphology.remove_small_holes(ndi.binary_closing(acc | b) & fuel | acc, cfg.max_hole_px) & fuel
        burnt[s] = acc
        active[s] = morphology.remove_small_objects((np.nan_to_num(Tmax[k]) > cfg.thr_active) & fuel, 2)
    tsnap = times[fidx]
    E = Tavg[fidx]                                   # display: brightness temperature (deg C)
    Ifl = np.array([radiant_flux(Tmax[k]) for k in fidx])  # feature: radiant flux proxy (W/m²)
    Tmap = np.repeat(tsnap[:, None, None], H, 1).repeat(W, 2).astype(np.float32)

    # --- ignitions / spots: new fire far from all earlier burnt area -----------
    ign_rows = []
    for s in range(n - 1):
        new = burnt[s + 1] & ~burnt[s]
        if not new.any():
            continue
        dist = ndi.distance_transform_edt(~burnt[s]) * res if burnt[s].any() else np.full((H, W), np.inf)
        lab, nl = ndi.label(new, structure=np.ones((3, 3)))
        for c in range(1, nl + 1):
            comp = lab == c
            if dist[comp].min() <= cfg.ign_detect_m or comp.sum() < 4:
                continue
            tc = np.nanmin(np.where(comp, arrival, np.nan))
            src = comp & (arrival <= tc + 10.0)  # pixels lit in the first ~2 frames = the ignition itself
            rr, cc = np.nonzero(src)
            for r_, c_ in zip(rr, cc):
                ign_rows.append(dict(row=float(r_), col=float(c_), t_utc=float(arrival[r_, c_]), interval=s,
                                     x=x0 + (c_ + 0.5) * a, y=y1 + (r_ + 0.5) * e))
    ign = pd.DataFrame(ign_rows, columns=["row", "col", "t_utc", "interval", "x", "y"])

    # --- wind -------------------------------------------------------------------
    wd, ws_open, _ = wind_from_metar(cfg, tsnap[0], tsnap[-1])
    U_mid = ONSITE_WX["wind_mph"] * 0.44704
    W_rows = [dict(pass_id=s, t0=tsnap[s], t1=tsnap[s + 1], dt=tsnap[s + 1] - tsnap[s], U=U_mid, U_vec=U_mid,
                   wd_from=wd, T_air=(ONSITE_WX["T_F"] - 32) / 1.8, RH=ONSITE_WX["RH"], U_open10m=ws_open)
              for s in range(n - 1)]
    Wdf = pd.DataFrame(W_rows)

    # --- fronts + ROS samples ------------------------------------------------------
    tile = cfg.tile_m / res
    fronts = []
    for s in range(n):
        fp = extract_front(burnt[s], E[s], Tmap[s], fuel, cfg, res, I_field=Ifl[s], act_field=active[s])
        if fp.empty:
            continue
        fp["pass_id"] = s
        if s < n - 1:
            adv = measure_advance(fp, burnt[s], burnt[s + 1], Tmap[s + 1], fuel, cfg, res)
            for k_, v in adv.items():
                fp[k_] = np.asarray(v)
            fp["dt_s"] = fp.t1 - fp.t
            fp["ros"] = fp.d_m / fp.dt_s
            g = ign[ign.interval == s]
            if len(g):
                tree = cKDTree(g[["row", "col"]].values)
                dmin = np.full(len(fp), np.inf)
                for q in np.linspace(0, 1, 6):
                    pr = fp.row + q * (fp.end_row - fp.row)
                    pc = fp.col + q * (fp.end_col - fp.col)
                    dmin = np.minimum(dmin, tree.query(np.c_[pr, pc])[0] * res)
                fp["ign"] = dmin < cfg.ign_exclude_m
            else:
                fp["ign"] = False
            fp["U"], fp["wd_from"] = U_mid, wd
            fp["theta"] = (fp.phi - (wd + 180) % 360 + 180) % 360 - 180
            fp["phase"] = "ignition" if len(g) else "spread"
        fp["x"] = x0 + (fp.col + 0.5) * a
        fp["y"] = y1 + (fp.row + 0.5) * e
        fp["tile"] = (fp.row // tile).astype(int) * 1000 + (fp.col // tile).astype(int)
        fronts.append(fp)
    F = pd.concat(fronts, ignore_index=True)
    flag = lambda c: F.get(c, False).fillna(True).astype(bool)
    F["clean"] = (F.active & ~F.edge & ~flag("merge") & ~flag("ign") & ~flag("long") & ~flag("off_unit")
                  & F.ros.notna() & (F.dt_s > 0.5 * cfg.snapshot_s))

    # --- threshold sensitivity (final burnt area vs the "hot" threshold) ----------
    sens = []
    for thr in (80, 100, 120, 160, 200, 300):
        h = np.nan_to_num(Tmax) > thr
        for s, k in enumerate(fidx):
            sens.append(dict(kind="delta", thr=thr, pass_id=s, burnt_ha=((burnt0 | h[:k + 1].any(0)) & fuel).sum() * res * res / 1e4))
    for thr in (200, 250, 300, 400, 500):
        for s, k in enumerate(fidx):
            sens.append(dict(kind="active", thr=thr, pass_id=s, burnt_ha=np.nan,
                             active_ha=((np.nan_to_num(Tmax[k]) > thr) & fuel).sum() * res * res / 1e4))

    np.savez_compressed(out / "masks.npz", E=E.astype(np.float32), T=Tmap, burnt=burnt, active=active, unit=fuel,
                        n_pre=cfg.first_snapshot, pre=np.zeros((H, W), np.float32), crown=crown, burnt0=burnt0,
                        arrival=np.nan_to_num(arrival, nan=-1).astype(np.float32), I=Ifl.astype(np.float32))
    ov = pd.DataFrame(dict(pass_id=np.arange(n), t_unit_mean=tsnap, frames_on_unit=frames[fidx].astype(str),
                           t_local=[f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{int(t % 60):02d}" for t in tsnap]))
    ov.to_csv(out / "overpasses.csv", index=False)
    F.to_csv(out / "front_points.csv", index=False)
    Wdf.to_csv(out / "wind_intervals.csv", index=False)
    ign.to_csv(out / "ignition_points.csv", index=False)
    pd.DataFrame(sens).to_csv(out / "threshold_sensitivity.csv", index=False)
    return dict(F=F, W=Wdf, ign=ign, burnt=burnt, active=active, fuel=fuel, crown=crown, tsnap=tsnap, fidx=fidx, n=n)


def report(r):
    F, ign, burnt, res = r["F"], r["ign"], r["burnt"], 0.5
    print(f"snapshots {r['n']} ({r['tsnap'][0]:.0f}-{r['tsnap'][-1]:.0f} s); fuel {r['fuel'].sum() * res * res:.0f} m²; "
          f"crowns masked {r['crown'].sum() * res * res:.0f} m²; burnt first/last {burnt[0].sum() * res * res:.0f} / {burnt[-1].sum() * res * res:.0f} m²")
    print("ignition/spot sources by interval:", ign.groupby("interval").size().to_dict())
    g = F[F.ros.notna()].groupby("pass_id")
    s = pd.DataFrame(dict(front=g.size(), active=g.active.sum(), clean=g.clean.sum(),
                          ros_p50=g.apply(lambda d: d.ros[d.clean].median()), ros_p90=g.apply(lambda d: d.ros[d.clean].quantile(0.9))))
    print(s.round(3).to_string())
    c = F[F.clean].copy()
    c["tb"] = pd.cut(c.theta.abs(), [0, 30, 60, 90, 120, 150, 180], include_lowest=True)
    print("clean", len(c)); print(c.groupby("tb", observed=True).ros.describe(percentiles=[.5, .9]).round(3).to_string())


if __name__ == "__main__":
    cfg = Config.for_site(sys.argv[1] if len(sys.argv) > 1 else "HH1")
    report(build(cfg))
