"""Model inputs on the 0.5 m analysis grid ("landscape", in FlamMap terms).

* Terrain: USGS 3DEP 1 m lidar DEM (exportImage service), smoothed with a
  2 m Gaussian (fire-scale terrain, not micro-relief) -> slope and upslope
  azimuth.
* Fuels and canopy: LANDFIRE 2022 (LF 2.3.0, conditions at the end of 2022,
  i.e. just before this February 2023 burn): Scott & Burgan (2005) fuel model
  (FBFM40), canopy cover, canopy height, canopy base height, canopy bulk
  density.  30 m pixels, sampled nearest-neighbour onto the 0.5 m grid.
* Wind reduction: Albini & Baughman (1979) / BehavePlus wind adjustment
  factor from the canopy (sheltered when the crown fill fraction
  f = CC * CR / 3 > 0.05, CR = (CH - CBH) / CH) or from the fuel-bed depth
  (unsheltered); midflame wind = WAF * 20-ft wind.
* Fine dead fuel moisture (1-h, fraction) from the thermal imagery: the
  pre-fire surface brightness temperature of each pixel sets the humidity at
  the fuel surface, RH_fuel = e_air / e_sat(T_fuel) (air vapour pressure from
  the crew's on-site temperature / humidity), and the Simard (1968)
  equilibrium moisture content follows.  Sunlit, warm patches come out drier
  than cool, shaded ones.  This is a physically based estimate, not a
  measurement (no fuel-moisture samples were taken on the plot).

All layers are snapped to the uint8 grids used by the web page (QUANT), so
Python and the browser compute every model from identical inputs.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import requests
from scipy import ndimage as ndi

from .config import Config

DEM_URL = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer/exportImage"
LF_URL = "https://edcintl.cr.usgs.gov/arcgis/rest/services/Landfire_LF2022/LF2022_{layer}_CONUS/ImageServer/exportImage"
LF_LAYERS = ["FBFM40", "CC", "CH", "CBH", "CBD"]
# page basemap: Esri World Topographic Map (attribution required, shown on the page)
BASEMAP_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer"
BASEMAP_MARGIN_M, BASEMAP_RES_M = 400.0, 0.5
TERRAIN3D_RES_M = 3.0   # 3DEP elevation over the basemap extent: relief shading and the 3D view
UA = {"User-Agent": "fire-spread-pipeline/0.2"}

# uint8 quantisation shared with the web page: value = lo + q * step
QUANT = {
    "m1": (0.02, 0.0005),          # 1-h dead fuel moisture, fraction (2 % ... 14.75 %)
    "tan_slope": (0.0, 0.002),     # tan(slope) (0 ... 0.51)
    "up_az": (0.0, 360.0 / 256),   # upslope azimuth, deg
    "waf": (0.0, 0.002),           # wind adjustment factor
    "cc": (0.0, 1.0),              # canopy cover, %
}
# fuel-model index used everywhere (LANDFIRE FBFM40 code -> name)
FBFM40 = {161: "TU1", 162: "TU2", 183: "TL3", 165: "TU5", 182: "TL2", 186: "TL6", 188: "TL8", 142: "SH2", 122: "GS2"}
NONBURN = 255
PRE_FIRE_GAP_S = 120.0   # frames closer than this to the pixel's fire arrival are not used for its pre-fire temperature
PRE_FIRE_DIST_M = 5.0    # ... nor frames in which burnt ground is closer than this (radiant preheating)


def raw_dir(cfg: Config) -> Path:
    return cfg.raw_dir / "landscape"


def _grid(cfg: Config):
    g = json.loads((cfg.interim_dir / "grid.json").read_text())
    a, _, x0, _, e, y1 = g["transform"]
    return g, x0, y1, x0 + g["width"] * a, y1 + g["height"] * e


def _get_tiff(url, params, path: Path):
    if path.exists():
        return
    r = requests.get(url, params=params, headers=UA, timeout=120)
    r.raise_for_status()
    if not r.headers.get("content-type", "").startswith("image/tiff"):
        raise RuntimeError(f"{url}: unexpected response {r.headers.get('content-type')}: {r.text[:200]}")
    path.write_bytes(r.content)


def download(cfg: Config, margin_m=20.0) -> None:
    out = raw_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    g, x0, y1, x1, y0 = _grid(cfg)
    res, crs = g["res"], g["crs"].split(":")[1]
    # DEM with a margin (for the smoothing / gradient), bilinear onto the 0.5 m lattice
    nx, ny = g["width"] + int(2 * margin_m / res), g["height"] + int(2 * margin_m / res)
    _get_tiff(DEM_URL, dict(bbox=f"{x0 - margin_m},{y0 - margin_m},{x1 + margin_m},{y1 + margin_m}", bboxSR=crs, imageSR=crs,
                            size=f"{nx},{ny}", format="tiff", pixelType="F32", interpolation="RSP_BilinearInterpolation",
                            noDataInterpretation="esriNoDataMatchAny", f="image"), out / "dem_3dep.tif")
    for L in LF_LAYERS:  # LANDFIRE exactly on the analysis grid, nearest neighbour
        _get_tiff(LF_URL.format(layer=L), dict(bbox=f"{x0},{y0},{x1},{y1}", bboxSR=crs, imageSR=crs,
                                               size=f"{g['width']},{g['height']}", format="tiff",
                                               interpolation="RSP_NearestNeighbor", f="image"), out / f"lf2022_{L.lower()}.tif")
    download_basemap(cfg)
    download_terrain3d(cfg)
    print(f"[landscape] downloaded DEM + LANDFIRE {LF_LAYERS} + basemap -> {out}", file=sys.stderr)


def download_basemap(cfg: Config):
    """Esri World Topographic Map around the plot, exported in the grid's CRS (one JPEG + its bounds)."""
    out = raw_dir(cfg)
    jpg, meta = out / "basemap_esri_world_topo.jpg", out / "basemap_esri_world_topo.json"
    if jpg.exists() and meta.exists():
        return
    g, x0, y1, x1, y0 = _grid(cfg)
    crs = g["crs"].split(":")[1]
    m = BASEMAP_MARGIN_M
    bb = (x0 - m, y0 - m, x1 + m, y1 + m)
    size = (int(round((bb[2] - bb[0]) / BASEMAP_RES_M)), int(round((bb[3] - bb[1]) / BASEMAP_RES_M)))
    r = requests.get(BASEMAP_URL + "/export", headers=UA, timeout=120, params=dict(
        bbox=",".join(f"{v:.2f}" for v in bb), bboxSR=crs, imageSR=crs, size=f"{size[0]},{size[1]}", format="jpg", f="image"))
    r.raise_for_status()
    if not r.headers.get("content-type", "").startswith("image/"):
        raise RuntimeError(f"basemap export failed: {r.text[:200]}")
    jpg.write_bytes(r.content)
    info = requests.get(BASEMAP_URL, params=dict(f="json"), headers=UA, timeout=60).json()
    meta.write_text(json.dumps(dict(bounds=bb, size=size, crs=g["crs"], name="Esri World Topographic Map",
                                    attribution=info.get("copyrightText", "Esri"))))


def download_terrain3d(cfg: Config):
    """USGS 3DEP elevation (bilinear, 3 m) over exactly the basemap extent."""
    out = raw_dir(cfg) / "dem_3dep_3m_basemap_extent.tif"
    if out.exists():
        return out
    info = json.loads((raw_dir(cfg) / "basemap_esri_world_topo.json").read_text())
    bb, crs = info["bounds"], info["crs"].split(":")[1]
    nx, ny = int(round((bb[2] - bb[0]) / TERRAIN3D_RES_M)), int(round((bb[3] - bb[1]) / TERRAIN3D_RES_M))
    _get_tiff(DEM_URL, dict(bbox=",".join(f"{v:.2f}" for v in bb), bboxSR=crs, imageSR=crs, size=f"{nx},{ny}", format="tiff",
                            pixelType="F32", interpolation="RSP_BilinearInterpolation", noDataInterpretation="esriNoDataMatchAny",
                            f="image"), out)
    return out


def _read(path):
    import rasterio
    with rasterio.open(path) as d:
        return d.read(1).astype(float)


def snap(name, v):
    lo, st = QUANT[name]
    return lo + np.clip(np.round((np.asarray(v, float) - lo) / st), 0, 255) * st


def to_uint8(name, v):
    lo, st = QUANT[name]
    return np.clip(np.round((np.asarray(v, float) - lo) / st), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# physics helpers
# ---------------------------------------------------------------------------
def e_sat_kpa(T_c):
    return 0.6108 * np.exp(17.27 * T_c / (T_c + 237.3))


def emc_simard_arr(T_F, RH):
    """Vectorised Simard (1968) EMC (fraction), RH in %."""
    T_F, RH = np.asarray(T_F, float), np.asarray(RH, float)
    lo = 0.03229 + 0.281073 * RH - 0.000578 * RH * T_F
    mid = 2.22749 + 0.160107 * RH - 0.01478 * T_F
    hi = 21.0606 + 0.005565 * RH ** 2 - 0.00035 * RH * T_F - 0.483199 * RH
    return np.where(RH < 10, lo, np.where(RH < 50, mid, hi)) / 100.0


def fuel_moisture_from_thermal(T_fuel_c, T_air_c, RH_air_pct):
    """1-h dead fuel moisture (fraction) from fuel surface temperature and air T / RH."""
    e_a = RH_air_pct / 100.0 * e_sat_kpa(T_air_c)
    rh_f = np.clip(100.0 * e_a / e_sat_kpa(T_fuel_c), 1.0, 100.0)
    return emc_simard_arr(T_fuel_c * 9 / 5 + 32, rh_f), rh_f


def waf_canopy(cc_pct, ch_m, cbh_m, depth_ft):
    """BehavePlus / Andrews (2012) wind adjustment factor: sheltered (canopy) where the crown
    fill fraction f > 0.05, else unsheltered (Albini & Baughman 1979, fuel-bed depth)."""
    H_ft = np.maximum(ch_m, 0.1) / 0.3048
    cr = np.clip((ch_m - cbh_m) / np.maximum(ch_m, 0.1), 0.0, 1.0)
    f = cc_pct / 100.0 * cr / 3.0
    shel = 0.555 / (np.sqrt(np.maximum(f * H_ft, 1e-9)) * np.log((20 + 0.36 * H_ft) / (0.13 * H_ft)))
    unshel = 1.83 / np.log((20 + 0.36 * depth_ft) / (0.13 * depth_ft))
    return np.where((f > 0.05) & (ch_m > 0), shel, unshel), f


# ---------------------------------------------------------------------------
def prefire_temperature(cfg: Config, frame_ok=None, min_frames=6, return_count=False):
    """Mean brightness temperature of each pixel before the fire: pre-ignition frames only, at least
    PRE_FIRE_GAP_S before the pixel's fire arrival and with no burnt ground within PRE_FIRE_DIST_M.
    frame_ok: optional boolean mask over frames (e.g. to split the record in two)."""
    fr = np.load(cfg.interim_dir / "frames.npz")
    mk = np.load(cfg.interim_dir / "masks.npz")
    Tavg, times = fr["Tavg"], fr["times"]
    arrival, burnt = mk["arrival"], mk["burnt"]
    res = json.loads((cfg.interim_dir / "grid.json").read_text())["res"]
    H, W = Tavg.shape[1:]
    t_ign = times[0] + cfg.first_snapshot * cfg.snapshot_s
    tsnap = times[0] + np.arange(burnt.shape[0]) * cfg.snapshot_s
    acc, cnt = np.zeros((H, W)), np.zeros((H, W))
    arr = np.where(arrival > 0, arrival, np.inf)
    dist_cache = {}
    for f, t in enumerate(times):
        if t >= t_ign:
            break
        if frame_ok is not None and not frame_ok[f]:
            continue
        s = int(np.clip(np.searchsorted(tsnap, t, side="right") - 1, 0, len(tsnap) - 1))
        if s not in dist_cache:
            dist_cache[s] = ndi.distance_transform_edt(~burnt[s]) * res
        ok = (t <= arr - PRE_FIRE_GAP_S) & (dist_cache[s] > PRE_FIRE_DIST_M) & np.isfinite(Tavg[f])
        acc[ok] += Tavg[f][ok]
        cnt[ok] += 1
    T = np.where(cnt >= min_frames, acc / np.maximum(cnt, 1), np.nan)
    return (T, cnt) if return_count else T


def build(cfg: Config, margin_m=20.0) -> dict:
    from .physics import SB40
    g, *_ = _grid(cfg)
    res, H, W = g["res"], g["height"], g["width"]
    rd = raw_dir(cfg)
    pad = int(margin_m / res)

    # --- terrain
    dem = _read(rd / "dem_3dep.tif")
    dem[dem < -1000] = np.nan
    dem = np.where(np.isnan(dem), np.nanmean(dem), dem)
    zs = ndi.gaussian_filter(dem, 2.0 / res)
    gy, gx = np.gradient(zs, res)            # rows run south, so north = -gy
    dz_e, dz_n = gx, -gy
    tan_s = np.hypot(dz_e, dz_n)
    up_az = np.degrees(np.arctan2(dz_e, dz_n)) % 360
    crop = (slice(pad, pad + H), slice(pad, pad + W))
    elev, tan_s, up_az = dem[crop], tan_s[crop], up_az[crop]

    # --- fuels / canopy (LANDFIRE codes; CH, CBH in m*10, CBD in kg/m3*100)
    fb = _read(rd / "lf2022_fbfm40.tif").astype(int)
    cc = _read(rd / "lf2022_cc.tif")
    ch = _read(rd / "lf2022_ch.tif") / 10.0
    cbh = _read(rd / "lf2022_cbh.tif") / 10.0
    cbd = _read(rd / "lf2022_cbd.tif") / 100.0
    names = [FBFM40.get(int(c)) for c in np.unique(fb)]
    if any(n is None or n not in SB40 for n in names):
        raise RuntimeError(f"LANDFIRE fuel codes {np.unique(fb)} not all in the fuel-model table")
    codes = sorted(np.unique(fb).tolist())
    fuel_names = [FBFM40[c] for c in codes]
    fuel_idx = np.searchsorted(codes, fb).astype(np.uint8)
    depth = np.array([SB40[n].depth for n in fuel_names])[fuel_idx]
    waf, crown_fill = waf_canopy(cc, ch, cbh, depth)

    # --- fine dead fuel moisture from the pre-fire thermal frames
    T_pre, cnt = prefire_temperature(cfg, return_count=True)
    good = np.isfinite(T_pre)
    # fill pixels without enough clean pre-fire frames (burnt early / next to the fire): normalised Gaussian
    num = ndi.gaussian_filter(np.where(good, T_pre, 0.0), 3.0 / res)
    den = ndi.gaussian_filter(good.astype(float), 3.0 / res)
    T_fill = np.where(den > 1e-3, num / np.maximum(den, 1e-9), np.nan)
    T_fill = np.where(np.isfinite(T_fill), T_fill, np.nanmedian(T_pre))
    T_fuel = np.where(good, T_pre, T_fill)
    T_fuel = ndi.gaussian_filter(T_fuel, 0.5 / res)  # light de-noising (frame noise ~0.1 C after averaging)
    mk = np.load(cfg.interim_dir / "masks.npz")
    from .flame3 import site_air
    T_air, RH_air = site_air(cfg)
    m1, rh_fuel = fuel_moisture_from_thermal(T_fuel, T_air, RH_air)

    # --- snap to the shared quantisation
    out = dict(
        m1=snap("m1", m1), tan_slope=snap("tan_slope", tan_s), up_az=snap("up_az", up_az), waf=snap("waf", waf),
        cc=snap("cc", cc), fuel=fuel_idx, elev=elev.astype(np.float32), ch=ch.astype(np.float32), cbh=cbh.astype(np.float32),
        cbd=cbd.astype(np.float32), crown_fill=crown_fill.astype(np.float32), T_fuel=T_fuel.astype(np.float32),
        rh_fuel=rh_fuel.astype(np.float32), n_pre_frames=cnt.astype(np.int16), fbfm40=fb.astype(np.int16))
    np.savez_compressed(cfg.interim_dir / "landscape.npz", **out)
    unit = mk["unit"].astype(bool)
    meta = dict(fuel_codes=codes, fuel_names=fuel_names, quant=QUANT, T_air=T_air, RH_air=RH_air,
                sources=dict(dem="USGS 3DEP 1 m lidar DEM (FL Hurricane Michael 2020)", landfire="LANDFIRE 2022 (LF 2.3.0)"),
                stats={k: [float(np.nanpercentile(v[unit], q)) for q in (5, 50, 95)] for k, v in
                       dict(m1=out["m1"], T_fuel=T_fuel, rh_fuel=rh_fuel, slope_pct=100 * out["tan_slope"], waf=out["waf"],
                            cc=cc, ch=ch, cbh=cbh, elev=elev).items()},
                fuel_share={n: float((fuel_idx[unit] == i).mean()) for i, n in enumerate(fuel_names)})
    (cfg.interim_dir / "landscape.json").write_text(json.dumps(meta, indent=1))
    return dict(meta=meta, **out)


SNAP_T_MAX_C = 45.0     # hotter unburnt pixels are being heated by the fire: the equilibrium estimate no longer applies
SNAP_DIST_M = 2.0       # ... and so are pixels this close to burnt ground
SNAP_FILL_SIGMA_M = 3.0 # gap filling from this snapshot's valid neighbours
# uint8 code of the per-snapshot surface temperature maps: 0.1 C steps from 20 C up to 40 C (where moisture is
# sensitive), then log steps (~5 %) from 40 to 625 C (flames; the camera saturates near 605 C)
T_TQ = dict(lin_lo=20.0, lin_st=0.1, split=200, log_lo=40.0, log_hi=625.0, log_n=55)


def t_to_q(T):
    T = np.asarray(T, float)
    lin = np.clip(np.round((T - T_TQ["lin_lo"]) / T_TQ["lin_st"]), 0, T_TQ["split"] - 1)
    lg = T_TQ["split"] + np.clip(np.round(T_TQ["log_n"] * np.log(np.maximum(T, T_TQ["log_lo"]) / T_TQ["log_lo"])
                                          / np.log(T_TQ["log_hi"] / T_TQ["log_lo"])), 0, T_TQ["log_n"])
    return np.where(T < T_TQ["log_lo"], lin, lg).astype(np.uint8)


def moisture_series(cfg: Config) -> dict:
    """Fine dead fuel moisture for every snapshot: the same humidity-at-the-fuel estimate as the pre-fire map,
    from the mean brightness temperature of the 30 s of frames ending at the snapshot (only data available at
    forecast time).  Where the pixel is burnt, within SNAP_DIST_M of burnt ground, or hotter than SNAP_T_MAX_C,
    it takes this snapshot's estimate from valid neighbours within a few metres (normalised Gaussian), and only
    where there are none keeps its last value (starting from the pre-fire map).  Stored as 1 m blocks (2 x 2 mean, aligned to the forecast crop box), snapped to QUANT."""
    from .flame3 import site_air
    from .plots import crop_box
    fr = np.load(cfg.interim_dir / "frames.npz")
    mk = np.load(cfg.interim_dir / "masks.npz")
    import pandas as pd
    tsnap = pd.read_csv(cfg.interim_dir / "overpasses.csv").t_unit_mean.values
    Tavg, Tmx_all, times, burnt = fr["Tavg"], fr["Tmax"], fr["times"], mk["burnt"]
    res = json.loads((cfg.interim_dir / "grid.json").read_text())["res"]
    T_air, RH_air = site_air(cfg)
    m_prev = load(cfg)["m1"].astype(float)
    r0, r1, c0, c1 = crop_box(mk["unit"])
    H2, W2 = (r1 - r0 + 1) // 2, (c1 - c0 + 1) // 2
    q = np.zeros((len(tsnap), H2, W2), np.uint8)
    tq = np.zeros((len(tsnap), H2, W2), np.uint8)    # the snapshot's surface temperature (display), 20 + 0.1 * q deg C
    valid_frac = np.zeros(len(tsnap))
    for s_, ts in enumerate(tsnap):
        sel = (times > ts - cfg.snapshot_s + 1e-6) & (times <= ts + 1e-6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            T = np.nanmean(Tavg[sel], 0)
        near = ndi.distance_transform_edt(~burnt[s_]) * res < SNAP_DIST_M
        ok = ~burnt[s_] & ~near & np.isfinite(T) & (T < SNAP_T_MAX_C)
        m_obs = fuel_moisture_from_thermal(np.where(ok, T, 30.0), T_air, RH_air)[0]
        # cells without a valid reading (burnt, next to the fire, too hot): this snapshot's estimate from valid
        # neighbours within a few metres (normalised Gaussian, sigma 3 m); the last value only where there is none
        num = ndi.gaussian_filter(np.where(ok, m_obs, 0.0), SNAP_FILL_SIGMA_M / res)
        den = ndi.gaussian_filter(ok.astype(float), SNAP_FILL_SIGMA_M / res)
        fill = np.where(den > 0.02, num / np.maximum(den, 1e-9), m_prev)
        m = np.where(ok, m_obs, fill)
        m_prev = m
        valid_frac[s_] = ok[mk["unit"]].mean()
        c = m[r0:r1, c0:c1]
        cp = np.pad(c, ((0, 2 * H2 - c.shape[0]), (0, 2 * W2 - c.shape[1])), mode="edge")
        q[s_] = to_uint8("m1", cp.reshape(H2, 2, W2, 2).mean(axis=(1, 3)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            Tpk = np.nanmax(Tmx_all[sel], 0)                       # peak in the window: flames
        disp = np.where(np.isfinite(Tpk) & (Tpk >= SNAP_T_MAX_C), Tpk, T)
        tc = np.nan_to_num(disp, nan=T_TQ["lin_lo"])[r0:r1, c0:c1]
        tp = np.pad(tc, ((0, 2 * H2 - tc.shape[0]), (0, 2 * W2 - tc.shape[1])), mode="edge")
        tq[s_] = t_to_q(tp.reshape(H2, 2, W2, 2).max(axis=(1, 3)))   # 1 m block maximum keeps the flame peaks
    np.savez_compressed(cfg.interim_dir / "moisture_t.npz", q=q, tq=tq, box=np.array([r0, r1, c0, c1]), tsnap=tsnap, valid_frac=valid_frac)
    return dict(q=q, box=(r0, r1, c0, c1), valid_frac=valid_frac)


def moisture_maps(cfg: Config, shape) -> np.ndarray:
    """(n_snapshots, H, W) moisture on the forecast crop grid (the shipped 1 m blocks, repeated)."""
    z = np.load(cfg.interim_dir / "moisture_t.npz")
    lo, st = QUANT["m1"]
    v = lo + z["q"].astype(float) * st
    return np.repeat(np.repeat(v, 2, axis=1), 2, axis=2)[:, :shape[0], :shape[1]]


def load(cfg: Config) -> dict:
    d = dict(np.load(cfg.interim_dir / "landscape.npz"))
    d["meta"] = json.loads((cfg.interim_dir / "landscape.json").read_text())
    return d


def sample(L: dict, rows, cols, key):
    """Layer value at (fractional) grid positions, nearest pixel with the web page's rounding."""
    H, W = L[key].shape
    r = np.clip(np.floor(np.asarray(rows, float) + 0.5).astype(int), 0, H - 1)
    c = np.clip(np.floor(np.asarray(cols, float) + 0.5).astype(int), 0, W - 1)
    return L[key][r, c]


def report(r):
    m = r["meta"]
    print("[landscape] fuel models:", {k: f"{v:.0%}" for k, v in m["fuel_share"].items()})
    for k, v in m["stats"].items():
        print(f"   {k:10s} p5 {v[0]:8.3f}  p50 {v[1]:8.3f}  p95 {v[2]:8.3f}")


if __name__ == "__main__":
    cfg = Config()
    download(cfg)
    report(build(cfg))
    moisture_series(cfg)
