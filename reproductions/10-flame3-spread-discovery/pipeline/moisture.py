"""Fuel moisture from satellites and weather, and checks of the thermal estimate.

Live fuel moisture (not visible to the drone's LWIR):
  * Sentinel-2 L2A (Earth Search, AWS) and Landsat 8/9 Collection 2 L2 (Planetary
    Computer): NDII = (NIR - SWIR 1.6 um) / (NIR + SWIR 1.6 um) and the Gao-type
    NDWI with the 2.2 um band, averaged over the plot's own pixels in every clear
    scene from 2019 to the day before the burn.
  * Sentinel-1 RTC (Planetary Computer): VH and VV backscatter over the plot, same
    relative orbit, as an independent (microwave) check.
  The burn-date value is placed within the plot's own 2019-2023 range (5th-95th
  percentile) and mapped linearly onto the NFDRS live-moisture limits -- herbaceous
  30-250 %, woody 60-200 % -- the relative-greenness method of Burgan & Hartford
  (1993) / Burgan et al. (1998), with the water-sensitive NDII instead of NDVI.
  The satellite sees the pine canopy and the understorey together, so the plot
  median is used (one value per plot), not the 20 m pattern.

Checks of the thermal fine dead fuel moisture (pipeline/landscape.py):
  * level vs the Canadian FFMC computed from KTLH noon weather since Sep 2022
    (Van Wagner 1987; FFMC targets the ~16:00 burning period, as here) and vs the
    screen-level Simard EMC;
  * repeatability: maps from two disjoint pre-fire time windows;
  * physical pattern: moister under denser canopy (LANDFIRE) and where Sentinel-2
    NDII is higher;
  * does it matter: physics-model skill with a uniform moisture instead of the map.
The same weather run gives the Buildup Index used by the FBP System.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import Config

ES_SEARCH = "https://earth-search.aws.element84.com/v1/search"
PC_SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
PC_TOKEN = "https://planetarycomputer.microsoft.com/api/sas/v1/token/{}"
HISTORY = ("2019-01-01T00:00:00Z", "2023-02-27T23:59:59Z")   # everything before the burn day
BURN_DATE = pd.Timestamp("2023-02-28")
LIVE_LIMITS = dict(herb=(0.30, 2.50), wood=(0.60, 2.00))     # NFDRS live fuel moisture range (fraction)
UA = {"User-Agent": "fire-spread-pipeline/0.2"}
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif,.TIF,.tiff,.TIFF"   # Sentinel-1 RTC assets are .tiff
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "4")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")


def raw_dir(cfg: Config) -> Path:
    return cfg.raw_dir / "satellite"


# ---------------------------------------------------------------------------
# STAC + windowed COG reads over the plot
# ---------------------------------------------------------------------------
def stac_search(url, body):
    feats, body = [], dict(body)
    while True:
        r = requests.post(url, json=body, headers=UA, timeout=120)
        r.raise_for_status()
        j = r.json()
        feats += j.get("features", [])
        nxt = [l for l in j.get("links", []) if l.get("rel") == "next"]
        if not nxt or not j.get("features"):
            return feats
        l = nxt[0]
        if l.get("method", "GET").upper() == "POST":
            body = {**body, **l.get("body", {})} if l.get("merge", False) or "body" in l else body
            url = l["href"]
        else:
            r = requests.get(l["href"], headers=UA, timeout=120)
            r.raise_for_status()
            j = r.json()
            feats += j.get("features", [])
            return feats


class Plot:
    """Plot footprint: analysis-grid unit mask + bounds (EPSG:26916)."""

    def __init__(self, cfg: Config, margin_m=60.0):
        g = json.loads((cfg.interim_dir / "grid.json").read_text())
        a, _, self.x0, _, e, self.y1 = g["transform"]
        self.res, self.H, self.W = a, g["height"], g["width"]
        self.unit = np.load(cfg.interim_dir / "masks.npz")["unit"].astype(bool)
        self.bounds = (self.x0 - margin_m, self.y1 + self.H * e - margin_m, self.x0 + self.W * a + margin_m, self.y1 + margin_m)

    def inside(self, xs, ys, crs):
        """Which native pixel centres (in crs) fall on the plot."""
        from rasterio.warp import transform
        X, Y = transform(crs, "EPSG:26916", list(np.ravel(xs)), list(np.ravel(ys)))
        c = np.floor((np.array(X) - self.x0) / self.res).astype(int)
        r = np.floor((self.y1 - np.array(Y)) / self.res).astype(int)
        ok = (r >= 0) & (r < self.H) & (c >= 0) & (c < self.W)
        out = np.zeros(len(c), bool)
        out[ok] = self.unit[r[ok], c[ok]]
        return out.reshape(np.shape(xs))


def read_window(href, plot: Plot):
    """Native pixels around the plot + a boolean 'on the plot' mask."""
    import rasterio
    from rasterio.warp import transform_bounds
    from rasterio.windows import from_bounds
    with rasterio.open(href) as d:
        b = transform_bounds("EPSG:26916", d.crs, *plot.bounds)
        w = from_bounds(*b, d.transform).round_offsets().round_lengths()
        a = d.read(1, window=w, boundless=True, fill_value=0).astype(float)
        tr = d.window_transform(w)
        rows, cols = np.mgrid[0:a.shape[0], 0:a.shape[1]]
        xs, ys = tr * (cols + 0.5, rows + 0.5)
        return a, plot.inside(xs, ys, d.crs), tr, d.crs


def _nd(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (a - b) / (a + b)


def s2_scene(item, plot: Plot, accept=(4, 5)):
    p = item["properties"]
    off = 0.0
    if p.get("s2:processing_baseline", "00.00") >= "04.00" and not p.get("earthsearch:boa_offset_applied", False):
        off = 1000.0
    A = item["assets"]
    scl, on, tr, crs = read_window(A["scl"]["href"], plot)
    clear = np.isin(scl, accept)
    if on.sum() == 0 or clear[on].mean() < 0.95:
        return None
    nir, *_ = read_window(A["nir08"]["href"], plot)
    s16, *_ = read_window(A["swir16"]["href"], plot)
    s22, *_ = read_window(A["swir22"]["href"], plot)
    nir, s16, s22 = [(x - off) / 1e4 for x in (nir, s16, s22)]
    ndii, ndwi = _nd(nir, s16), _nd(nir, s22)
    sel = on & clear
    return dict(date=p["datetime"][:10], sensor="Sentinel-2", id=item["id"], n_px=int(sel.sum()),
                ndii=float(np.nanmean(ndii[sel])), ndwi22=float(np.nanmean(ndwi[sel])), nir=float(np.nanmean(nir[sel])),
                _map=dict(ndii=ndii, on=on, transform=list(tr)[:6], crs=str(crs)))


def landsat_scene(item, plot: Plot, token):
    A = item["assets"]
    href = lambda k: A[k]["href"] + "?" + token
    qa, on, *_ = read_window(href("qa_pixel"), plot)
    q = qa.astype(np.int64)
    clear = ((q >> 6) & 1 == 1) & ((q >> 1) & 1 == 0) & ((q >> 2) & 1 == 0) & ((q >> 3) & 1 == 0) & ((q >> 4) & 1 == 0)
    if on.sum() == 0 or clear[on].mean() < 0.95:
        return None
    sr = lambda k: read_window(href(k), plot)[0] * 0.0000275 - 0.2
    nir, s16, s22 = sr("nir08"), sr("swir16"), sr("swir22")
    sel = on & clear
    return dict(date=item["properties"]["datetime"][:10], sensor=item["properties"]["platform"], id=item["id"], n_px=int(sel.sum()),
                ndii=float(np.nanmean(_nd(nir, s16)[sel])), ndwi22=float(np.nanmean(_nd(nir, s22)[sel])), nir=float(np.nanmean(nir[sel])))


def s1_scene(item, plot: Plot, token):
    A = item["assets"]
    vh, on, *_ = read_window(A["vh"]["href"] + "?" + token, plot)
    vv, *_ = read_window(A["vv"]["href"] + "?" + token, plot)
    sel = on & (vh > 0) & (vv > 0)
    if sel.sum() < 0.9 * on.sum() or on.sum() == 0:
        return None
    p = item["properties"]
    return dict(date=p["datetime"][:10], datetime=p["datetime"], sensor=p.get("platform", "sentinel-1"), id=item["id"],
                orbit=int(p.get("sat:relative_orbit", -1)), state=p.get("sat:orbit_state", ""), n_px=int(sel.sum()),
                vh_db=float(10 * np.log10(vh[sel].mean())), vv_db=float(10 * np.log10(vv[sel].mean())))


def _bbox_ll(plot: Plot):
    from rasterio.warp import transform_bounds
    return list(transform_bounds("EPSG:26916", "EPSG:4326", *plot.bounds))


def _run(fn, items, workers=8):
    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(lambda it: _safe(fn, it), items))
    return [o for o in out if o]


def _safe(fn, it):
    try:
        return fn(it)
    except Exception as e:  # one unreadable scene must not stop the series
        print(f"[moisture] skip {it.get('id')}: {type(e).__name__} {str(e)[:120]}", file=sys.stderr)
        return None


def download(cfg: Config):
    out = raw_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    plot = Plot(cfg)
    bbox = _bbox_ll(plot)
    dt = "/".join(HISTORY)

    if not (out / "s2_series.csv").exists():
        items = stac_search(ES_SEARCH, dict(collections=["sentinel-2-l2a"], bbox=bbox, datetime=dt, limit=200,
                                            query={"eo:cloud_cover": {"lt": 60}}))
        items = [i for i in items if "_16RGU_" in i["id"]]   # one tile -> one pixel grid for the whole series
        print(f"[moisture] Sentinel-2: {len(items)} candidate scenes", file=sys.stderr)
        rows = _run(lambda it: s2_scene(it, plot), items)
        rows.sort(key=lambda r: r["date"])
        last = rows[-1]["_map"]
        np.savez_compressed(out / "s2_last_clear.npz", ndii=last["ndii"], on=last["on"], transform=np.array(last["transform"]),
                            crs=last["crs"], date=rows[-1]["date"])
        pd.DataFrame([{k: v for k, v in r.items() if k != "_map"} for r in rows]).to_csv(out / "s2_series.csv", index=False)

    if not (out / "s2_cirrus.csv").exists():   # scenes after the last clear one where only thin cirrus covers the plot
        last = pd.read_csv(out / "s2_series.csv").date.max()
        items = stac_search(ES_SEARCH, dict(collections=["sentinel-2-l2a"], bbox=bbox, limit=50,
                                            datetime=f"{last}T23:59:59Z/{HISTORY[1]}"))
        rows = _run(lambda it: s2_scene(it, plot, accept=(10,)), [i for i in items if "_16RGU_" in i["id"]])
        pd.DataFrame([{k: v for k, v in r.items() if k != "_map"} for r in rows],
                     columns=["date", "sensor", "id", "n_px", "ndii", "ndwi22", "nir"]).to_csv(out / "s2_cirrus.csv", index=False)

    if not (out / "landsat_series.csv").exists():
        tok = requests.get(PC_TOKEN.format("landsat-c2-l2"), headers=UA, timeout=60).json()["token"]
        items = stac_search(PC_SEARCH, dict(collections=["landsat-c2-l2"], bbox=bbox, datetime=dt, limit=200,
                                            query={"platform": {"in": ["landsat-8", "landsat-9"]}, "eo:cloud_cover": {"lt": 70}}))
        print(f"[moisture] Landsat 8/9: {len(items)} candidate scenes", file=sys.stderr)
        rows = _run(lambda it: landsat_scene(it, plot, tok), items)
        pd.DataFrame(rows).sort_values("date").to_csv(out / "landsat_series.csv", index=False)

    if not (out / "s1_series.csv").exists():
        tok = requests.get(PC_TOKEN.format("sentinel-1-rtc"), headers=UA, timeout=60).json()["token"]
        items = stac_search(PC_SEARCH, dict(collections=["sentinel-1-rtc"], bbox=bbox, datetime=dt, limit=200))
        print(f"[moisture] Sentinel-1 RTC: {len(items)} candidate scenes", file=sys.stderr)
        rows = _run(lambda it: s1_scene(it, plot, tok), items)
        if not rows:
            raise RuntimeError("no Sentinel-1 scene could be read")
        pd.DataFrame(rows).sort_values("datetime").to_csv(out / "s1_series.csv", index=False)

    day = out / "KTLH_20230228_full.csv"   # every report on the burn day: temperature, dew point, humidity, wind, gusts, sky, pressure
    if not day.exists():
        r = requests.get("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py", timeout=120, headers=UA, params=dict(
            station="TLH", data=["tmpf", "dwpf", "relh", "drct", "sknt", "gust", "alti", "vsby", "skyc1", "skyl1", "wxcodes", "p01i"],
            year1=2023, month1=2, day1=28, year2=2023, month2=3, day2=1, tz="Etc/GMT+5", format="onlycomma", latlon="no",
            missing="M", trace="T", report_type=[3, 4]))
        r.raise_for_status()
        day.write_text(r.text)
    wx = out / "ktlh_hourly_2022-09_2023-02.csv"
    if not wx.exists():
        r = requests.get("https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py", timeout=120, headers=UA, params=dict(
            station="TLH", data=["tmpf", "relh", "sknt", "p01i"], year1=2022, month1=9, day1=1, year2=2023, month2=3, day2=1,
            tz="Etc/GMT+5", format="onlycomma", latlon="no", missing="M", trace="0.0001", report_type=3))
        r.raise_for_status()
        wx.write_text(r.text)


# ---------------------------------------------------------------------------
# Canadian Fire Weather Index codes (Van Wagner 1987), daily at noon LST
# ---------------------------------------------------------------------------
LE_15_33N = [7.9, 8.4, 8.9, 9.5, 9.9, 10.2, 10.1, 9.7, 9.1, 8.6, 8.1, 7.8]   # DMC day-length factors, 15-33 N
LF_N = [-1.6, -1.6, -1.6, 0.9, 3.8, 5.8, 6.4, 5.0, 2.4, 0.4, -1.6, -1.6]      # DC day-length factors, > 20 N


def ffmc_step(f0, T, H, W, ro):
    mo = 147.2 * (101 - f0) / (59.5 + f0)
    if ro > 0.5:
        rf = ro - 0.5
        add = 42.5 * rf * np.exp(-100 / (251 - mo)) * (1 - np.exp(-6.93 / rf))
        if mo > 150:
            add += 0.0015 * (mo - 150) ** 2 * rf ** 0.5
        mo = min(mo + add, 250.0)
    ed = 0.942 * H ** 0.679 + 11 * np.exp((H - 100) / 10) + 0.18 * (21.1 - T) * (1 - np.exp(-0.115 * H))
    ew = 0.618 * H ** 0.753 + 10 * np.exp((H - 100) / 10) + 0.18 * (21.1 - T) * (1 - np.exp(-0.115 * H))
    if mo > ed:
        ko = 0.424 * (1 - (H / 100) ** 1.7) + 0.0694 * W ** 0.5 * (1 - (H / 100) ** 8)
        m = ed + (mo - ed) * 10 ** (-ko * 0.581 * np.exp(0.0365 * T))
    elif mo < ew:
        k1 = 0.424 * (1 - ((100 - H) / 100) ** 1.7) + 0.0694 * W ** 0.5 * (1 - ((100 - H) / 100) ** 8)
        m = ew - (ew - mo) * 10 ** (-k1 * 0.581 * np.exp(0.0365 * T))
    else:
        m = mo
    return 59.5 * (250 - m) / (147.2 + m)


def dmc_step(p0, T, H, ro, month):
    if ro > 1.5:
        re = 0.92 * ro - 1.27
        mo = 20 + np.exp(5.6348 - p0 / 43.43)
        b = 100 / (0.5 + 0.3 * p0) if p0 <= 33 else (14 - 1.3 * np.log(p0) if p0 <= 65 else 6.2 * np.log(p0) - 17.2)
        mr = mo + 1000 * re / (48.77 + b * re)
        p0 = max(244.72 - 43.43 * np.log(mr - 20), 0.0)
    k = 1.894 * (max(T, -1.1) + 1.1) * (100 - H) * LE_15_33N[month - 1] * 1e-6
    return p0 + 100 * k


def dc_step(d0, T, ro, month):
    if ro > 2.8:
        rd = 0.83 * ro - 1.27
        qr = 800 * np.exp(-d0 / 400) + 3.937 * rd
        d0 = max(400 * np.log(800 / qr), 0.0)
    v = max(0.36 * (max(T, -2.8) + 2.8) + LF_N[month - 1], 0.0)
    return d0 + 0.5 * v


def bui_of(p, d):
    if p <= 0.4 * d:
        u = 0.8 * p * d / (p + 0.4 * d) if p + d > 0 else 0.0
    else:
        u = p - (1 - 0.8 * d / (p + 0.4 * d)) * (0.92 + (0.0114 * p) ** 1.7)
    return max(u, 0.0)


def fwi_series(cfg: Config) -> pd.DataFrame:
    w = pd.read_csv(raw_dir(cfg) / "ktlh_hourly_2022-09_2023-02.csv", na_values="M")
    w["t"] = pd.to_datetime(w["valid"])
    w["p01i"] = w["p01i"].fillna(0.0)
    days = pd.date_range("2022-09-01", "2023-02-28", freq="D")
    rows, f, p, d = [], 85.0, 6.0, 15.0           # standard start-up values; ~6 months of spin-up
    for day in days:
        noon = day + pd.Timedelta(hours=12)
        obs = w.iloc[(w.t - noon).abs().argsort()[:1]].iloc[0]
        rain = 25.4 * w[(w.t > noon - pd.Timedelta(hours=24)) & (w.t <= noon)].p01i.sum()
        T, H, W = float(obs.tmpf - 32) / 1.8, float(min(obs.relh, 100.0)), float(obs.sknt) * 1.852
        f = ffmc_step(f, T, H, W, rain)
        p = dmc_step(p, T, H, rain, day.month)
        d = dc_step(d, T, rain, day.month)
        rows.append(dict(date=day.date().isoformat(), T=T, RH=H, wind_kmh=W, rain_mm=rain, FFMC=f, DMC=p, DC=d, BUI=bui_of(p, d)))
    df = pd.DataFrame(rows)
    df["m_ffmc"] = 147.2 * (101 - df.FFMC) / (59.5 + df.FFMC) / 100.0   # fine-fuel moisture, fraction
    return df


# ---------------------------------------------------------------------------
# live fuel moisture from the satellite series
# ---------------------------------------------------------------------------
def relative_position(series: pd.Series, dates: pd.Series, value_date_max=BURN_DATE, max_age_days=21):
    """Latest value within max_age_days before the burn, placed within the 5-95th percentile range of the whole
    pre-burn record.  Returns dict(value, date, p5, p95, r, n)."""
    d = pd.to_datetime(dates)
    ok = series.notna() & (d < value_date_max)
    s, d = series[ok], d[ok]
    p5, p95 = float(np.percentile(s, 5)), float(np.percentile(s, 95))
    recent = d >= value_date_max - pd.Timedelta(days=max_age_days)
    if not recent.any():
        return dict(value=np.nan, date=None, p5=p5, p95=p95, r=np.nan, n=int(len(s)))
    i = d[recent].idxmax()
    v = float(s[i])
    return dict(value=v, date=d[i].date().isoformat(), p5=p5, p95=p95, r=float(np.clip((v - p5) / (p95 - p5), 0, 1)),
                n=int(len(s)), pct=float((s < v).mean()))


def live_moisture(r):
    return {k: lo + (hi - lo) * r for k, (lo, hi) in LIVE_LIMITS.items()}


# ---------------------------------------------------------------------------
def build(cfg: Config, sensitivity=True) -> dict:
    from . import landscape as LS
    out_dir = raw_dir(cfg)
    s2 = pd.read_csv(out_dir / "s2_series.csv")
    ls = pd.read_csv(out_dir / "landsat_series.csv")
    s1 = pd.read_csv(out_dir / "s1_series.csv")
    fwi = fwi_series(cfg)
    fwi.to_csv(cfg.interim_dir / "fwi_ktlh.csv", index=False)
    burn = fwi[fwi.date == "2023-02-28"].iloc[0]

    # --- live fuel moisture
    live = dict(
        s2_ndii=relative_position(s2.ndii, s2.date, max_age_days=45), s2_ndwi22=relative_position(s2.ndwi22, s2.date, max_age_days=45),
        landsat_ndii=relative_position(ls.ndii, ls.date, max_age_days=45) if len(ls) else None)
    ci = pd.read_csv(out_dir / "s2_cirrus.csv") if (out_dir / "s2_cirrus.csv").exists() else pd.DataFrame()
    if len(ci):   # for reference only: thin cirrus biases SWIR indices
        last = ci.sort_values("date").iloc[-1]
        p5, p95 = live["s2_ndii"]["p5"], live["s2_ndii"]["p95"]
        live["s2_ndii_cirrus"] = dict(date=last.date, value=float(last.ndii), p5=p5, p95=p95, n=live["s2_ndii"]["n"],
                                      r=float(np.clip((last.ndii - p5) / (p95 - p5), 0, 1)))
    # Sentinel-1: the relative orbit that imaged the plot last before the burn
    s1["d"] = pd.to_datetime(s1.date)
    pre = s1[s1.d < BURN_DATE]
    orb = int(pre.sort_values("datetime").iloc[-1].orbit) if len(pre) else None
    s1o = s1[s1.orbit == orb] if orb is not None else s1
    live["s1_vh"] = relative_position(s1o.vh_db, s1o.date)
    live["s1_vh"]["orbit"] = orb
    r_main = live["s2_ndii"]["r"]
    lm = live_moisture(r_main)
    live.update(r=r_main, herb=lm["herb"], wood=lm["wood"], method="Sentinel-2 NDII relative to its 2019-2023 range at the plot",
                cross=[x for x in (live["s2_ndwi22"]["r"], (live["landsat_ndii"] or {}).get("r"), live["s1_vh"]["r"]) if x is not None])

    # --- thermal dead-fuel moisture checks
    L = LS.load(cfg)
    unit = np.load(cfg.interim_dir / "masks.npz")["unit"].astype(bool)
    m1 = L["m1"]
    checks = dict(thermal_median=float(np.median(m1[unit])), thermal_p5=float(np.percentile(m1[unit], 5)),
                  thermal_p95=float(np.percentile(m1[unit], 95)), ffmc=float(burn.FFMC), m_ffmc=float(burn.m_ffmc),
                  bui=float(burn.BUI), dmc=float(burn.DMC), dc=float(burn.DC),
                  rain_7d_mm=float(fwi[fwi.date > "2023-02-21"].rain_mm.sum()),
                  last_rain=str(fwi[fwi.rain_mm > 0.5].date.iloc[-1]) if (fwi.rain_mm > 0.5).any() else None)
    from .physics import emc_simard
    from .flame3 import ONSITE_WX
    checks["emc_air"] = float(emc_simard(ONSITE_WX["T_F"], ONSITE_WX["RH"]))
    checks.update(repeatability(cfg, L, unit))
    checks.update(pattern_checks(cfg, L, unit))
    res = dict(live=live, checks=checks, fwi_burn=burn.to_dict())
    (cfg.interim_dir / "moisture.json").write_text(json.dumps(res, indent=1, default=float))
    if sensitivity:
        res["checks"]["sensitivity"] = forecast_sensitivity(cfg, checks)
        (cfg.interim_dir / "moisture.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def repeatability(cfg: Config, L, unit):
    """Thermal moisture maps from the first and second half of the clean pre-fire frames."""
    from . import landscape as LS
    from scipy import ndimage as ndi
    fr = np.load(cfg.interim_dir / "frames.npz")
    t = fr["times"]
    t_ign = t[0] + cfg.first_snapshot * cfg.snapshot_s
    mid = t[0] + 0.5 * (t_ign - t[0])
    a = LS.prefire_temperature(cfg, frame_ok=t < mid)
    b = LS.prefire_temperature(cfg, frame_ok=(t >= mid) & (t < t_ign))
    from .flame3 import site_air
    Ta, RHa = site_air(cfg)
    ma = LS.fuel_moisture_from_thermal(a, Ta, RHa)[0]
    mb = LS.fuel_moisture_from_thermal(b, Ta, RHa)[0]
    ok = unit & np.isfinite(ma) & np.isfinite(mb)
    r_px = float(np.corrcoef(ma[ok], mb[ok])[0, 1])
    k = int(round(5.0 / json.loads((cfg.interim_dir / "grid.json").read_text())["res"]))
    sa, sb = ndi.uniform_filter(np.nan_to_num(ma), k), ndi.uniform_filter(np.nan_to_num(mb), k)
    r_5m = float(np.corrcoef(sa[ok][::k], sb[ok][::k])[0, 1])
    return dict(repeat_r_px=r_px, repeat_r_5m=r_5m, repeat_mean_diff=float(np.nanmean(ma[ok] - mb[ok])),
                repeat_windows=[float((mid - t[0]) / 60), float((t_ign - mid) / 60)])


def pattern_checks(cfg: Config, L, unit):
    """Is the thermal moisture higher under denser canopy and where Sentinel-2 NDII is higher?"""
    from scipy.stats import spearmanr
    from .landscape import QUANT
    g = json.loads((cfg.interim_dir / "grid.json").read_text())
    res = g["res"]
    out = {}
    # LANDFIRE canopy cover: one value per 30 m block -> compare block means
    cc, m1 = L["cc"], L["m1"]
    blocks = pd.DataFrame(dict(cc=cc[unit], m1=m1[unit])).groupby("cc").m1.agg(["median", "size"])
    out["cc_blocks"] = {int(k): dict(m1=float(v["median"]), n_px=int(v["size"])) for k, v in blocks.iterrows()}
    rho, p = spearmanr(cc[unit], m1[unit])
    out["cc_spearman"], out["cc_p"] = float(rho), float(p)
    # Sentinel-2 NDII (last clear pre-burn scene) at 20 m: mean thermal moisture inside each S2 pixel on the plot
    z = np.load(raw_dir(cfg) / "s2_last_clear.npz", allow_pickle=True)
    nd, on, tr = z["ndii"], z["on"].astype(bool), z["transform"]
    from rasterio.warp import transform
    a, b, c, d, e, f = tr
    xs, ys, vals = [], [], []
    for rr, cc_ in zip(*np.nonzero(on)):
        X0, Y0 = a * cc_ + b * rr + c, d * cc_ + e * rr + f      # pixel corner (UTM, WGS84)
        xs.append(X0); ys.append(Y0); vals.append(nd[rr, cc_])
    pts = []
    for X0, Y0, v in zip(xs, ys, vals):
        X, Y = transform(str(z["crs"]), "EPSG:26916", [X0, X0 + a], [Y0, Y0 + e])
        c0 = int(np.floor((min(X) - g["transform"][2]) / res)); c1 = int(np.ceil((max(X) - g["transform"][2]) / res))
        r0 = int(np.floor((g["transform"][5] - max(Y)) / res)); r1 = int(np.ceil((g["transform"][5] - min(Y)) / res))
        sub_u = unit[max(r0, 0):max(r1, 0), max(c0, 0):max(c1, 0)]
        sub_m = m1[max(r0, 0):max(r1, 0), max(c0, 0):max(c1, 0)]
        if sub_u.size and sub_u.mean() > 0.6:
            pts.append((v, float(np.median(sub_m[sub_u]))))
    if len(pts) >= 5:
        v, mm = np.array(pts).T
        rho, p = spearmanr(v, mm)
        out.update(s2_spearman=float(rho), s2_p=float(p), s2_n=int(len(pts)), s2_date=str(z["date"]))
    return out


def forecast_sensitivity(cfg: Config, checks):
    """Test-split skill of the physics models with the thermal moisture map vs uniform moisture."""
    from . import evaluate as EV
    from . import forecast as FC
    from .landscape import snap
    m, F, W, ign, _ = EV.load(cfg)
    ctx0 = m["ctx"]
    n_pre, n_pass = cfg.first_snapshot, m["burnt"].shape[0]
    _, test = EV.split_passes(n_pre, n_pass, cfg.train_fire_passes)
    r0, r1, c0, c1 = ctx0["box"]
    burnt_c, fuel_c = m["burnt"][:, r0:r1, c0:c1], m["unit"][r0:r1, c0:c1]
    variants = {"thermal map": ctx0["L"]["m1"],
                "uniform, FFMC (weather)": np.full_like(ctx0["L"]["m1"], snap("m1", checks["m_ffmc"])),
                "uniform, air EMC": np.full_like(ctx0["L"]["m1"], snap("m1", checks["emc_air"]))}
    out = {}
    for name, m1 in variants.items():
        m1_t = ctx0["m1_t"] if name == "thermal map" or ctx0.get("m1_t") is None else np.broadcast_to(m1, ctx0["m1_t"].shape)
        ctx = dict(ctx0, L=dict(ctx0["L"], m1=m1), fields={}, m1_t=m1_t)
        mm = dict(m, ctx=ctx)
        for model in ("flammap", "physics", "fbp"):
            vals = []
            for i in test:
                src = EV.interval_sources(i, F, ign, W, ctx)
                pred, AT = EV.run_forecast(model, i, src, W, burnt_c, fuel_c, ctx, res=0.5, ign_r0=cfg.ign_r0_m)
                vals.append(FC.metrics(pred, burnt_c[i + 1], burnt_c[i], fuel_c, AT, W.loc[i, "dt"], 0.5))
            v = pd.DataFrame(vals)
            out.setdefault(name, {})[model] = dict(iou_new=float(v.iou_new.mean()), front_mean_m=float(v.front_mean_m.mean()))
    return out


def load(cfg: Config) -> dict | None:
    p = cfg.interim_dir / "moisture.json"
    return json.loads(p.read_text()) if p.exists() else None


def report(r):
    lv, ck = r["live"], r["checks"]
    print("[moisture] live fuel moisture (relative position of the burn-date index in its 2019-2023 range):")
    for k in ("s2_ndii", "s2_ndwi22", "landsat_ndii", "s1_vh"):
        v = lv.get(k)
        if v:
            print(f"   {k:12s} {v['date']}  value {v['value']:.4f}  p5-p95 {v['p5']:.4f}-{v['p95']:.4f}  r {v['r']:.2f}  (n={v['n']})")
    print(f"   -> herb {lv['herb']:.2f}, woody {lv['wood']:.2f} (r = {lv['r']:.2f})")
    print(f"[moisture] thermal 1-h: median {ck['thermal_median']:.3f} (p5-p95 {ck['thermal_p5']:.3f}-{ck['thermal_p95']:.3f}); "
          f"FFMC {ck['ffmc']:.1f} -> {ck['m_ffmc']:.3f}; air EMC {ck['emc_air']:.3f}; BUI {ck['bui']:.1f}")
    print(f"   repeatability r {ck['repeat_r_px']:.2f} (0.5 m), {ck['repeat_r_5m']:.2f} (5 m); canopy rho {ck['cc_spearman']:.2f}; "
          f"S2 NDII rho {ck.get('s2_spearman', float('nan')):.2f} (n={ck.get('s2_n')})")
    if "sensitivity" in ck:
        for k, v in ck["sensitivity"].items():
            print(f"   {k:24s}", {m: round(x["iou_new"], 3) for m, x in v.items()})


if __name__ == "__main__":
    cfg = Config()
    download(cfg)
    report(build(cfg))
