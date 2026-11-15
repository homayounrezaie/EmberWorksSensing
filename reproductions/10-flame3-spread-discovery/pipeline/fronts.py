"""Generic fire-front tools: sub-pixel fronts of a burnt mask and the
rate of spread (ROS) along each front point's outward normal.

* front points : sub-pixel 0.5-contour of the smoothed burnt mask, ~1 px apart,
                 with outward normal, curvature, local radiant flux and flags
* ROS          : distance along the outward normal from front(t) to front(t+1),
                 divided by the time difference
* flags        : the ray re-enters older burnt area before reaching front(t+1)
                 (merge: the advance is censored), leaves the fuel/plot area,
                 or exceeds `max_advance_m`
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure

from .config import Config


def _resample_contour(c, step=1.0):
    seg = np.hypot(*np.diff(c, axis=0).T)
    s = np.r_[0, np.cumsum(seg)]
    if s[-1] < 2 * step:
        return c[:1]
    q = np.arange(0, s[-1], step)
    return np.c_[np.interp(q, s, c[:, 0]), np.interp(q, s, c[:, 1])]


def curvature_field(b, sigma):
    """Signed curvature (1/px) of the level sets of the smoothed burnt mask;
    > 0 where the front bulges into unburnt fuel."""
    f = ndi.gaussian_filter(b.astype(float), sigma)
    gr, gc = np.gradient(f)
    nrm = np.hypot(gr, gc) + 1e-9
    nr, nc = -gr / nrm, -gc / nrm  # outward normal, pixel axes
    return np.gradient(nr, axis=0) + np.gradient(nc, axis=1)


def extract_front(b, E_i, T_i, U, cfg: Config, res=3.0, I_field=None, act_field=None):
    """Front points of burnt mask `b` (pixel coords row/col), with features.
    E_i: radiant exitance (W/m²) used for the FRFD feature and the activity
    test, unless I_field (W/m²) / act_field (bool) are given separately."""
    f = ndi.gaussian_filter(b.astype(float), cfg.contour_sigma_px)
    pts = [_resample_contour(c) for c in measure.find_contours(f, 0.5)]
    pts = [p for p in pts if len(p) > 1]
    if not pts:
        return pd.DataFrame()
    P = np.concatenate(pts)
    gr, gc = np.gradient(f)
    g_r = ndi.map_coordinates(gr, P.T, order=1)
    g_c = ndi.map_coordinates(gc, P.T, order=1)
    nrm = np.hypot(g_r, g_c) + 1e-12
    n_r, n_c = -g_r / nrm, -g_c / nrm                       # outward, pixel axes
    kap = ndi.map_coordinates(curvature_field(b, cfg.curvature_sigma_px), P.T, order=1) / res
    emax = ndi.maximum_filter(np.nan_to_num(E_i if I_field is None else I_field), size=5)
    ri, ci = np.clip(np.rint(P[:, 0]).astype(int), 0, b.shape[0] - 1), np.clip(np.rint(P[:, 1]).astype(int), 0, b.shape[1] - 1)
    act = emax > cfg.thr_active if act_field is None else ndi.maximum_filter(act_field.astype(np.uint8), size=5) > 0
    inside_dist = ndi.distance_transform_edt(U) * res
    df = pd.DataFrame(dict(row=P[:, 0], col=P[:, 1], n_r=n_r, n_c=n_c,
                           phi=np.degrees(np.arctan2(n_c, -n_r)) % 360,   # azimuth of outward normal
                           kappa=kap, frfd=emax[ri, ci] / 1000.0, t=T_i[ri, ci],
                           active=act[ri, ci],
                           edge=inside_dist[ri, ci] < cfg.edge_exclude_m))
    return df


def measure_advance(fp, b0, b1, T1, U, cfg: Config, res=3.0):
    """Ray-march along the outward normal of each front point."""
    f0 = ndi.gaussian_filter(b0.astype(float), cfg.contour_sigma_px)
    f1 = ndi.gaussian_filter(b1.astype(float), cfg.contour_sigma_px)
    fu = ndi.gaussian_filter(U.astype(float), 1.0)
    dmax = cfg.max_advance_m / res
    s = np.arange(0.0, dmax + 0.5, 0.5)
    R = fp.row.values[:, None] + s[None, :] * fp.n_r.values[:, None]
    C = fp.col.values[:, None] + s[None, :] * fp.n_c.values[:, None]
    v1 = ndi.map_coordinates(f1, [R.ravel(), C.ravel()], order=1, cval=0).reshape(R.shape)
    v0 = ndi.map_coordinates(f0, [R.ravel(), C.ravel()], order=1, cval=0).reshape(R.shape)
    vu = ndi.map_coordinates(fu, [R.ravel(), C.ravel()], order=1, cval=0).reshape(R.shape)
    out1 = v1 < 0.5
    has_exit = out1.any(1)
    k = np.where(has_exit, out1.argmax(1), len(s) - 1)
    # sub-sample crossing
    k0 = np.maximum(k - 1, 0)
    a, bb = v1[np.arange(len(k)), k0], v1[np.arange(len(k)), k]
    frac = np.where(np.abs(a - bb) > 1e-9, (a - 0.5) / (a - bb + 1e-12), 0.0)
    d = np.where(k > 0, s[k0] + frac * (s[k] - s[k0]), 0.0)
    # merge: after leaving old burnt area, ray re-enters old burnt area before exiting new
    left0 = np.maximum.accumulate(v0 < 0.45, axis=1)
    reenter = left0 & (v0 >= 0.5)
    idx = np.arange(len(s))[None, :]
    merge = (reenter & (idx < k[:, None])).any(1)
    off_unit = ((vu < 0.5) & (idx <= k[:, None])).any(1)
    rr = np.clip(np.rint(fp.row + d * fp.n_r).astype(int), 0, U.shape[0] - 1)
    cc = np.clip(np.rint(fp.col + d * fp.n_c).astype(int), 0, U.shape[1] - 1)
    t1 = T1[rr, cc]
    return dict(d_m=d * res, long=~has_exit, merge=merge, off_unit=off_unit, t1=t1,
                end_row=fp.row + d * fp.n_r, end_col=fp.col + d * fp.n_c)
