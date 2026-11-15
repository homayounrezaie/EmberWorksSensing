"""Shared forecast engine + metrics (mirrored 1:1 in web/index.html JS).

Every model is a *normal spread rate* R(n) — the speed of a front whose
outward normal points in absolute azimuth n.  By Huygens' principle the
matching wavelet is the Wulff shape of R (the intersection of half-planes
x·n <= R(n)); for the physics model that is exactly the Rothermel/Anderson
ellipse.  The arrival time of fire at pixel x is

    AT(x) = min over sources s of  tau0_s + max(|x - s| - r0_s, 0) / r_s(dir(x - s))

where r_s(.) is the wavelet's polar radius (m/s).  Sources are the active
front points of the current overpass (tau0 = 0) and the ignition-track points
driven during the interval (tau0 = time after the overpass, r0 = 2 m line
width).  The forecast burnt area at t + dt is  burnt(t) ∪ {AT <= dt} ∩ fuel.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

NDIR = 72
DIRS = np.arange(NDIR) * (360.0 / NDIR)          # azimuths, deg
_D = np.radians(DIRS[:, None] - DIRS[None, :])   # [travel k, normal j]
_COS = np.cos(_D)
_VALID = _COS > 0.05
_INV = 1.0 / np.where(_VALID, _COS, 1.0)
IGN_R0 = 2.0  # m


def wulff_radius(Rn: np.ndarray) -> np.ndarray:
    """Rn: (N, NDIR) normal speeds (>= 0) at normal azimuths DIRS.
    Returns (N, NDIR) polar radius of the wavelet at travel azimuths DIRS."""
    Rn = np.maximum(np.atleast_2d(Rn), 0.0)
    out = np.empty_like(Rn)
    for s in range(0, len(Rn), 256):
        blk = Rn[s:s + 256]
        out[s:s + 256] = np.min(np.where(_VALID[None], blk[:, None, :] * _INV[None], np.inf), axis=2)
    return out


def arrival_time(rows, cols, tau0, r, r0, shape, horizon, res=3.0):
    """Minimum arrival time (s) over all sources, up to `horizon` (else inf)."""
    AT = np.full(shape, np.inf)
    H, W = shape
    for s in range(len(rows)):
        tmax = horizon - tau0[s]
        if tmax < 0:
            continue
        reach = (r[s].max() * tmax + r0[s]) / res
        if reach < 0.5:
            reach = 0.5
        r_lo, r_hi = max(int(np.floor(rows[s] - reach)), 0), min(int(np.ceil(rows[s] + reach)) + 1, H)
        c_lo, c_hi = max(int(np.floor(cols[s] - reach)), 0), min(int(np.ceil(cols[s] + reach)) + 1, W)
        if r_lo >= r_hi or c_lo >= c_hi:
            continue
        dr = np.arange(r_lo, r_hi)[:, None] - rows[s]
        dc = np.arange(c_lo, c_hi)[None, :] - cols[s]
        rho = np.hypot(dr, dc) * res
        az = np.degrees(np.arctan2(dc, -dr)) % 360.0
        fi = az / (360.0 / NDIR)
        i0 = np.floor(fi).astype(int) % NDIR
        f = fi - np.floor(fi)
        rr = r[s][i0] * (1 - f) + r[s][(i0 + 1) % NDIR] * f
        d = np.maximum(rho - r0[s], 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(d <= 0, 0.0, np.where(rr > 0, d / rr, np.inf)) + tau0[s]
        sub = AT[r_lo:r_hi, c_lo:c_hi]
        np.minimum(sub, t, out=sub)
    AT[AT > horizon] = np.inf
    return AT


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def _front(mask, fuel):
    """Boundary pixels of `mask` that touch unburnt fuel."""
    return mask & ndi.binary_dilation(~mask & fuel)


def metrics(pred, actual, old, fuel, AT, dt, res=3.0):
    """IoU, new-area IoU, front distance (mean, p90, m), arrival-time error (s)."""
    pred, actual, old = pred & fuel, actual & fuel, old & fuel
    inter, union = (pred & actual).sum(), (pred | actual).sum()
    pn, an = pred & ~old, actual & ~old
    iou_new = (pn & an).sum() / max((pn | an).sum(), 1)
    Pf, Af = _front(pred, fuel), _front(actual, fuel)
    region = ndi.binary_dilation(pn | an, iterations=3)
    dP = ndi.distance_transform_edt(~Pf) * res if Pf.any() else np.full(pred.shape, np.inf)
    dA = ndi.distance_transform_edt(~Af) * res if Af.any() else np.full(pred.shape, np.inf)
    d = np.r_[dP[Af & region], dA[Pf & region]]
    d = d[np.isfinite(d)]
    # arrival-time error: late where it burnt but the forecast had not arrived by dt
    # (AT capped at 2 dt), early where the forecast arrived but it had not burnt
    at = np.minimum(np.where(np.isfinite(AT), AT, 2 * dt), 2 * dt)
    late = np.maximum(at[an] - dt, 0.0)
    early = np.maximum(dt - at[pn & ~an], 0.0)
    ate = np.r_[late, early]
    return dict(iou=inter / max(union, 1), iou_new=iou_new,
                front_mean_m=float(d.mean()) if d.size else np.nan,
                front_p90_m=float(np.percentile(d, 90)) if d.size else np.nan,
                arrival_err_s=float(ate.mean()) if ate.size else 0.0,
                new_actual_ha=an.sum() * res * res / 1e4, new_pred_ha=pn.sum() * res * res / 1e4)


def forecast_mask(old, fuel, AT, dt):
    return (old | (AT <= dt)) & fuel


# ----------------------------------------------------------------------------
# FlamMap-style minimum travel time (Finney 2002)
# ----------------------------------------------------------------------------
MTT_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1),
               (-1, -2), (-1, 2), (1, -2), (1, 2), (-2, -1), (-2, 1), (2, -1), (2, 1)]  # 16 directions


def ellipse_ray_speed(psi_rad, el):
    """Spread rate along a ray from the ignition point (rear focus of the
    elliptical fire) at angle psi to the head direction:
    R(psi) = R_head (1 - e) / (1 - e cos psi), e = c / a."""
    a, c = el["a"], el["c"]
    if a <= 0:
        return np.zeros_like(np.asarray(psi_rad, float))
    e = c / a
    return (a + c) * (1 - e) / (1 - e * np.cos(psi_rad))


def mtt_inverse_speeds(field, res=0.5):
    """(H*W, 16) inverse ray speeds (s/m) of every cell's elliptical fire along the 16 lattice
    directions; the cell's fire is the ellipse `field` (a, b, c, head_az arrays) with the
    ignition point at its rear focus."""
    a, c, az = field["a"].ravel(), field["c"].ravel(), field["head_az"].ravel()
    e = np.where(a > 0, c / np.where(a > 0, a, 1.0), 0.0)
    inv = np.full((a.size, len(MTT_OFFSETS)), np.inf)
    for d, (dr, dc) in enumerate(MTT_OFFSETS):
        psi = np.radians(np.degrees(np.arctan2(dc, -dr)) - az)
        r = (a + c) * (1 - e) / (1 - e * np.cos(psi))
        inv[:, d] = np.where((a > 0) & (r > 0), 1.0 / np.where(r > 0, r, 1.0), np.inf)
    return inv


def mtt_arrival(rows, cols, tau0, inv, passable, shape, horizon, res=0.5):
    """Arrival time (s) by Dijkstra over a 16-neighbour lattice (FlamMap minimum travel time,
    Finney 2002).  Each cell has its own elliptical fire (spatially varying fuel, moisture,
    wind and slope); crossing an edge takes half its length at the speed of the cell it leaves
    plus half at the speed of the cell it enters.  Fire may only enter unburnt fuel
    (`passable`).  Sources = front points (tau0 = 0) and ignition points (tau0 = ignition time
    after the snapshot)."""
    import heapq
    H, W = shape
    AT = np.full(H * W, np.inf)
    half = [0.5 * np.hypot(dr, dc) * res for dr, dc in MTT_OFFSETS]
    heap = []
    for r_, c_, t_ in zip(rows, cols, tau0):
        rr, cc = int(np.floor(r_ + 0.5)), int(np.floor(c_ + 0.5))  # same rounding as the JS engine
        if 0 <= rr < H and 0 <= cc < W and t_ <= horizon:
            q = rr * W + cc
            if t_ < AT[q]:
                AT[q] = t_
                heapq.heappush(heap, (t_, rr, cc))
    pas = passable.ravel()
    while heap:
        t, r_, c_ = heapq.heappop(heap)
        p = r_ * W + c_
        if t > AT[p] or t > horizon:
            continue
        ip = inv[p]
        for d, (dr, dc) in enumerate(MTT_OFFSETS):
            nr, nc = r_ + dr, c_ + dc
            if 0 <= nr < H and 0 <= nc < W:
                q = nr * W + nc
                if not pas[q]:
                    continue
                nt = t + half[d] * (ip[d] + inv[q, d])
                if nt < AT[q] and nt <= horizon:
                    AT[q] = nt
                    heapq.heappush(heap, (nt, nr, nc))
    AT = AT.reshape(shape)
    AT[AT > horizon] = np.inf
    return AT
