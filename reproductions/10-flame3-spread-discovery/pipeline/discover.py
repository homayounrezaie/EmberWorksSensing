"""Stage 4 — equation discovery: sparse regression (STLSQ) of the normal rate
of spread on a library of candidate terms.

    ROS = sum_k beta_k * f_k(theta, inputs at the front point)

theta : angle between the outward front normal and the wind direction
FRFD  : fire radiative flux density at the front (LWIR radiant exitance), kW/m²
kappa : front curvature, 1/m (> 0 where the front bulges into fuel)
M     : 1-h dead fuel moisture (fraction) from the pre-fire thermal image
S     : terrain slope along the front normal, tan(slope) * cos(normal - upslope)
U     : midflame wind, m/s (20-ft wind x LANDFIRE canopy wind adjustment factor)
R0    : Rothermel no-wind, no-slope rate for the LANDFIRE fuel model and M, m/s
P     : spread rate observed near the point in the previous interval, m/s
        (fire "memory", the time-series input of WSTS+-style spread models)

Everything is computed from per-block sufficient statistics (A'A, A'y, y'y,
n) so the browser can redo the fit (and the bootstrap).  Blocks = (interval,
15 m tile).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DISPLAY = {"1": "1", "cos": "cos θ", "sin": "sin θ", "cos2": "cos² θ", "U": "U", "Ucos": "U·cos θ",
           "I": "FRFD", "Icos": "FRFD·cos θ", "k": "κ", "kcos": "κ·cos θ", "m": "M", "s": "S", "R0": "R₀", "P": "P",
           "Rr": "R_Roth", "Rf": "R_FBP", "T": "T_s"}
DEFAULT_KEYS = ["1", "cos", "sin", "cos2", "I", "Icos", "k", "kcos", "T", "m", "s", "Ucos", "R0", "P", "Rr", "Rf"]
# per-point inputs (up = upslope azimuth, deg).  Rr / Rf: the normal spread rate of the per-cell Rothermel /
# FBP ellipse at the front normal -- every landscape and weather input passed through each model's own physics.
FEATURES = ["U", "I", "k", "T", "m", "tan", "up", "R0", "P", "Rr", "Rf"]   # T: fuel surface temperature before the fire, C
# Physics sign priors (sign-constrained STLSQ): a kept term must not contradict established fire
# physics -- upslope / downwind / drier / more energetic fuel / faster-moving fire cannot slow spread.
SIGN_PRIOR = {"s": 1, "m": -1, "Ucos": 1, "R0": 1, "P": 1, "Rr": 1, "Rf": 1, "T": 1}  # per-point inputs (up = upslope azimuth, deg)


def terms_for(keys):
    return [(k, DISPLAY[k]) for k in keys]


def signs_for(keys, use=True):
    return np.array([SIGN_PRIOR.get(k, 0) if use else 0 for k in keys], float)


def forced_for(keys, forced=()):
    return np.array([k in forced for k in keys], bool)


def pen_for(keys, unpenalized=()):
    return np.array([0.0 if k in unpenalized else 1.0 for k in keys])


def library(theta_rad, f: dict, keys=DEFAULT_KEYS, wind_to_deg=0.0):
    """Candidate terms.  theta_rad: normal angle to the wind (broadcasts against the per-point
    features in f); the normal azimuth is theta + wind_to (for the slope term)."""
    ct, st = np.cos(theta_rad), np.sin(theta_rad)
    one = np.ones(np.broadcast(ct, *[np.asarray(f[k]) for k in FEATURES]).shape)
    cols = {}
    for c in keys:
        if c == "1": v = one
        elif c == "cos": v = ct * one
        elif c == "sin": v = st * one
        elif c == "cos2": v = ct ** 2 * one
        elif c == "U": v = f["U"] * one
        elif c == "Ucos": v = f["U"] * ct
        elif c == "I": v = f["I"] * one
        elif c == "Icos": v = f["I"] * ct
        elif c == "k": v = f["k"] * one
        elif c == "kcos": v = f["k"] * ct
        elif c == "m": v = f["m"] * one
        elif c == "s": v = f["tan"] * np.cos(theta_rad + np.radians(wind_to_deg - np.asarray(f["up"])))
        elif c == "R0": v = f["R0"] * one
        elif c == "P": v = f["P"] * one
        elif c == "T": v = f["T"] * one
        elif c == "Rr": v = f["Rr"] * one
        elif c == "Rf": v = f["Rf"] * one
        else: raise KeyError(c)
        cols[c] = v * one
    return np.stack([cols[c] for c in keys], axis=-1)


def features(F: pd.DataFrame) -> dict:
    return dict(U=F.Umid.values, I=F.frfd.values, k=F.kappa.values, m=F.m1.values, tan=F.tan_s.values,
                up=F.up_az.values, R0=F.R0.values, P=F.Pm.values, Rr=F.Rr.values, Rf=F.Rf.values, T=F.Ts.values)


def design(F: pd.DataFrame, keys=DEFAULT_KEYS):
    # theta + wind_to = normal azimuth phi, so pass theta with wind_to = phi - theta per point
    wt = (F.phi.values - F.theta.values)
    return library(np.radians(F.theta.values), features(F), keys, wt), F.ros.values


def block_stats(F: pd.DataFrame, keys=DEFAULT_KEYS) -> pd.DataFrame:
    A, y = design(F, keys)
    rows = []
    for (p, t), idx in F.groupby(["pass_id", "tile"]).indices.items():
        a, yy = A[idx], y[idx]
        rows.append(dict(pass_id=int(p), tile=int(t), n=len(idx), AtA=a.T @ a, Aty=a.T @ yy, yty=float(yy @ yy)))
    return pd.DataFrame(rows)


def sum_stats(B: pd.DataFrame):
    return (np.sum(np.stack(B.AtA.values), 0), np.sum(np.stack(B.Aty.values), 0), float(B.yty.sum()), int(B.n.sum()))


def _std_transform(AtA, n):
    M = AtA.shape[0]
    mu = AtA[0] / n
    var = np.diag(AtA) / n - mu ** 2
    sd = np.sqrt(np.maximum(var, 1e-18))
    T = np.eye(M)
    for k in range(1, M):
        T[0, k] = -mu[k] / sd[k]
        T[k, k] = 1.0 / sd[k]
    return T, mu, sd


def stlsq(AtA, Aty, n, lam, ridge=1e-3, iters=20, signs=None, forced=None, pen=None):
    """Sequentially thresholded least squares on standardised terms.  signs: optional array of
    +1 / -1 / 0 per term; a term whose coefficient has the wrong sign is dropped like a small one.
    Returns beta (original units), xi (standardised), active mask."""
    T, mu, sd = _std_transform(AtA, n)
    sg = np.zeros(AtA.shape[0]) if signs is None else np.asarray(signs, float)
    fo = np.zeros(AtA.shape[0], bool) if forced is None else np.asarray(forced, bool)
    M = AtA.shape[0]
    G = T.T @ AtA @ T
    b = T.T @ Aty
    pw = np.ones(M) if pen is None else np.asarray(pen, float)   # per-term ridge weight (0 = not shrunk)
    reg = np.diag([0.0] + [ridge * n * pw[k] for k in range(1, M)])
    active = np.ones(M, bool)
    xi = np.zeros(M)
    for _ in range(iters):
        idx = np.where(active)[0]
        xi = np.zeros(M)
        xi[idx] = np.linalg.solve(G[np.ix_(idx, idx)] + reg[np.ix_(idx, idx)], b[idx])
        new = ((np.abs(xi) >= lam) & (sg * xi >= 0)) | fo
        new[0] = True  # the base rate is always kept
        if (new == active).all():
            break
        active = new
    idx = np.where(active)[0]
    xi = np.zeros(M)
    xi[idx] = np.linalg.solve(G[np.ix_(idx, idx)] + reg[np.ix_(idx, idx)], b[idx])
    beta = T @ xi
    return beta, xi, active


def r2(beta, AtA, Aty, yty, n):
    sse = yty - 2 * beta @ Aty + beta @ AtA @ beta
    sst = yty - Aty[0] ** 2 / n
    return float(1 - sse / sst), float(np.sqrt(max(sse, 0) / n))


def choose_lambda(Btrain: pd.DataFrame, lams, ridge=1e-3, signs=None, forced=None, pen=None):
    """Leave-one-interval-out CV inside the training set; 1-SE rule on the
    *paired* fold differences (interval-to-interval variation in the error
    level is shared by all lambdas and would otherwise swamp the SE)."""
    passes = sorted(Btrain.pass_id.unique())
    errs = np.zeros((len(lams), len(passes)))
    for j, p in enumerate(passes):
        tr, te = Btrain[Btrain.pass_id != p], Btrain[Btrain.pass_id == p]
        AtA, Aty, _, n = sum_stats(tr)
        AtAe, Atye, ytye, ne = sum_stats(te)
        for i, lam in enumerate(lams):
            beta, _, _ = stlsq(AtA, Aty, n, lam, ridge, signs=signs, forced=forced, pen=pen)
            errs[i, j] = (ytye - 2 * beta @ Atye + beta @ AtAe @ beta) / ne
    m = errs.mean(1)
    best = int(np.argmin(m))
    diff = errs - errs[best]
    se_diff = diff.std(1, ddof=1) / np.sqrt(len(passes))
    ok = np.where(diff.mean(1) <= se_diff)[0]
    return float(lams[ok.max()]), pd.DataFrame(dict(lam=lams, cv_mse=m, cv_se_paired=se_diff))


def bootstrap(Btrain: pd.DataFrame, lam, n_boot=200, ridge=1e-3, seed=0, keys=DEFAULT_KEYS, signs=None, forced=None, pen=None):
    rng = np.random.default_rng(seed)
    AtA = np.stack(Btrain.AtA.values); Aty = np.stack(Btrain.Aty.values); n = Btrain.n.values
    betas, act = [], []
    for _ in range(n_boot):
        k = rng.integers(0, len(Btrain), len(Btrain))
        beta, _, a = stlsq(AtA[k].sum(0), Aty[k].sum(0), int(n[k].sum()), lam, ridge, signs=signs, forced=forced, pen=pen)
        betas.append(beta); act.append(a)
    betas, act = np.array(betas), np.array(act)
    return pd.DataFrame(dict(term=[DISPLAY[k] for k in keys], inclusion=act.mean(0),
                             p05=np.percentile(betas, 5, 0), p50=np.percentile(betas, 50, 0),
                             p95=np.percentile(betas, 95, 0)))


def normal_speed(beta, theta_rad, f: dict, keys=DEFAULT_KEYS, wind_to_deg=0.0):
    """Model normal ROS (m/s), clipped at 0."""
    return np.maximum(library(theta_rad, f, keys, wind_to_deg) @ beta, 0.0)


def wavelet_summary(Rn_of_theta, n=720):
    """Head/flank/back rates, L/B and head direction (deg relative to the wind,
    + = clockwise) of the Wulff wavelet of a normal-speed function of theta."""
    th = np.linspace(-np.pi, np.pi, n, endpoint=False)
    R = np.maximum(Rn_of_theta(th), 0)
    # Wulff shape polar radius on the same grid
    D = th[:, None] - th[None, :]
    c = np.cos(D)
    inv = np.where(c > 0.02, 1.0 / np.where(c > 0.02, c, 1.0), np.inf)
    r = np.min(R[None, :] * inv, axis=1)
    i = int(np.argmax(r))
    head, ang = r[i], th[i]
    back = r[(i + n // 2) % n]
    lat = r * np.sin(th - ang)
    flank = 0.5 * (lat.max() - lat.min())
    lb = (head + back) / (2 * flank) if flank > 0 else np.inf
    return dict(head=float(head), back=float(back), flank=float(flank), lb=float(lb), head_dir=float(np.degrees(ang)))


def explain(beta, active, inclusion, ref, keys=DEFAULT_KEYS, wind_label="20-ft wind"):
    """One plain-language sentence per kept term (ref = typical U, FRFD)."""
    out = []
    iU = keys.index("Ucos") if "Ucos" in keys else None
    for k, (key, disp) in enumerate(terms_for(keys)):
        if not active[k]:
            continue
        b = beta[k]
        sgn = "faster" if b > 0 else "slower"
        if key == "1":
            s = f"Base rate {b:+.3f} m/s: the part of the spread rate that does not depend on the other terms."
        elif key == "cos":
            s = (f"Fronts facing straight downwind spread {abs(b):.3f} m/s {sgn} than flanks, and backing fronts "
                 f"{abs(b):.3f} m/s {'slower' if b > 0 else 'faster'}.")
        elif key == "sin":
            side = "clockwise" if b > 0 else "anticlockwise"
            s = (f"Fronts facing 90° {side} of the measured wind spread {abs(b):.3f} m/s faster than those facing the "
                 f"other way, i.e. the fire behaves as if the wind were turned {side}.")
        elif key == "cos2":
            s = (f"Head and backing fronts both spread {abs(b):.3f} m/s {sgn} than flanks "
                 f"({'elongated along' if b > 0 else 'widened across'} the wind axis).")
        elif key == "U":
            s = f"Every front spreads {abs(b):.3f} m/s {sgn} for each extra 1 m/s of {wind_label}, regardless of direction."
        elif key == "Ucos":
            s = (f"Head-fire spread is {abs(b):.3f} m/s {sgn} with each 1 m/s of wind along the front normal "
                 f"(and backing spread {'slower' if b > 0 else 'faster'} by the same).")
        elif key == "I":
            s = f"Each extra 10 kW/m² of radiant flux at the front makes it {10 * abs(b):.3f} m/s {sgn}: intense fronts move {'faster' if b > 0 else 'slower'}."
        elif key == "Icos":
            s = (f"The intensity effect is {'stronger' if b > 0 else 'weaker'} on head fronts: "
                 f"{10 * abs(b):.3f} m/s per 10 kW/m² × cos θ.")
        elif key == "k":
            s = (f"A front bulging into fuel with a 20 m radius of curvature spreads {0.05 * abs(b):.3f} m/s "
                 f"{sgn} than a straight one (κ = 0.05 1/m).")
        elif key == "kcos":
            s = f"Curvature matters {'more' if b > 0 else 'less'} on head fronts: {0.05 * abs(b):.3f} m/s per κ = 0.05 1/m × cos θ."
        elif key == "m":
            s = (f"Each extra 0.01 of fine-fuel moisture (from the pre-fire thermal image) makes fronts "
                 f"{0.01 * abs(b):.4f} m/s {sgn}.")
        elif key == "s":
            s = (f"Fronts running up a 10 % slope spread {0.1 * abs(b):.4f} m/s {sgn} than on flat ground "
                 f"(and down it {'slower' if b > 0 else 'faster'} by the same).")
        elif key == "R0":
            s = (f"The rate scales with Rothermel's no-wind, no-slope rate R₀ for the local LANDFIRE fuel model "
                 f"and moisture: {abs(b):.2f} × R₀ {'added' if b > 0 else 'subtracted'}.")
        elif key == "T":
            s = f"Each extra 1 C of fuel surface temperature before the fire makes fronts {abs(b):.4f} m/s {sgn}."
        elif key == "Rr":
            s = f"{b:+.2f} x Rothermel's own rate at the point (fuel model, moisture, canopy-reduced wind, slope)."
        elif key == "Rf":
            s = f"{b:+.2f} x the FBP rate at the point (moisture via FFMC, wind, slope, Buildup Index)."
        elif key == "P":
            s = (f"Fire memory: each 0.01 m/s of spread observed at the same place in the previous 30 s adds "
                 f"{0.01 * b:+.4f} m/s now.")
        if key == "cos" and iU is not None and active[iU]:  # cos θ and U·cos θ act together
            net = b + beta[iU] * ref["U"]
            s += (f" Together with the U·cos θ term, the net head-vs-flank difference at the typical wind "
                  f"({ref['U']:.1f} m/s) is {net:+.3f} m/s.")
        noise = inclusion is not None and inclusion[k] < 0.6 and key != "1"
        if noise:
            s += (f" Caution: this term is kept in only {inclusion[k]:.0%} of bootstrap resamples, so it may be noise. "
                  f"Try raising the simplicity threshold.")
        out.append(dict(term=disp, sentence=s, noise=bool(noise)))
    return out
