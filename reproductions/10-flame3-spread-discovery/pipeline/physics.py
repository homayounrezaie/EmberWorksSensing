"""Physics / operational spread models for the Hanna Hammock case.

* Rothermel (1972) surface rate of spread, Albini (1976) / Andrews (2018)
  formulation, Anderson (1982) fuel model 2 (timber, grass understorey);
  elliptical fire shape from Anderson (1983).  Propagated two ways:
  - Huygens elliptical wavelets (the FARSITE method, Richards 1990)
  - minimum travel time on a lattice (the FlamMap MTT method, Finney 2002)
* Canadian Forest Fire Behavior Prediction (FBP) System, Forestry Canada
  (1992) ST-X-3, fuel type C-7 (open pine with a grass understorey):
  ISI from FFMC and 10 m wind, rate of spread, back rate and length-to-breadth.

Inputs are never fitted to the fire: dead-fuel moisture from the on-site
temperature / humidity (Simard 1968 EMC), on-site wind speed (midflame) for
Rothermel, open 10 m airport wind for FBP (its standard input), wind
direction from the airport.  The optional "calibrated" Rothermel variant fits
ONLY a wind multiplier and direction offset on training samples.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TON_AC = 2000.0 / 43560.0  # lb/ft² per ton/acre
FTMIN = 0.00508            # m/s per ft/min
MPH = 0.44704              # m/s per mi/h


@dataclass(frozen=True)
class FuelModel:
    name: str
    w1: float     # 1-h dead load, t/ac
    w10: float
    w100: float
    w_herb: float
    w_wood: float
    sav1: float   # 1/ft
    sav_herb: float
    sav_wood: float
    depth: float  # ft
    mx_dead: float  # fraction
    heat: float = 8000.0
    dynamic: bool = True


GR2 = FuelModel("GR2", 0.10, 0.0, 0.0, 1.00, 0.0, 2000.0, 1800.0, 9999.0, 1.0, 0.15)
GR1 = FuelModel("GR1", 0.10, 0.0, 0.0, 0.30, 0.0, 2200.0, 2000.0, 9999.0, 0.4, 0.15)
GR3 = FuelModel("GR3", 0.10, 0.40, 0.0, 1.50, 0.0, 1500.0, 1300.0, 9999.0, 2.0, 0.30)
FM1 = FuelModel("FM1", 0.74, 0.0, 0.0, 0.0, 0.0, 3500.0, 1500.0, 1500.0, 1.0, 0.12, dynamic=False)
# Anderson (1982) fuel model 2, "timber (grass and understory)": open pine stands with a grass
# understory -- the standard choice for frequently burned southeastern upland pine.
FM2 = FuelModel("FM2", 2.0, 1.0, 0.5, 0.5, 0.0, 3000.0, 1500.0, 1500.0, 1.0, 0.15, dynamic=False)
FUEL_MODELS = {m.name: m for m in (GR1, GR2, GR3, FM1, FM2)}
# Scott & Burgan (2005) models mapped by LANDFIRE 2022 FBFM40 over the plot (Table 7 of RMRS-GTR-153):
#   TU1 (161) low-load dry-climate timber-grass-shrub (dynamic), TU2 (162) moderate-load humid-climate
#   timber-shrub, TL3 (183) moderate-load conifer litter.
TU1 = FuelModel("TU1", 0.20, 0.90, 1.50, 0.20, 0.90, 2000.0, 1800.0, 1600.0, 0.6, 0.20, dynamic=True)
TU2 = FuelModel("TU2", 0.95, 1.80, 1.25, 0.00, 0.20, 2000.0, 9999.0, 1600.0, 1.0, 0.30, dynamic=False)
TL3 = FuelModel("TL3", 0.50, 2.20, 2.80, 0.00, 0.00, 2000.0, 9999.0, 9999.0, 0.3, 0.20, dynamic=False)
SB40 = {m.name: m for m in (TU1, TU2, TL3)}
# live fuel moisture (not observable in LWIR): Scott & Burgan "L2" scenario (herb 60 %, woody 90 %),
# late-February dormant-season understorey before green-up
M_HERB, M_WOOD = 0.60, 0.90
M10_ADD, M100_ADD = 0.01, 0.02   # 10-h / 100-h = 1-h + 1 % / + 2 % (Rothermel 1983)

def emc_simard(T_F, RH):
    """Equilibrium moisture content (fraction) of fine dead fuel, Simard (1968)."""
    if RH < 10:
        e = 0.03229 + 0.281073 * RH - 0.000578 * RH * T_F
    elif RH < 50:
        e = 2.22749 + 0.160107 * RH - 0.01478 * T_F
    else:
        e = 21.0606 + 0.005565 * RH ** 2 - 0.00035 * RH * T_F - 0.483199 * RH
    return e / 100.0




def _size_class(sav):
    edges = [1200, 192, 96, 48, 16]
    for k, e in enumerate(edges):
        if sav >= e:
            return k
    return len(edges)


def rothermel(fm: FuelModel, m1, m10, m100, m_herb, m_wood, u_mid_ms, slope=0.0):
    """Head-fire rate of spread (m/s) and reaction intensity (kW/m²) with wind and slope aligned.
    Moistures are fractions; u_mid_ms is the midflame wind speed (m/s); slope in degrees."""
    k = rothermel_core(fm, m1, m10, m100, m_herb, m_wood)
    u = min(u_mid_ms / FTMIN, 0.9 * k["IR"])  # Rothermel wind limit (never active at these winds)
    phi_w = k["C"] * u ** k["B"] * k["ratio"] ** -k["E"] if u > 0 else 0.0
    phi_s = 5.275 * k["beta"] ** -0.3 * np.tan(np.radians(slope)) ** 2
    return k["R0"] * (1 + phi_w + phi_s), k["IR"] * 0.1135  # BTU/ft²/min -> kW/m²


def rothermel_core(fm: FuelModel, m1, m10, m100, m_herb, m_wood):
    """No-wind, no-slope Rothermel rate R0 (m/s), reaction intensity IR (BTU/ft²/min) and the
    fuel-bed constants of the wind and slope factors: phi_w = C u^B (beta/beta_op)^-E (u in ft/min),
    phi_s = 5.275 beta^-0.3 tan² slope."""
    w_herb_live, w_herb_dead = fm.w_herb, 0.0
    if fm.dynamic and fm.w_herb > 0:
        cured = float(np.clip((1.20 - m_herb) / 0.90, 0.0, 1.0))
        w_herb_dead, w_herb_live = cured * fm.w_herb, (1 - cured) * fm.w_herb
    dead = [(fm.w1, fm.sav1, m1), (fm.w10, 109.0, m10), (fm.w100, 30.0, m100), (w_herb_dead, fm.sav_herb, m1)]
    live = [(w_herb_live, fm.sav_herb, m_herb), (fm.w_wood, fm.sav_wood, m_wood)]
    dead = [(w * TON_AC, s, m) for w, s, m in dead if w > 0]
    live = [(w * TON_AC, s, m) for w, s, m in live if w > 0]
    rho_p, St, Se = 32.0, 0.0555, 0.010

    def cat(parts):
        if not parts:
            return None
        a = np.array([s * w / rho_p for w, s, m in parts])
        f = a / a.sum()
        sav = np.array([s for _, s, _ in parts]); w = np.array([w for w, _, _ in parts]); mo = np.array([m for *_, m in parts])
        cls = np.array([_size_class(s) for s in sav])
        g = np.array([f[cls == c].sum() for c in cls])
        return dict(A=a.sum(), f=f, sav=sav, w=w, m=mo, sigma=(f * sav).sum(), wn=(g * w * (1 - St)).sum(), M=(f * mo).sum())

    D, L = cat(dead), cat(live)
    cats = [c for c in (D, L) if c]
    AT = sum(c["A"] for c in cats)
    for c in cats:
        c["fc"] = c["A"] / AT
    sigma = sum(c["fc"] * c["sigma"] for c in cats)
    wtot = sum(c["w"].sum() for c in cats)
    beta = wtot / (rho_p * fm.depth)
    beta_op = 3.348 * sigma ** -0.8189
    ratio = beta / beta_op
    A = 133.0 * sigma ** -0.7913
    gmax = sigma ** 1.5 / (495.0 + 0.0594 * sigma ** 1.5)
    gamma = gmax * ratio ** A * np.exp(A * (1 - ratio))
    eta_s = min(1.0, 0.174 * Se ** -0.19)

    def eta_m(M, Mx):
        r = min(M / Mx, 1.0)
        return 1 - 2.59 * r + 5.11 * r ** 2 - 3.52 * r ** 3

    IR = gamma * fm.heat * D["wn"] * eta_m(D["M"], fm.mx_dead) * eta_s
    if L:
        num = (D["w"] * np.exp(-138 / D["sav"])).sum()
        den = (L["w"] * np.exp(-500 / L["sav"])).sum()
        mf_dead = (D["w"] * np.exp(-138 / D["sav"]) * D["m"]).sum() / num
        mx_live = max(2.9 * num / den * (1 - mf_dead / fm.mx_dead) - 0.226, fm.mx_dead)
        IR += gamma * fm.heat * L["wn"] * eta_m(L["M"], mx_live) * eta_s
    xi = np.exp((0.792 + 0.681 * sigma ** 0.5) * (beta + 0.1)) / (192 + 0.2595 * sigma)
    rho_b = wtot / fm.depth
    sink = rho_b * sum(c["fc"] * (c["f"] * np.exp(-138 / c["sav"]) * (250 + 1116 * c["m"])).sum() for c in cats)
    C = 7.47 * np.exp(-0.133 * sigma ** 0.55)
    B = 0.02526 * sigma ** 0.54
    E = 0.715 * np.exp(-3.59e-4 * sigma)
    return dict(R0=float(IR * xi / sink * FTMIN), IR=float(IR), C=float(C), B=float(B), E=float(E),
                ratio=float(ratio), beta=float(beta))


def waf_unsheltered(depth_ft):
    """Albini & Baughman (1979) wind adjustment factor, unsheltered fuel."""
    return 1.83 / np.log((20 + 0.36 * depth_ft) / (0.13 * depth_ft))


def anderson_lb(u_mid_ms):
    """Length-to-breadth ratio vs midflame wind (Anderson 1983, mi/h)."""
    u = u_mid_ms / MPH
    return 0.936 * np.exp(0.2566 * u) + 0.461 * np.exp(-0.1548 * u) - 0.397


def anderson_lb_inverse(lb):
    """Midflame wind (m/s) that gives the length-to-breadth ratio `lb`."""
    if lb <= 1.0:
        return 0.0
    us = np.linspace(0, 30, 30001) * MPH
    return float(np.interp(lb, anderson_lb(us), us))


def lb_from_head_back(head, back, lb_max=8.0):
    """Anderson-ellipse L/B with the same head:back ratio (e = (HB-1)/(HB+1))."""
    if back <= 0:
        return lb_max
    hb = head / back
    if hb <= 1:
        return 1.0
    e = (hb - 1) / (hb + 1)
    return float(min(1 / np.sqrt(1 - e ** 2), lb_max))


def ellipse(R_head, lb):
    """Huygens wavelet per unit time: semi-axes a (along wind), b, centre
    offset c downwind; also back and flank rates (m/s)."""
    lb = max(lb, 1.0)
    e = np.sqrt(1 - 1 / lb ** 2)
    R_back = R_head * (1 - e) / (1 + e)
    a = 0.5 * (R_head + R_back)
    b = a / lb
    c = 0.5 * (R_head - R_back)
    return dict(R_head=R_head, R_back=R_back, R_flank=b, a=a, b=b, c=c, lb=lb)


def ellipse_normal_speed(theta_rad, el):
    """Normal spread rate of a front whose outward normal is at angle theta
    to the wind (support function of the elliptical wavelet)."""
    ct, st = np.cos(theta_rad), np.sin(theta_rad)
    return el["c"] * ct + np.sqrt((el["a"] * ct) ** 2 + (el["b"] * st) ** 2)


@dataclass
class PhysicsParams:
    fuel: FuelModel = FM2
    m1: float = 0.08
    m10: float = 0.09
    m100: float = 0.10
    m_herb: float = 1.0
    m_wood: float = 1.0
    wind_mult: float = 1.0   # calibrated mode only
    dir_offset: float = 0.0  # deg, calibrated mode only (+ = clockwise)
    waf_override: float | None = None  # 1.0 when the input wind is already a midflame wind

    @property
    def waf(self):
        return self.waf_override if self.waf_override is not None else waf_unsheltered(self.fuel.depth)


def interval_ellipse(U20, p: PhysicsParams):
    u_mid = p.wind_mult * p.waf * U20
    R, IR = rothermel(p.fuel, p.m1, p.m10, p.m100, p.m_herb, p.m_wood, u_mid)
    el = ellipse(R, anderson_lb(u_mid))
    el.update(u_mid=u_mid, IR=IR)
    return el


def head_table(p: "PhysicsParams", U_max=15.0, step=0.05):
    """Rothermel head ROS (m/s) vs 20-ft wind (m/s) for parameters p."""
    U = np.arange(0.0, U_max + step / 2, step)
    return U, np.array([interval_ellipse(u, p)["R_head"] for u in U])


def effective_wind(R_head, p: "PhysicsParams"):
    """Rothermel 'effective wind speed': the 20-ft wind at which the physics
    model's head-fire rate equals R_head (clipped to the table range)."""
    U, R = head_table(p)
    return float(np.interp(R_head, R, U))


def calibrate(theta_deg, U20, ros, p0: PhysicsParams, mults=None, offsets=None):
    """Grid-search an effective wind multiplier and direction offset that
    minimise squared ROS error on TRAINING samples only."""
    mults = np.exp(np.linspace(np.log(0.25), np.log(8), 61)) if mults is None else mults
    offsets = np.arange(-60, 61, 2.0) if offsets is None else offsets
    U_vals = np.unique(U20)
    best = (np.inf, 1.0, 0.0)
    for m in mults:
        p = PhysicsParams(**{**p0.__dict__, "wind_mult": m, "dir_offset": 0.0})
        els = {u: interval_ellipse(u, p) for u in U_vals}
        for d in offsets:
            th = np.radians(theta_deg - d)
            pred = np.empty_like(ros)
            for u, el in els.items():
                k = U20 == u
                pred[k] = ellipse_normal_speed(th[k], el)
            sse = float(((pred - ros) ** 2).sum())
            if sse < best[0]:
                best = (sse, m, d)
    return PhysicsParams(**{**p0.__dict__, "wind_mult": best[1], "dir_offset": best[2]}), best[0]


def site_physics(cfg=None) -> PhysicsParams:
    """Hanna Hammock Rothermel inputs (never fitted to fire data): on-site 15:00 obs 84 F / 52 % -> Simard
    EMC for 1-h dead fuel (+1 % / +2 % for 10-h / 100-h); live herbaceous moisture assumed 100 % (late-February
    partial green-up); the on-site in-stand wind is used directly as the midflame wind (WAF = 1)."""
    m1 = emc_simard(84.0, 52.0)
    return PhysicsParams(fuel=FM2, m1=m1, m10=m1 + 0.01, m100=m1 + 0.02, m_herb=1.0, m_wood=1.0, waf_override=1.0)


def ellipse_hbl(head, back, lb):
    """Elliptical wavelet from independent head and back rates and L/B (FBP style)."""
    lb = max(lb, 1.0)
    a = 0.5 * (head + back)
    b = a / lb
    return dict(R_head=head, R_back=back, R_flank=b, a=a, b=b, c=0.5 * (head - back), lb=lb)


# ----------------------------------------------------------------------------
# Canadian FBP System (Forestry Canada Fire Danger Group 1992, ST-X-3)
# ----------------------------------------------------------------------------
FBP_FUELS = {"C-7": dict(a=45.0, b=0.0305, c=2.0, q=0.85, bui0=106.0)}


def ffmc_from_moisture(m_pct):
    """Fine Fuel Moisture Code with fine-fuel moisture m (%) (inverse of ST-X-3 eq. 46)."""
    return 59.5 * (250.0 - m_pct) / (147.2 + m_pct)


def _isi(m_pct, ws_kmh):
    f_f = 91.9 * np.exp(-0.1386 * m_pct) * (1 + m_pct ** 5.31 / 4.93e7)
    return 0.208 * np.exp(0.05039 * ws_kmh) * f_f


def fbp(m_pct, ws10_kmh, fuel="C-7", bui=None):
    """Surface head / back rate of spread (m/s), L/B and ISI.  BUI effect = 1
    when BUI is unknown (BUI = BUI0)."""
    f = FBP_FUELS[fuel]
    be = 1.0 if bui is None else float(np.exp(50 * np.log(f["q"]) * (1 / bui - 1 / f["bui0"])))
    isi = _isi(m_pct, ws10_kmh)
    ros = f["a"] * (1 - np.exp(-f["b"] * isi)) ** f["c"] * be                     # m/min
    bisi = _isi(m_pct, -ws10_kmh)                                                  # f(BW) = exp(-0.05039 WSV)
    bros = f["a"] * (1 - np.exp(-f["b"] * bisi)) ** f["c"] * be
    lb = 1.0 + 8.729 * (1 - np.exp(-0.030 * ws10_kmh)) ** 2.155                   # ST-X-3 eq. 79
    return dict(head=ros / 60.0, back=bros / 60.0, lb=float(lb), isi=float(isi), ffmc=float(ffmc_from_moisture(m_pct)))


def fbp_ellipse(m_pct, ws10_kmh, fuel="C-7"):
    r = fbp(m_pct, ws10_kmh, fuel)
    el = ellipse_hbl(r["head"], r["back"], r["lb"])
    el.update(isi=r["isi"], ffmc=r["ffmc"])
    return el


# ----------------------------------------------------------------------------
# spatially varying models: one elliptical wavelet per grid cell
# ----------------------------------------------------------------------------
def rothermel_tables(fuel_names, quant_m1, m_herb=M_HERB, m_wood=M_WOOD):
    """R0 (m/s) and IR (BTU/ft²/min) for every fuel model x every quantised 1-h moisture
    (m1 = lo + q*step, q = 0..255; 10-h / 100-h = m1 + 1 % / + 2 %) plus the per-model wind and
    slope constants.  Shipped to the web page, so both sides use identical numbers."""
    lo, st = quant_m1
    m1s = lo + np.arange(256) * st
    R0 = np.zeros((len(fuel_names), 256)); IR = np.zeros_like(R0)
    const = {k: np.zeros(len(fuel_names)) for k in ("C", "B", "E", "ratio", "beta")}
    for f, name in enumerate(fuel_names):
        fm = SB40[name]
        for q, m1 in enumerate(m1s):
            k = rothermel_core(fm, m1, m1 + M10_ADD, m1 + M100_ADD, m_herb, m_wood)
            R0[f, q], IR[f, q] = k["R0"], k["IR"]
        for c in const:
            const[c][f] = k[c]  # moisture-independent
    return dict(R0=R0, IR=IR, **const, fuels=list(fuel_names))


def _ellipse_arrays(R_head, lb):
    lb = np.maximum(lb, 1.0)
    e = np.sqrt(1 - 1 / lb ** 2)
    R_back = R_head * (1 - e) / (1 + e)
    a = 0.5 * (R_head + R_back)
    return dict(a=a, b=a / lb, c=0.5 * (R_head - R_back), R_head=R_head, R_back=R_back, lb=lb)


def rothermel_field(L, tab, U20, wind_to_deg, quant_m1):
    """Per-cell Rothermel ellipse: midflame wind = WAF * U20 (canopy / fuel-bed sheltering),
    wind and slope factors added as vectors (Rothermel 1972; FARSITE / FlamMap), effective
    wind from the resultant -> Anderson (1983) L/B."""
    f = L["fuel"].astype(int)
    q = np.clip(np.round((L["m1"] - quant_m1[0]) / quant_m1[1]), 0, 255).astype(int)
    R0, IR = tab["R0"][f, q], tab["IR"][f, q]
    C, B, E, ratio, beta = (tab[k][f] for k in ("C", "B", "E", "ratio", "beta"))
    u = np.minimum(L["waf"] * U20 / FTMIN, 0.9 * IR)
    phi_w = np.where(u > 0, C * np.power(np.maximum(u, 1e-12), B) * ratio ** -E, 0.0)
    phi_s = 5.275 * beta ** -0.3 * L["tan_slope"] ** 2
    wt, up = np.radians(wind_to_deg), np.radians(L["up_az"])
    vx = phi_w * np.sin(wt) + phi_s * np.sin(up)
    vy = phi_w * np.cos(wt) + phi_s * np.cos(up)
    phi = np.hypot(vx, vy)
    head_az = np.where(phi > 0, np.degrees(np.arctan2(vx, vy)) % 360, wind_to_deg % 360)
    u_eff = np.power(phi / (C * ratio ** -E), 1 / B) * FTMIN   # effective midflame wind, m/s
    out = _ellipse_arrays(R0 * (1 + phi), anderson_lb(u_eff))
    out.update(head_az=head_az, R0=R0, u_mid=L["waf"] * U20, u_eff=u_eff)
    return out


FBP_SLOPE_MAX_PCT = 70.0


def fbp_field(L, U10, wind_to_deg, fuel="C-7", bui=None):
    """Per-cell FBP (ST-X-3) surface spread: FFMC from the 1-h moisture map, 10 m open wind
    plus the slope-equivalent wind (eqs. 39-48) as a vector -> WSV, RAZ; ISI, ROS, BROS, L/B;
    Buildup Index effect BE (eq. 54) when the BUI is known."""
    p = FBP_FUELS[fuel]
    a, b, c = p["a"], p["b"], p["c"]
    be = 1.0 if not bui else float(np.exp(50 * np.log(p["q"]) * (1 / bui - 1 / p["bui0"])))
    m = L["m1"] * 100.0
    ff = 91.9 * np.exp(-0.1386 * m) * (1 + m ** 5.31 / 4.93e7)
    gs = np.minimum(100.0 * L["tan_slope"], FBP_SLOPE_MAX_PCT)
    sf = np.exp(3.533 * (gs / 100.0) ** 1.2)
    rsz = a * (1 - np.exp(-b * 0.208 * ff)) ** c
    rsf = rsz * sf
    x = 1 - np.power(np.minimum(rsf / a, 1.0), 1 / c)
    isf = np.where(x >= 0.01, np.log(np.maximum(x, 1e-12)), np.log(0.01)) / -b
    wse = np.log(np.maximum(isf, 1e-12) / (0.208 * ff)) / 0.05039
    wse2 = 28.0 - np.log(1 - np.minimum(0.999 * 2.496 * ff, isf) / (2.496 * ff)) / 0.0818
    wse = np.where(gs > 0, np.where(wse > 40, wse2, np.maximum(wse, 0.0)), 0.0)
    ws = U10 * 3.6
    wt, up = np.radians(wind_to_deg), np.radians(L["up_az"])
    wx, wy = ws * np.sin(wt) + wse * np.sin(up), ws * np.cos(wt) + wse * np.cos(up)
    wsv = np.hypot(wx, wy)
    raz = np.where(wsv > 0, np.degrees(np.arctan2(wx, wy)) % 360, wind_to_deg % 360)
    fw = np.where(wsv > 40, 12 * (1 - np.exp(-0.0818 * (wsv - 28))), np.exp(0.05039 * wsv))
    ros = a * (1 - np.exp(-b * 0.208 * fw * ff)) ** c * be / 60.0
    bros = a * (1 - np.exp(-b * 0.208 * np.exp(-0.05039 * wsv) * ff)) ** c * be / 60.0
    lb = 1.0 + 8.729 * (1 - np.exp(-0.030 * wsv)) ** 2.155
    lb = np.maximum(lb, 1.0)
    A = 0.5 * (ros + bros)
    return dict(a=A, b=A / lb, c=0.5 * (ros - bros), R_head=ros, R_back=bros, lb=lb, head_az=raz, wsv=wsv,
                ffmc=ffmc_from_moisture(m))


def field_normal_speed(field, idx, dir_deg):
    """Normal speeds (len(idx), len(dir_deg)) of the cell wavelets at flat indices idx for
    outward-normal azimuths dir_deg."""
    th = np.radians(np.asarray(dir_deg)[None, :] - field["head_az"].ravel()[idx][:, None])
    a, b, c = (field[k].ravel()[idx][:, None] for k in ("a", "b", "c"))
    ct, st = np.cos(th), np.sin(th)
    return c * ct + np.sqrt((a * ct) ** 2 + (b * st) ** 2)
