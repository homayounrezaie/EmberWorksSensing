"""Unit tests for the numerical core (no data download needed).
Run: python -m pytest -q tests/"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import discover as D  # noqa: E402
from pipeline import forecast as FC  # noqa: E402
from pipeline import physics as PH  # noqa: E402
from pipeline.export import rle  # noqa: E402

CH = 0.005588  # m/s per chain/hour


@pytest.mark.parametrize("fm,expected", [(PH.FM1, 78.0),
                                         (PH.FuelModel("FM3", 3.01, 0, 0, 0, 0, 1500, 1500, 1500, 2.5, 0.25, dynamic=False), 104.0)])
def test_rothermel_matches_anderson_1982_table(fm, expected):
    R, _ = PH.rothermel(fm, 0.08, 0.09, 0.10, 1.0, 1.0, 5 * PH.MPH)
    assert abs(R / CH - expected) / expected < 0.06


def test_rothermel_monotone_in_wind_and_moisture():
    r = [PH.rothermel(PH.GR2, 0.08, 0.09, 0.1, 0.8, 1.3, u)[0] for u in (0, 0.5, 1, 2)]
    assert np.all(np.diff(r) > 0)
    assert PH.rothermel(PH.GR2, 0.05, .06, .07, .8, 1.3, 1)[0] > PH.rothermel(PH.GR2, 0.12, .13, .14, .8, 1.3, 1)[0]


def test_wulff_of_ellipse_support_is_the_ellipse():
    el = PH.ellipse(0.1, 2.0)
    th = np.radians(FC.DIRS)  # wind blowing to azimuth 0
    r = FC.wulff_radius(PH.ellipse_normal_speed(th, el)[None, :])[0]
    # analytic polar radius of the offset ellipse from the ignition point
    ct, st = np.cos(th), np.sin(th)
    A = (ct / el["a"]) ** 2 + (st / el["b"]) ** 2
    Bq = -2 * el["c"] * ct / el["a"] ** 2
    Cq = (el["c"] / el["a"]) ** 2 - 1
    exact = (-Bq + np.sqrt(Bq ** 2 - 4 * A * Cq)) / (2 * A)
    assert np.allclose(r, exact, rtol=0.01)
    assert abs(r[0] - 0.1) < 1e-3 and abs(r[36] - el["R_back"]) < 1e-3


def test_arrival_time_isotropic_disc():
    R = 0.05
    r = FC.wulff_radius(np.full((1, FC.NDIR), R))
    AT = FC.arrival_time(np.array([50.0]), np.array([50.0]), np.array([0.0]), r, np.array([0.0]), (101, 101), 1000, 3.0)
    rr, cc = np.mgrid[:101, :101]
    d = np.hypot(rr - 50, cc - 50) * 3.0
    ok = np.isfinite(AT)
    assert np.allclose(AT[ok], d[ok] / R, rtol=0.02, atol=1.0)
    assert not ok[0, 0]  # beyond the horizon


def _synthetic_front(n, rng):
    th = rng.uniform(-np.pi, np.pi, n)
    F = pd.DataFrame(dict(theta=np.degrees(th), phi=(np.degrees(th) + 60.0) % 360, Umid=rng.choice([0.5, 0.8, 1.1], n),
                          frfd=rng.gamma(2, 3, n), kappa=rng.normal(0, 0.03, n), m1=rng.uniform(0.07, 0.10, n),
                          tan_s=rng.uniform(0, 0.08, n), up_az=rng.uniform(0, 360, n), R0=rng.uniform(0.001, 0.005, n),
                          P=rng.gamma(2, 0.02, n), Pm=rng.gamma(2, 0.02, n), Rr=rng.uniform(0.002, 0.03, n),
                          Rf=rng.uniform(0.01, 0.2, n), Ts=rng.uniform(28, 33, n), pass_id=rng.integers(0, 6, n),
                          tile=rng.integers(0, 20, n)))
    return F


def test_stlsq_recovers_sparse_law():
    rng = np.random.default_rng(1)
    keys = D.DEFAULT_KEYS
    F = _synthetic_front(6000, rng)
    true = np.zeros(len(keys))
    true[keys.index("1")], true[keys.index("Icos")], true[keys.index("s")] = 0.03, 0.005, 0.4
    A, _ = D.design(F.assign(ros=0.0), keys)
    F["ros"] = A @ true + rng.normal(0, 0.005, len(F))
    AtA, Aty, yty, nn = D.sum_stats(D.block_stats(F, keys))
    beta, xi, active = D.stlsq(AtA, Aty, nn, lam=0.003)
    want = {keys.index("1"), keys.index("Icos"), keys.index("s")}
    assert set(np.flatnonzero(active)) == want
    assert np.allclose(beta[sorted(want)], true[sorted(want)], rtol=0.05, atol=0.002)
    assert D.r2(beta, AtA, Aty, yty, nn)[0] > 0.9


def test_sign_priors_drop_unphysical_terms():
    rng = np.random.default_rng(2)
    keys = D.DEFAULT_KEYS
    F = _synthetic_front(4000, rng)
    true = np.zeros(len(keys)); true[keys.index("1")], true[keys.index("s")] = 0.05, -0.5   # "upslope slows the fire"
    A, _ = D.design(F.assign(ros=0.0), keys)
    F["ros"] = A @ true + rng.normal(0, 0.002, len(F))
    AtA, Aty, _, nn = D.sum_stats(D.block_stats(F, keys))
    _, _, free = D.stlsq(AtA, Aty, nn, lam=0.003)
    _, _, con = D.stlsq(AtA, Aty, nn, lam=0.003, signs=D.signs_for(keys))
    assert free[keys.index("s")] and not con[keys.index("s")]


def test_rothermel_field_matches_scalar_model_and_adds_slope():
    from pipeline.landscape import QUANT
    tab = PH.rothermel_tables(["TU1", "TU2", "TL3"], QUANT["m1"])
    m1 = QUANT["m1"][0] + 140 * QUANT["m1"][1]
    L = dict(fuel=np.array([[1, 1]]), m1=np.array([[m1, m1]]), waf=np.array([[0.12, 0.12]]),
             tan_slope=np.array([[0.0, 0.1]]), up_az=np.array([[0.0, 60.0]]))
    f = PH.rothermel_field(L, tab, 6.0, 60.0, QUANT["m1"])
    R, _ = PH.rothermel(PH.TU2, m1, m1 + PH.M10_ADD, m1 + PH.M100_ADD, PH.M_HERB, PH.M_WOOD, 0.12 * 6.0)
    assert np.isclose(f["R_head"][0, 0], R, rtol=1e-12)                 # flat: same as the scalar Rothermel
    assert f["R_head"][0, 1] > f["R_head"][0, 0]                        # upslope aligned with the wind: faster
    assert np.isclose(f["head_az"][0, 1], 60.0) and f["lb"][0, 1] > f["lb"][0, 0]
    L2 = dict(L, up_az=np.array([[0.0, 150.0]]))
    g = PH.rothermel_field(L2, tab, 6.0, 60.0, QUANT["m1"])
    assert 60.0 < g["head_az"][0, 1] < 150.0                            # head turns towards upslope


def test_fbp_field_matches_scalar_fbp_on_flat_ground():
    L = dict(m1=np.array([0.09]), tan_slope=np.array([0.0]), up_az=np.array([0.0]))
    f = PH.fbp_field(L, 7.9, 60.0)
    r = PH.fbp(9.0, 7.9 * 3.6)
    assert np.isclose(f["R_head"][0], r["head"], rtol=1e-12) and np.isclose(f["lb"][0], r["lb"], rtol=1e-12)
    up = PH.fbp_field(dict(L, tan_slope=np.array([0.2]), up_az=np.array([60.0])), 7.9, 60.0)
    assert up["R_head"][0] > f["R_head"][0]


def test_fuel_moisture_from_thermal_and_canopy_wind():
    from pipeline.landscape import fuel_moisture_from_thermal, waf_canopy
    m_cool, _ = fuel_moisture_from_thermal(np.array([28.9]), 28.9, 52.0)
    m_warm, rh = fuel_moisture_from_thermal(np.array([35.0]), 28.9, 52.0)
    assert np.isclose(m_cool[0], PH.emc_simard(84.0, 52.0), atol=2e-4)  # fuel at air temperature -> air EMC
    assert m_warm[0] < m_cool[0] and rh[0] < 52.0                       # sun-warmed fuel is drier
    waf, f = waf_canopy(np.array([55.0, 10.0]), np.array([19.0, 19.0]), np.array([4.3, 4.3]), np.array([1.0, 1.0]))
    assert f[0] > 0.05 > f[1] and waf[0] < waf[1]                       # sheltered under the canopy


def test_mtt_uniform_isotropic_field():
    H, W = 21, 21
    fld = dict(a=np.full((H, W), 0.1), b=np.full((H, W), 0.1), c=np.zeros((H, W)), head_az=np.zeros((H, W)))
    inv = FC.mtt_inverse_speeds(fld, 0.5)
    AT = FC.mtt_arrival([10.0], [10.0], [0.0], inv, np.ones((H, W), bool), (H, W), 1e9, 0.5)
    assert np.isclose(AT[10, 20], 10 * 0.5 / 0.1) and np.isclose(AT[0, 10], 50.0)


def test_metrics_perfect_and_empty_forecasts():
    fuel = np.ones((60, 60), bool)
    old = np.zeros_like(fuel); old[20:40, 20:30] = True
    act = old.copy(); act[20:40, 30:36] = True
    AT = np.full(fuel.shape, np.inf); AT[act & ~old] = 50.0
    m = FC.metrics(act, act, old, fuel, AT, 100.0)
    assert m["iou"] == 1 and m["iou_new"] == 1 and m["front_mean_m"] == 0 and m["arrival_err_s"] == 0
    m0 = FC.metrics(old, act, old, fuel, np.full(fuel.shape, np.inf), 100.0)
    assert m0["iou_new"] == 0 and m0["front_mean_m"] > 0 and m0["arrival_err_s"] == pytest.approx(100.0)


def test_rle_roundtrip():
    rng = np.random.default_rng(0)
    m = rng.random((37, 53)) > 0.6
    code = rle(m)
    out = np.zeros(m.size, bool); v, p = code[0], 0
    for run in code[1:]:
        out[p:p + run] = bool(v); p += run; v = 1 - v
    assert (out.reshape(m.shape) == m).all()


def test_effective_wind_inverts_head_rate():
    p = PH.PhysicsParams()
    R = PH.interval_ellipse(3.0, p)["R_head"]
    assert abs(PH.effective_wind(R, p) - 3.0) < 0.06


def test_fm2_matches_anderson_table_and_emc():
    R, _ = PH.rothermel(PH.FM2, 0.08, 0.09, 0.10, 1.0, 1.0, 5 * PH.MPH)
    assert abs(R / CH - 35.0) / 35.0 < 0.10           # Anderson (1982): FM2 35 ch/h at 8 %, 5 mi/h
    assert 0.09 < PH.emc_simard(84.0, 52.0) < 0.10      # Simard (1968) EMC, Hanna Hammock afternoon


def test_crop_box_is_clipped_to_the_array():
    from pipeline.plots import crop_box
    U = np.zeros((50, 40), bool); U[2:48, 1:39] = True
    r0, r1, c0, c1 = crop_box(U, margin=20)
    assert (r0, r1, c0, c1) == (0, 50, 0, 40)


def test_radiant_flux_proxy():
    from pipeline.flame3 import radiant_flux, SITE_GAS_T_BG
    assert radiant_flux(np.array([SITE_GAS_T_BG]))[0] == 0.0
    q = radiant_flux(np.array([600.0]))[0] / 1000      # kW/m2 at 600 C above a 30 C background
    assert 31 < q < 34


def test_fwi_codes_match_van_wagner_pickett_1985():
    """First day of the standard FWI test data (17 C, 42 %, 25 km/h, no rain; start 85 / 6 / 15, April)."""
    from pipeline import moisture as MO
    assert abs(MO.ffmc_step(85.0, 17.0, 42.0, 25.0, 0.0) - 87.692980) < 1e-4
    assert abs(MO.dc_step(15.0, 17.0, 0.0, 4) - 19.0133) < 1e-3
    k = 1.894 * (17 + 1.1) * (100 - 42) * MO.LE_15_33N[3] * 1e-6        # DMC with this latitude's day-length factor
    assert abs(MO.dmc_step(6.0, 17.0, 42.0, 0.0, 4) - (6 + 100 * k)) < 1e-9
    assert abs(MO.bui_of(8.5450225, 19.0133) - 8.4904) < 1e-3


def test_live_moisture_relative_position():
    from pipeline import moisture as MO
    d = pd.Series(pd.date_range("2019-01-01", "2023-02-20", freq="10D").astype(str))
    v = pd.Series(np.linspace(0.0, 0.4, len(d)))
    r = MO.relative_position(v, d)
    assert r["date"] == d.iloc[-1][:10] and 0.9 < r["r"] <= 1.0
    lm = MO.live_moisture(0.0)
    assert lm["herb"] == 0.30 and lm["wood"] == 0.60


def test_forced_terms_are_kept():
    rng = np.random.default_rng(3)
    keys = D.DEFAULT_KEYS
    F = _synthetic_front(3000, rng)
    F["ros"] = 0.05 + rng.normal(0, 0.01, len(F))                      # nothing but noise around a constant
    AtA, Aty, _, nn = D.sum_stats(D.block_stats(F, keys))
    _, _, act = D.stlsq(AtA, Aty, nn, lam=0.05, forced=D.forced_for(keys, ("Rr", "Rf")))
    assert act[keys.index("Rr")] and act[keys.index("Rf")] and act.sum() == 3
