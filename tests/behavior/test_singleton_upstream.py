"""Singleton / FE-separation sample rules pinned to the Stata upstream commands.

Expected values come from Stata 18 MP runs of reghdfe 6.12.5, ivreghdfe and
ppmlhdfe 2.3.0 on exactly these data (e(N), e(num_singletons), e(N_full),
e(num_separated), _b[x]).  ivppmlhdfe 0.9.4 is not installed; its
expectations follow its source: first-stage drops are e(num_singletons),
while singletons created by simplex/ReLU trims count in e(num_separated).
"""
import numpy as np
import pandas as pd
import pytest

from econhdfe import IVPPMLConfig, PPMLConfig, ivhdfe, ivppmlhdfe, olshdfe, ppmlhdfe
from econhdfe.compute.block_design import BlockDesign, DenseDesignBlock
from econhdfe.hdfe.plan import FEPlan
from econhdfe.models.ppml.heterogeneous import fit_block_arrays as ppml_block
from econhdfe.models.ppml_iv.heterogeneous import fit_block_arrays as ivppml_block


def _coef(result, i=0):
    return float(result.params[i])


# group() mode: member rows, all regression variables and fw constant within gid.
GROUP = pd.DataFrame(dict(
    gid=[0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 8],
    fe=[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 3],
    fw=[1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 3, 3, 1, 1, 1, 1, 1],
    x=[0.001, 0.001, 0.299, 0.299, 0.299, -0.274, -0.274, -0.891, -0.891, -0.455, -0.455,
       -0.455, -0.992, -0.992, 0.06, 0.06, 1.34, 1.34, -0.492, -0.492, -0.492],
    y=[0.38, 0.38, 1.639, 1.639, 1.639, 1.22, 1.22, 0.66, 0.66, -0.158, -0.158, -0.158,
       0.475, 0.475, 1.725, 1.725, 0.326, 0.326, 0.296, 0.296, 0.296],
))
GROUP["z"] = GROUP.x + 0.1 * (GROUP.gid % 3)

# PPML: a=5 has an all-zero outcome; dropping it leaves b=6 with one row.
FE_SEP = pd.DataFrame(dict(
    y=[5.0, 5.0, 3.0, 0.0, 3.0, 1.0, 3.0, 4.0, 5.0, 1.0, 4.0, 2.0, 0.0, 3.0, 3.0, 3.0, 3.0,
       1.0, 5.0, 5.0, 0.0, 0.0, 0.0, 2.0],
    x=[0.034, 1.36, 1.225, -0.51, -0.298, -0.527, 0.57, -0.056, 0.747, -1.847, 1.567, -0.096,
       0.68, -0.137, -0.379, 0.463, 0.825, -0.203, -0.153, 0.686, 0.567, 1.302, -1.6, 0.25],
    a=[1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 1],
    b=[1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 6, 6],
))
FE_SEP["w"] = 1.0 + np.arange(len(FE_SEP)) % 3

# PPML: x2>0 only with y=0 (regressor separation); the trim leaves b=7 with one row.
SIMPLEX_SEP = pd.DataFrame(dict(
    y=[4.0, 6.0, 2.0, 3.0, 4.0, 5.0, 0.0, 1.0, 4.0, 5.0, 5.0, 3.0, 4.0, 6.0, 7.0, 5.0, 1.0, 0.0,
       4.0, 5.0, 0.0, 0.0, 3.0],
    x=[-0.802, -1.324, -0.248, 0.42, 1.136, 0.11, -0.553, -0.785, 0.749, 1.635, 0.273, -1.233,
       -0.958, 1.6, 0.203, -1.732, -0.084, -1.163, -0.629, -0.488, 0.4, -0.2, 0.6],
    x2=[0.0] * 20 + [1.0, 1.0, 0.0],
    a=[1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 1, 2, 3],
    b=[1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 7, 7],
))
SIMPLEX_SEP["w"] = 1.0 + np.arange(len(SIMPLEX_SEP)) % 3


def test_ols_fweight_singleton_uses_mass_and_aweight_uses_rows():
    # reghdfe y x [fw=fw], absorb(g): the one-row level with fw=3 is kept.
    d = pd.DataFrame(dict(g=[0, 0, 0, 1, 1, 1, 2, 3], fw=[1, 1, 1, 1, 1, 1, 3, 1],
                          x=[0.3, -1.2, 0.8, 1.5, -0.4, 0.9, 0.2, -0.7],
                          y=[1.0, 0.2, 2.1, 3.3, 0.4, 2.0, 1.1, 0.5]))
    fw = olshdfe(d, y="y", x=["x"], absorb=["g"], weights="fw", weight_type="fweight")
    assert (fw.nobs, fw.dropped_singletons) == (9, 1)          # Stata: N=9, num_singletons=1, N_full=10
    aw = olshdfe(d, y="y", x=["x"], absorb=["g"], weights="fw", weight_type="aweight")
    assert (aw.nobs, aw.dropped_singletons) == (6, 2)          # Stata: N=6, num_singletons=2
    assert _coef(fw) == pytest.approx(1.14473678539391, rel=1e-6)


def test_group_mode_fweight_singletons_use_group_level_mass():
    # reghdfe y x [fw=fw], absorb(fe) group(gid): fe=2 is one group with fw=3 -> kept.
    r = olshdfe(GROUP, y="y", x=["x"], absorb=["fe"], group="gid", weights="fw", weight_type="fweight")
    assert (r.nobs, r.dropped_singletons) == (12, 0)           # Stata: N=12, num_singletons=0, N_full=12
    assert _coef(r) == pytest.approx(-0.0902256477041823, rel=1e-6)
    assert float(r.stderr[0]) == pytest.approx(0.304086477284481, rel=1e-6)
    a = olshdfe(GROUP, y="y", x=["x"], absorb=["fe"], group="gid", weights="fw", weight_type="aweight")
    assert (a.nobs, a.dropped_singletons) == (8, 1)            # Stata [aw]: N=8, num_singletons=1
    u = olshdfe(GROUP, y="y", x=["x"], absorb=["fe"], group="gid")
    assert (u.nobs, u.dropped_singletons) == (8, 1)            # Stata: N=8, num_singletons=1
    assert _coef(u) == pytest.approx(-0.0351036152775875, rel=1e-6)


def test_group_mode_iv_fweight_singletons_use_group_level_mass():
    # ivreghdfe y (x=z) [fw=fw], absorb(fe) group(gid)
    r = ivhdfe(GROUP, y="y", endog=["x"], instruments=["z"], absorb=["fe"], group="gid",
               weights="fw", weight_type="fweight")
    assert (r.nobs, r.dropped_singletons) == (12, 0)           # Stata: N=12, num_singletons=0
    assert _coef(r) == pytest.approx(-0.063451093424691, rel=1e-6)
    assert float(r.stderr[0]) == pytest.approx(0.307308224289309, rel=1e-6)


def _ppml(d, xs, sep=None, w=None):
    cfg = PPMLConfig() if sep is None else PPMLConfig(separation=sep)
    return ppmlhdfe(y=d.y.to_numpy(), x=d[xs].to_numpy(), absorb=[d.a.to_numpy(), d.b.to_numpy()],
                    names=xs, weights=None if w is None else d[w].to_numpy(), config=cfg)


def _counts(r):
    return int(r.nobs), int(r.n_singletons), int(r.n_separated)


@pytest.mark.parametrize("sep, w, expected", [
    (None, None, (20, 4, 0)),            # default: FE separation joins the singleton loop
    (("fe",), None, (20, 4, 0)),
    (None, "w", (20, 4, 0)),             # [pw=w]
    (("simplex", "relu"), None, (20, 1, 3)),  # ReLU drops a=5, reload drops the b=6 singleton
    (("simplex", "relu"), "w", (20, 1, 3)),
    (("mu",), None, (20, 1, 3)),         # IRLS mu-separation trim also re-prunes singletons
])
def test_ppml_fe_separation_cascade_matches_ppmlhdfe(sep, w, expected):
    r = _ppml(FE_SEP, ["x"], sep, w)
    assert _counts(r) == expected
    assert r.nobs_full == 24                                   # e(N_full) = N + singletons + separated
    assert int(np.sum(~r.sample_mask)) == expected[1] + expected[2]
    assert int(np.sum(r.separation_mask)) == expected[2]
    b = 0.510367045052091 if w else 0.500562133096165
    assert _coef(r) == pytest.approx(b, rel=1e-6)


@pytest.mark.parametrize("sep", [None, ("simplex",), ("relu",), ("mu",)])
def test_ppml_new_singleton_after_separation_matches_ppmlhdfe(sep):
    r = _ppml(SIMPLEX_SEP, ["x", "x2"], sep)
    assert _counts(r) == (20, 1, 2)                            # Stata: N=20, num_singletons=1, num_separated=2
    assert r.nobs_full == 23
    assert not r.sample_mask[22] and not r.separation_mask[22]
    assert _coef(r) == pytest.approx(-0.237977172459743, rel=1e-6)


def _one_block(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    n, k = values.shape
    return BlockDesign(n, k, (DenseDesignBlock(0, np.arange(n), np.arange(k), values),))


@pytest.mark.parametrize("d, xs, expected, b", [
    (FE_SEP, ["x"], (20, 4, 0), 0.500562133096165),
    (SIMPLEX_SEP, ["x", "x2"], (20, 1, 2), -0.237977172459743),
])
def test_ppml_block_path_uses_the_same_upstream_sample(d, xs, expected, b):
    n = len(d)
    r = ppml_block(d.y.to_numpy(), _one_block(d[xs].to_numpy()),
                   FEPlan.from_arrays([d.a.to_numpy(), d.b.to_numpy()]),
                   offset=np.zeros(n), true_w=np.ones(n), vce="robust", clusters=None,
                   names=tuple(xs), config=PPMLConfig())
    assert _counts(r) == expected
    assert _coef(r) == pytest.approx(b, rel=1e-6)


def _ivppml(d, wk=None):
    xs = [c for c in ("x2",) if c in d]
    z = (d.x + 0.3 * np.sin(np.arange(len(d)))).to_numpy()[:, None]
    return ivppmlhdfe(y=d.y.to_numpy(), exog=d[xs].to_numpy() if xs else None, endog=d[["x"]].to_numpy(),
                      instruments=z, absorb=[d.a.to_numpy(), d.b.to_numpy()],
                      weights=None if wk is None else d.w.to_numpy(), weight_type=wk,
                      exog_names=xs or None, endog_names=["x"])


@pytest.mark.parametrize("wk, rows, expected", [
    (None, 20, (4, 0)), ("pweight", 20, (4, 0)), ("fweight", 20, (4, 0)),
])
def test_ivppml_first_stage_uses_rows_and_joint_fe_separation(wk, rows, expected):
    # ivppmlhdfe source: fixed_effects(absorb, touse, "iweight", depvar) -> user
    # weights never enter; FE-separated rows land in e(num_singletons).
    r = _ivppml(FE_SEP, wk)
    assert int(r.sample_mask.sum()) == rows
    assert (int(r.n_singletons), int(r.n_separated)) == expected


@pytest.mark.parametrize("wk, rows, expected", [
    (None, 20, (0, 3)), ("pweight", 20, (0, 3)),
    # fweight: trim_separated_obs reloads with the user's fweight, and the
    # remaining b=7 row has fw=2, so it is not a singleton.
    ("fweight", 21, (0, 2)),
])
def test_ivppml_post_separation_singletons_count_as_separated(wk, rows, expected):
    r = _ivppml(SIMPLEX_SEP, wk)
    assert int(r.sample_mask.sum()) == rows
    assert (int(r.n_singletons), int(r.n_separated)) == expected
    assert r.sample_mask[22] == (wk == "fweight")


def test_ivppml_block_path_matches_dense_sample():
    d = SIMPLEX_SEP
    n = len(d)
    z = (d.x + 0.3 * np.sin(np.arange(n))).to_numpy()
    r = ivppml_block(
        d.y.to_numpy(), _one_block(d[["x2"]].to_numpy()), _one_block(d.x.to_numpy()), _one_block(z),
        FEPlan.from_arrays([d.a.to_numpy(), d.b.to_numpy()]), offset=np.zeros(n), true_w=np.ones(n),
        vce="robust", exog_names=("x2",), endog_names=("x",), instrument_names=("z",), config=IVPPMLConfig(),
    )
    dense = _ivppml(d)
    np.testing.assert_array_equal(r.sample_mask, dense.sample_mask)
    assert (int(r.n_singletons), int(r.n_separated)) == (0, 3)
