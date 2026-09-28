import numpy as np
import pandas as pd

from econhdfe import olshdfe, ivhdfe, ppmlhdfe, ivppmlhdfe
from econhdfe.models.ppml import PPMLConfig
from econhdfe.models.ppml_iv import IVPPMLConfig
from econhdfe.effects import (
    NormalizationSpec,
    recover_linear_result,
    recover_ppml_result,
    recover_ivppml_result,
)


def _contribution(res, *groups):
    out = np.zeros(len(groups[0]), dtype=float)
    for term, group in zip(res.terms, groups, strict=False):
        mapping = {v: term.coefficients[j] for j, v in enumerate(term.levels.tolist())}
        out += np.array([mapping[v] for v in np.asarray(group)], dtype=float)
    return out


def test_ols_adapter_recovers_akm_style_worker_firm_effects():
    rng = np.random.default_rng(1001)
    n = 5000
    worker = rng.integers(0, 350, n)
    firm = rng.integers(0, 70, n)
    x = rng.normal(size=n)
    aw = rng.normal(scale=0.7, size=350)
    pf = rng.normal(scale=0.4, size=70)
    y = 1.25 * x + aw[worker] + pf[firm] + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame({"y": y, "x": x, "worker": worker, "firm": firm})

    fit = olshdfe(df, y="y", x=["x"], absorb=["worker", "firm"], drop_singletons=False)
    fe = recover_linear_result(
        fit, df[["x"]].to_numpy(), [worker, firm], x_names=["x"], fe_names=["worker", "firm"]
    )
    target = fit.fitted - df[["x"]].to_numpy() @ fit.params
    np.testing.assert_allclose(_contribution(fe, worker, firm), target, atol=2e-8, rtol=2e-8)
    assert fe.identification.extra_nullity == 0
    assert fe.diagnostics.normalization_complete
    assert len(fe.term("worker").coefficients) + len(fe.term("firm").coefficients) < n


def test_linear_iv_adapter_uses_same_contract():
    rng = np.random.default_rng(1002)
    n = 4500
    worker = rng.integers(0, 300, n)
    firm = rng.integers(0, 60, n)
    z = rng.normal(size=n)
    c = rng.normal(size=n)
    aw = rng.normal(scale=0.5, size=300)
    pf = rng.normal(scale=0.3, size=60)
    x = 0.9 * z + 0.2 * c + rng.normal(scale=0.6, size=n)
    y = 1.4 * x + 0.25 * c + aw[worker] + pf[firm] + rng.normal(scale=0.3, size=n)
    df = pd.DataFrame({"y": y, "x": x, "c": c, "z": z, "worker": worker, "firm": firm})

    fit = ivhdfe(
        df, y="y", exog=["c"], endog=["x"], instruments=["z"],
        absorb=["worker", "firm"], drop_singletons=False,
    )
    X = df[["c", "x"]].to_numpy()
    fe = recover_linear_result(
        fit, X, [worker, firm], x_names=["c", "x"], fe_names=["worker", "firm"]
    )
    active = {name: j for j, name in enumerate(["c", "x"])}
    Xa = X[:, [active[nm] for nm in fit.names]]
    target = fit.fitted - Xa @ fit.params
    np.testing.assert_allclose(_contribution(fe, worker, firm), target, atol=3e-8, rtol=3e-8)


def test_ppml_adapter_artuc_mclaren_style_origin_destination_reference_normalization():
    rng = np.random.default_rng(1003)
    n_origin, n_dest = 45, 35
    origin, dest = np.meshgrid(np.arange(n_origin), np.arange(n_dest), indexing="ij")
    origin, dest = origin.ravel(), dest.ravel()
    n = len(origin)
    cost = rng.normal(size=n)
    ao = rng.normal(scale=0.25, size=n_origin)
    ld = rng.normal(scale=0.20, size=n_dest)
    eta = 0.18 * cost + ao[origin] + ld[dest]
    y = rng.poisson(np.exp(eta))
    # Avoid an all-zero FE level in a small random realization.
    y += (np.bincount(origin, weights=(y > 0), minlength=n_origin)[origin] == 0)
    y += (np.bincount(dest, weights=(y > 0), minlength=n_dest)[dest] == 0)

    fit = ppmlhdfe(
        y, cost[:, None], absorb=[origin, dest], names=["cost"], vce="model",
        config=PPMLConfig(engine="optimized", separation=("fe", "simplex", "relu")),
    )
    fe = recover_ppml_result(
        fit, cost[:, None], [origin, dest], x_names=["cost"],
        fe_names=["origin", "destination"],
        normalization=NormalizationSpec(
            "reference", baseline="origin", references={"destination": 0}
        ),
    )
    d = fe.term("destination")
    assert abs(d.coefficients[d.index_of(0)]) < 1e-10
    sample = fit.sample_mask
    target = fit.eta[sample] - cost[sample] * fit.coef[0]
    np.testing.assert_allclose(
        _contribution(fe, origin[sample], dest[sample]), target, atol=2e-7, rtol=2e-7
    )
    assert fe.diagnostics.normalization_complete


def test_ivppml_adapter_reconstructs_eta_after_reported_constant():
    rng = np.random.default_rng(1004)
    n = 5000
    origin = rng.integers(0, 100, n)
    dest = rng.integers(0, 80, n)
    z = rng.normal(size=n)
    c = rng.normal(size=n)
    x = 0.75 * z + 0.25 * c + rng.normal(scale=0.7, size=n)
    ao = rng.normal(scale=0.18, size=100)
    dd = rng.normal(scale=0.15, size=80)
    mu = np.exp(0.12 * x + 0.08 * c + ao[origin] + dd[dest])
    y = rng.poisson(mu)

    fit = ivppmlhdfe(
        y, exog=c[:, None], endog=x[:, None], instruments=z[:, None], absorb=[origin, dest],
        exog_names=["c"], endog_names=["x"], instrument_names=["z"], vce="model",
        config=IVPPMLConfig(engine="optimized", separation=("fe", "simplex", "relu")),
    )
    X = np.column_stack([c, x])
    fe = recover_ivppml_result(
        fit, X, [origin, dest], x_names=["c", "x"], fe_names=["origin", "destination"]
    )
    sample = fit.sample_mask
    name_to_col = {"c": c, "x": x}
    xb = np.zeros(int(sample.sum()))
    for name, b in zip(fit.names, fit.coef, strict=False):
        if name == "_cons":
            xb += b
        else:
            xb += b * name_to_col[name][sample]
    target = fit.eta[sample] - xb
    np.testing.assert_allclose(
        _contribution(fe, origin[sample], dest[sample]), target, atol=3e-7, rtol=3e-7
    )


def test_linear_adapter_reconstructs_recursive_singleton_sample_from_raw_groups():
    # Two connected core pairs plus one recursively pruned leaf observation.
    worker = np.array([0, 0, 1, 1, 2])
    firm = np.array([0, 1, 0, 1, 2])
    x = np.array([0.2, -0.1, 0.3, 0.4, 1.0])
    y = 0.5 * x + np.array([0.2, 0.2, -0.1, -0.1, 0.7]) + np.array([0.1, -0.2, 0.1, -0.2, 0.4])
    df = pd.DataFrame({"y": y, "x": x, "worker": worker, "firm": firm})
    fit = olshdfe(df, y="y", x=["x"], absorb=["worker", "firm"], drop_singletons=True)
    assert fit.dropped_singletons == 1
    fe = recover_linear_result(
        fit, df[["x"]].to_numpy(), [worker, firm], x_names=["x"], fe_names=["worker", "firm"]
    )
    keep = np.array([True, True, True, True, False])
    target = fit.fitted - df.loc[keep, ["x"]].to_numpy() @ fit.params
    np.testing.assert_allclose(_contribution(fe, worker[keep], firm[keep]), target, atol=2e-8, rtol=2e-8)
    assert any(issue.code == "fe_recovery.singleton_pruned" for issue in fe.diagnosis.issues)
    assert 2 not in set(fe.term("worker").levels.tolist())
    assert 2 not in set(fe.term("firm").levels.tolist())
