import numpy as np
from econhdfe import ivppmlhdfe
from econhdfe.models.ppml_iv import IVPPMLConfig
from econhdfe.effects import recover_ivppml_result


def _contribution(res, *groups):
    out = np.zeros(len(groups[0]), dtype=float)
    for term, group in zip(res.terms, groups, strict=False):
        mapping = {v: term.coefficients[j] for j, v in enumerate(term.levels.tolist())}
        out += np.array([mapping[v] for v in np.asarray(group)], dtype=float)
    return out


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
        y=y, exog=c[:, None], endog=x[:, None], instruments=z[:, None], absorb=[origin, dest],
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
    for name, b in zip(fit.names, fit.params, strict=False):
        if name == "_cons":
            xb += b
        else:
            xb += b * name_to_col[name][sample]
    target = fit.eta[sample] - xb
    np.testing.assert_allclose(
        _contribution(fe, origin[sample], dest[sample]), target, atol=3e-7, rtol=3e-7
    )
