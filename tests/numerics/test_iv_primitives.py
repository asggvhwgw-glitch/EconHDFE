from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
from scipy.stats import chi2, f
from econhdfe.compute.vcov import ols_vcov
from econhdfe.compute.weights import prepare_weights
from econhdfe.models.linear_iv import diagnostics as diag
from econhdfe.models.linear_iv import estimators as iv_estimators

from econhdfe.iv import IVDesign, additive_moments, weighted_2sls
from econhdfe.models.linear_iv.estimators import fit_iv_kclass


def test_iv_design_is_outcome_agnostic_and_builds_full_x_z():
    n = 8
    c = np.arange(n, dtype=float)[:, None]
    e = (np.arange(n, dtype=float) + 1)[:, None]
    z = np.column_stack([np.ones(n), np.linspace(-1, 1, n)])
    d = IVDesign.from_arrays(c, e, z, n)
    assert d.X.shape == (n, 2)
    assert d.Z.shape == (n, 3)
    np.testing.assert_array_equal(d.X[:, 0], c[:, 0])
    np.testing.assert_array_equal(d.X[:, 1], e[:, 0])
    np.testing.assert_array_equal(d.Z[:, 0], c[:, 0])
    np.testing.assert_array_equal(d.Z[:, 1:], z)


def test_shared_weighted_2sls_matches_linear_iv_2sls_core():
    rng = np.random.default_rng(4201)
    n = 700
    c = rng.normal(size=(n, 2))
    z = rng.normal(size=(n, 2))
    e = 0.8 * z[:, :1] - 0.3 * z[:, 1:] + 0.4 * c[:, :1] + rng.normal(size=(n, 1))
    y = 0.5 * c[:, 0] - 0.2 * c[:, 1] + 1.4 * e[:, 0] + rng.normal(size=n)
    w = np.exp(0.15 * rng.normal(size=n))

    d = IVDesign.from_arrays(c, e, z, n)
    shared = weighted_2sls(y, d.X, d.Z, weights=w)
    beta, *_ = fit_iv_kclass(y, c, e, z, weights=w, estimator="2sls")
    np.testing.assert_allclose(shared.beta, beta, rtol=1e-12, atol=1e-12)


def test_additive_moment_primitive_matches_direct_formula():
    rng = np.random.default_rng(4202)
    n = 200
    q = rng.normal(size=(n, 4))
    r = rng.normal(size=n)
    w = rng.uniform(0.5, 1.5, size=n)
    got = additive_moments(r, q, weights=w, normalize=False)
    expected = q.T @ (w * r)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)

# Full-rank reference uses scalar NumPy/SVD solves. Rank-sensitive cases
# instead test compatibility with the original scalar GELSY policy.


def _solve(A, B, legacy=False):
    if legacy:
        return diag.la.lstsq(A, B, cond=None, lapack_driver='gelsy')[0]
    return np.linalg.lstsq(A, B, rcond=None)[0]


def _partial(A, C, legacy=False):
    return A.copy() if C.shape[1] == 0 else A - C @ _solve(C, A, legacy)


def _scalar_stage(dep, C, Z, *, vce, clusters=None, df_absorbed=0,
                  nested_adj=0, effective_n=None, score_scale=None,
                  df1_override=None, legacy=False, **covariance):
    zp = _partial(Z, C, legacy)
    ep = _partial(dep[:, None], C, legacy).ravel()
    b = _solve(zp, ep, legacy)
    resid = ep - zp @ b
    tss, rss = float(ep @ ep), float(resid @ resid)
    q, n_eff = Z.shape[1], len(dep) if effective_n is None else effective_n
    df1 = df1_override or q
    df2 = max(n_eff - C.shape[1] - q - df_absorbed, 1)
    classic = ((tss - rss) / df1) / (rss / df2) if rss > 0 else np.inf
    bread = np.linalg.pinv(zp.T @ zp, hermitian=True)
    V = ols_vcov(zp, resid, bread, kind=vce, clusters=clusters,
                 k_total=q + C.shape[1] + df_absorbed, nested_adj=nested_adj,
                 effective_n=n_eff, score_scale=score_scale, **covariance)
    wald = float(b @ np.linalg.pinv(V, hermitian=True) @ b)
    return dict(partial_r2=0.0 if tss <= 0 else np.clip(1 - rss / tss, 0, 1),
                f_classic=classic, f_classic_pvalue=f.sf(classic, df1, df2),
                f_robust=wald / df1, f_robust_chi2_pvalue=chi2.sf(wald, df1),
                excluded_instruments=q)


def _assert_same(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_same(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            _assert_same(a, b)
    else:
        np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=1e-9, equal_nan=True)


def _problem(k=5, n=240, controls=2):
    rng = np.random.default_rng(950 + k)
    C = rng.normal(size=(n, controls))
    Z = rng.normal(size=(n, k + 2))
    E = .6 * Z[:, :k] + rng.normal(size=(n, k))
    if controls:
        E += C @ rng.normal(scale=.2, size=(controls, k))
    return E, C, Z


def _compare_stages(E, C, Z, work, kwargs, options, legacy=False):
    first = diag.first_stage_diagnostics(E, C, Z, _workspace=work, **kwargs)
    Ew, Cw, Zw = work.E, work.C, work.Z
    shea = diag._shea_partial_r2(Ew, Cw, Zw)  # Unchanged, not a projection oracle.
    for j, got in enumerate(first):
        expected = _scalar_stage(Ew[:, j], Cw, Zw, legacy=legacy, **options)
        expected['shea_partial_r2'] = shea[j]
        _assert_same(got, expected)
    conditional = diag.sanderson_windmeijer_diagnostics(E, C, Z, _workspace=work, **kwargs)
    k = E.shape[1]
    if k == 1:
        _assert_same(conditional, [{'ap': first[0], 'sw': first[0], 'df1': Z.shape[1]}])
        return
    Q = np.column_stack([Cw, Zw])
    Ehat = Q @ _solve(Q, Ew, legacy)
    X, Xhat = np.column_stack([Cw, Ew]), np.column_stack([Cw, Ehat])
    for j, got in enumerate(conditional):
        keep = np.arange(X.shape[1]) != Cw.shape[1] + j
        b = _solve(Xhat[:, keep], Ew[:, j], legacy)
        df1 = Z.shape[1] - k + 1
        ap = _scalar_stage(Ew[:, j] - Xhat[:, keep] @ b, Cw, Zw,
                           df1_override=df1, legacy=legacy, **options)
        sw = _scalar_stage(Ew[:, j] - X[:, keep] @ b, Cw, Zw,
                           df1_override=df1, legacy=legacy, **options)
        _assert_same(got, {'ap': ap, 'sw': sw, 'df1': df1})


@pytest.mark.parametrize('k', [1, 5])
@pytest.mark.parametrize('vce,weight_type', [
    (v, w) for v in ('iid', 'robust', 'cluster', 'multiway', 'hac', 'dk')
    for w in (None, 'aweight', 'fweight', 'pweight')
    if not (w == 'fweight' and v in ('hac', 'dk'))
])
def test_workspace_matches_scalar_svd_oracle(k, vce, weight_type):
    E, C, Z = _problem(k, controls=0 if k == 1 else 2)
    n = len(E)
    winfo = prepare_weights(None if weight_type is None else 1.0 + np.arange(n) % 4, n, weight_type)
    w = winfo.estimation
    kwargs = dict(weights=w, weight_info=winfo, vce='cluster' if vce == 'multiway' else vce,
                  df_absorbed=3, nested_adj=int(vce in ('cluster', 'multiway')))
    if vce in ('cluster', 'multiway'):
        kwargs['clusters'] = [np.arange(n) % 20]
    if vce == 'multiway':
        kwargs['clusters'].append(np.arange(n) // 20)
    if vce in ('hac', 'dk'):
        kwargs.update(time=np.arange(n) // 12, panel=np.arange(n) % 12, bandwidth=2)
    options = {key: val for key, val in kwargs.items() if key not in ('weights', 'weight_info')}
    options['effective_n'] = winfo.effective_n
    if weight_type == 'fweight' and vce == 'robust':
        options['score_scale'] = 1 / np.sqrt(w)
    work = diag._IVDiagnosticWorkspace(E, C, Z, w)
    _compare_stages(E, C, Z, work, kwargs, options)
    cd = dict(weights=w, effective_n=winfo.effective_n, df_absorbed=3)
    assert diag.cragg_donald_stat(E, C, Z, **cd) == diag.cragg_donald_stat(E, C, Z, _workspace=work, **cd)
    _assert_same(diag.kleibergen_paap_stats(E, C, Z, **kwargs),
                 diag.kleibergen_paap_stats(E, C, Z, _workspace=work, **kwargs))


def test_workspace_reuses_projections_and_bounds_rhs(monkeypatch):
    E, C, Z = _problem(k=12)
    calls, batches = [], []
    original, tests = diag.la.lstsq, diag._IVDiagnosticWorkspace.tests
    def counted(A, B, *args, **kwargs):
        calls.append((A.shape, B.shape))
        return original(A, B, *args, **kwargs)
    def batched(self, dependent=None, **kwargs):
        if dependent is not None:
            batches.append(dependent.shape[1])
        return tests(self, dependent, **kwargs)
    monkeypatch.setattr(diag.la, 'lstsq', counted)
    monkeypatch.setattr(diag._IVDiagnosticWorkspace, 'tests', batched)
    work = diag._IVDiagnosticWorkspace(E, C, Z)
    for fn in (diag.first_stage_diagnostics, diag.cragg_donald_stat,
               diag.kleibergen_paap_stats, diag.sanderson_windmeijer_diagnostics):
        fn(E, C, Z, _workspace=work)
    assert len(calls) == 22  # Mechanism, not a machine-dependent timing assertion.
    assert batches == [8, 8, 8]
    assert work.bread is work.bread


@pytest.mark.parametrize('case', ['no_controls', 'duplicate_controls', 'duplicate_instrument',
                                  'readonly_strided', 'perfect', 'near_perfect'])
def test_workspace_rank_and_input_boundaries(case):
    E, C, Z = _problem(k=3)
    if case == 'no_controls':
        C = C[:, :0]
    elif case == 'duplicate_controls':
        C = np.column_stack([C, C[:, 0]])
    elif case == 'duplicate_instrument':
        Z = np.column_stack([Z, Z[:, 0]])
    elif case in ('perfect', 'near_perfect'):
        E = Z[:, :3].copy() + (0.0 if case == 'perfect' else 1e-10 * E)
    else:
        E, C, Z = E[::-2], C[::-2], Z[::-2]
        for A in (E, C, Z):
            A.flags.writeable = False
    copies = [A.copy() for A in (E, C, Z)]
    work = diag._IVDiagnosticWorkspace(E, C, Z)
    _compare_stages(E, C, Z, work, {'vce': 'robust'}, {'vce': 'robust'}, legacy=True)
    for A, before in zip((E, C, Z), copies):
        np.testing.assert_array_equal(A, before)


def test_workspace_empty_instruments_and_failure_retry():
    E, C, Z = _problem(k=1)
    out = diag.first_stage_diagnostics(E, C, Z[:, :0])
    assert out[0]['excluded_instruments'] == 0 and np.isnan(out[0]['f_robust'])
    assert diag.sanderson_windmeijer_diagnostics(E, C, Z[:, :0]) == []
    assert np.isnan(diag.cragg_donald_stat(E, C, Z[:, :0]))
    with pytest.raises(ValueError):
        diag._IVDiagnosticWorkspace(E, C[:-1], Z)
    expected = diag.first_stage_diagnostics(E, C, Z)
    good = E[0, 0]
    E[0, 0] = np.nan
    with pytest.raises(ValueError):
        diag.first_stage_diagnostics(E, C, Z)
    E[0, 0] = good
    _assert_same(diag.first_stage_diagnostics(E, C, Z), expected)


def test_workspace_units_and_permutations():
    E, C, Z = _problem(k=5)
    pe = np.array([4, 2, 0, 3, 1])
    changed = (E[:, pe] * [.5, 2, -1, 4, .25], C[:, ::-1] * [2, -.5],
               Z[:, ::-1] * np.linspace(.5, 2, Z.shape[1]))
    for fn in (diag.first_stage_diagnostics, diag.sanderson_windmeijer_diagnostics):
        original = fn(E, C, Z)
        _assert_same(fn(*changed), [original[j] for j in pe])
    _assert_same(diag.kleibergen_paap_stats(*changed), diag.kleibergen_paap_stats(E, C, Z))
    _assert_same(diag.cragg_donald_stat(*changed), diag.cragg_donald_stat(E, C, Z))


@pytest.mark.parametrize('estimator', ['2sls', 'liml', 'kclass', 'gmm2s'])
def test_workspace_is_per_fit_and_diagnostics_off_unchanged(monkeypatch, estimator):
    from econhdfe import ivhdfe, InferenceConfig
    E, C, Z = _problem(k=3)
    workspaces = []
    def make(*args, **kwargs):
        work = diag._IVDiagnosticWorkspace(*args, **kwargs)
        workspaces.append(work)
        return work
    monkeypatch.setattr(iv_estimators, '_IVDiagnosticWorkspace', make)
    kwargs = dict(y=E.sum(axis=1) + .2 * C[:, 0], exog=C, endog=E, instruments=Z,
                  absorb=[np.arange(len(E)) % 12], vce='robust', estimator=estimator,
                  inference_config=InferenceConfig(diagnostics='off'))
    if estimator == 'kclass':
        kwargs['kappa'] = .9
    a = ivhdfe(**kwargs)
    b = ivhdfe(**kwargs, weights=1.0 + np.arange(len(E)) % 3, weight_type='aweight')
    assert len(workspaces) == 2 and workspaces[0] is not workspaces[1]
    assert not np.array_equal(workspaces[0].Zp, workspaces[1].Zp)
    for result in (a, b):
        assert result.diagnostics_mode == 'off'
        assert len(result.first_stage['diagnostics']) == E.shape[1]
        assert len(result.diagnostics['sanderson_windmeijer']) == E.shape[1]
        assert set(result.diagnostics) >= {'cragg_donald_f', 'kleibergen_paap', 'overidentification', 'stock_yogo'}
