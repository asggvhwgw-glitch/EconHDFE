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
    tall = [shape for shape, _ in calls if shape[0] == len(E)]
    small = [shape for shape, _ in calls if shape[0] != len(E)]
    assert len(tall) == 3  # C projections of E/Z, then one multi-RHS reduced form.
    assert small == [(Z.shape[1], E.shape[1] - 1)] * E.shape[1]
    assert batches == []  # No further observation-level AP/SW regression batches.
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


def _full_score_kp_reference(E, C, Z, *, weights=None, weight_info=None,
                             vce='robust', clusters=None, time=None, panel=None,
                             bandwidth=None):
    """Independent full Kronecker path for the reduced-score optimization.

    The canonical transform, rank statistic and raw covariance engine are
    unchanged; only the full versus factored moment computation is compared.
    """
    from econhdfe.compute.vcov import score_covariance
    neff = None if weight_info is None else weight_info.effective_n
    Ep, Zp, pi, theta, rz, ry, _, neff = diag._canonical_components(
        E, C, Z, weights, effective_n=neff)
    transform = np.kron(ry.T, rz.T)
    out = []
    scale = None
    if weight_info is not None and weight_info.kind == 'fweight' and vce == 'robust':
        scale = 1 / np.sqrt(weight_info.estimation)
    for V in (Ep, Ep - Zp @ pi):
        scores = np.einsum('ni,nj->nij', V, Zp).reshape(len(Ep), -1)
        S = score_covariance(scores, kind=vce, clusters=clusters,
                             small_sample=False, time=time, panel=panel,
                             bandwidth=bandwidth, kernel='bartlett',
                             effective_n=neff, score_scale=scale)
        out.append(diag._kp_stat(theta, transform @ (S / neff) @ transform.T, neff))
    return out


@pytest.mark.parametrize('k', [1, 4])
@pytest.mark.parametrize('vce', ['iid', 'robust', 'cluster', 'multiway', 'hac', 'dk'])
def test_kp_projected_scores_match_full_kronecker_reference(k, vce):
    E, C, Z = _problem(k=k, n=280)
    n = len(E)
    kind = 'cluster' if vce == 'multiway' else vce
    wi = prepare_weights(1. + np.arange(n) % 4, n,
                         'aweight' if vce in ('hac', 'dk') else 'fweight')
    kwargs = dict(vce=kind, weights=wi.estimation, weight_info=wi)
    if vce in ('cluster', 'multiway'):
        kwargs['clusters'] = [np.arange(n) % 20]
    if vce == 'multiway':
        kwargs['clusters'].append(np.arange(n) // 20)
    if vce in ('hac', 'dk'):
        kwargs.update(time=2*(np.arange(n)//7), panel=np.arange(n)%7, bandwidth=3)
    (lm, lr), (wald, wr) = _full_score_kp_reference(E, C, Z, **kwargs)
    got = diag.kleibergen_paap_stats(E, C, Z, **kwargs)
    _assert_same(got['rk_lm'], lm)
    _assert_same(got['rk_wald_chi2'], wald)
    assert (got['rank_lm_cov'], got['rank_wald_cov']) == (lr, wr)


def test_kp_only_materializes_projected_scores(monkeypatch):
    E, C, Z = _problem(k=6)
    shapes = []
    original = diag._kron_scores
    def capture(v, z, *args, **kwargs):
        shapes.append((v.shape, z.shape))
        return original(v, z, *args, **kwargs)
    monkeypatch.setattr(diag, '_kron_scores', capture)
    diag.kleibergen_paap_stats(E, C, Z)
    # L*K = 48 full scores are replaced by L-K+1 = 3 scores per observation.
    assert shapes == [((len(E), 1), (len(E), 3))] * 2


@pytest.mark.parametrize('case', ['collinear', 'huge_controls', 'tiny_instruments', 'perfect'])
def test_conditional_reduction_uses_original_path_at_boundaries(case):
    E, C, Z = _problem(k=4)
    if case == 'collinear':
        E[:, -1] = E[:, 0] + 1e-7 * E[:, -1]
    elif case == 'huge_controls':
        C *= 1e20
    elif case == 'tiny_instruments':
        Z *= 1e-20
    else:
        E = Z[:, :4].copy()
    work = diag._IVDiagnosticWorkspace(E, C, Z)
    assert diag._conditional_from_reduced_form(work, vce='robust') is None


def test_kp_whitening_guard_keeps_full_score_order(monkeypatch):
    E, C, Z = _problem(k=4)
    E[:, -1] = E[:, 0] + 1e-7 * E[:, -1]
    shapes = []
    original = diag._kron_scores
    def capture(v, z, *args, **kwargs):
        shapes.append((v.shape[1], z.shape[1]))
        return original(v, z, *args, **kwargs)
    monkeypatch.setattr(diag, '_kron_scores', capture)
    got = diag.kleibergen_paap_stats(E, C, Z)
    assert shapes == [(E.shape[1], Z.shape[1])] * 2
    (lm, lr), (wald, wr) = _full_score_kp_reference(E, C, Z)
    _assert_same(got['rk_lm'], lm)
    _assert_same(got['rk_wald_chi2'], wald)
    assert (got['rank_lm_cov'], got['rank_wald_cov']) == (lr, wr)


def test_kp_covariance_rank_boundary_uses_full_score_order(monkeypatch):
    E, C, Z = _problem(k=4)
    n = len(E)
    shapes = []
    original = diag._kron_scores
    def capture(v, z, *args, **kwargs):
        shapes.append((v.shape[1], z.shape[1]))
        return original(v, z, *args, **kwargs)
    monkeypatch.setattr(diag, '_kron_scores', capture)
    kwargs = dict(vce='cluster', clusters=[np.arange(n) % 2])
    got = diag.kleibergen_paap_stats(E, C, Z, **kwargs)
    # Three final scores with only two groups: preserve the original rank edge.
    assert (E.shape[1], Z.shape[1]) in shapes
    (lm, lr), (wald, wr) = _full_score_kp_reference(E, C, Z, **kwargs)
    _assert_same(got['rk_lm'], lm)
    _assert_same(got['rk_wald_chi2'], wald)
    assert (got['rank_lm_cov'], got['rank_wald_cov']) == (lr, wr)


def test_reduced_form_is_lazy_and_reused_without_changing_inputs(monkeypatch):
    E, C, Z = _problem(k=4)
    work = diag._IVDiagnosticWorkspace(E, C, Z)
    assert work._reduced_form is None
    copies = [a.copy() for a in (E, C, Z)]
    fit = work.reduced_form
    def must_not_refactor(*args, **kwargs):
        raise AssertionError('repeated first-stage solve')
    monkeypatch.setattr(diag.la, 'lstsq', must_not_refactor)
    assert work.reduced_form is fit
    work.tests(vce='robust', clusters=None, df_absorbed=0, nested_adj=0)
    for before, after in zip(copies, (E, C, Z)):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize('k', [1, 5])
@pytest.mark.parametrize('vce', ['iid', 'robust', 'cluster', 'multiway', 'hac', 'dk'])
def test_joint_first_stage_suite_reuses_only_identical_covariance(monkeypatch, k, vce):
    E, C, Z = _problem(k=k, n=280)
    n = len(E)
    wi = prepare_weights(1. + np.arange(n) % 4, n,
                         'aweight' if vce in ('hac', 'dk') else 'fweight')
    options = dict(vce='cluster' if vce == 'multiway' else vce,
                   weights=wi.estimation, weight_info=wi, df_absorbed=3)
    if vce in ('cluster', 'multiway'):
        options.update(clusters=[np.arange(n) % 20], nested_adj=1)
    if vce == 'multiway':
        options['clusters'].append(np.arange(n) // 20)
    if vce in ('hac', 'dk'):
        options.update(time=2*(np.arange(n)//7), panel=np.arange(n)%7, bandwidth=3)
    expected = (diag.first_stage_diagnostics(E, C, Z, **options),
                diag.sanderson_windmeijer_diagnostics(E, C, Z, **options))
    calls, original = [], diag.ols_vcov
    def counted(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)
    monkeypatch.setattr(diag, 'ols_vcov', counted)
    work = diag._IVDiagnosticWorkspace(E, C, Z, wi.estimation)
    actual = diag._first_stage_suite(E, C, Z, _workspace=work, **options)
    _assert_same(actual, expected)
    assert len(calls) == (1 if k == 1 else 2*k)
    # No inference object or covariance cache persists into another invocation.
    assert set(vars(work)) == {'E', 'C', 'Z', 'Ep', 'Zp', 'zz', '_batch_safe',
                              '_bread', '_shea', '_reduced_form'}


def test_joint_suite_cache_budget_changes_execution_not_results(monkeypatch):
    E, C, Z = _problem(k=5)
    expected = diag._first_stage_suite(E, C, Z, vce='robust')
    monkeypatch.setattr(diag, '_FIRST_STAGE_CACHE_BYTES', 0)
    original, calls = diag.ols_vcov, []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(diag, 'ols_vcov', counted)
    _assert_same(diag._first_stage_suite(E, C, Z, vce='robust'), expected)
    assert len(calls) == 3 * E.shape[1]


def test_joint_suite_does_not_reuse_inverses_after_inference_mutation():
    E, C, Z = _problem(k=3, n=280)
    work = diag._IVDiagnosticWorkspace(E, C, Z)
    clusters = [np.arange(len(E)) % 20]
    time = np.arange(len(E)) // 7
    configurations = [dict(vce='robust'), dict(vce='cluster', clusters=clusters),
                      dict(vce='cluster', clusters=clusters, df_absorbed=8, nested_adj=1),
                      dict(vce='hac', time=time, panel=np.arange(len(E)) % 7, bandwidth=2),
                      dict(vce='dk', time=time, bandwidth=3)]
    for options in configurations:
        got = diag._first_stage_suite(E, C, Z, _workspace=work, **options)
        expected = (diag.first_stage_diagnostics(E, C, Z, **options),
                    diag.sanderson_windmeijer_diagnostics(E, C, Z, **options))
        _assert_same(got, expected)
        # Preserve identity while changing contents: identity-keyed caching
        # would be wrong on the next applicable invocation.
        clusters[0][:] = (np.arange(len(E)) // 2) % 14
        time[:] *= 2


def test_joint_suite_exception_does_not_poison_retry(monkeypatch):
    E, C, Z = _problem(k=3)
    work = diag._IVDiagnosticWorkspace(E, C, Z)
    expected = diag._first_stage_suite(E, C, Z, vce='robust')
    original, calls = diag.ols_vcov, []
    def fail_second(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise np.linalg.LinAlgError('injected covariance failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(diag, 'ols_vcov', fail_second)
    with pytest.raises(np.linalg.LinAlgError, match='injected'):
        diag._first_stage_suite(E, C, Z, _workspace=work, vce='robust')
    monkeypatch.setattr(diag, 'ols_vcov', original)
    _assert_same(diag._first_stage_suite(E, C, Z, _workspace=work, vce='robust'), expected)


@pytest.mark.parametrize('case', ['duplicate_controls', 'duplicate_instruments', 'near_perfect'])
def test_joint_suite_does_not_cache_sensitive_scalar_residuals(case):
    E, C, Z = _problem(k=3)
    if case == 'duplicate_controls':
        C = np.column_stack([C, C[:, 0]])
    elif case == 'duplicate_instruments':
        Z = np.column_stack([Z, Z[:, 0]])
    else:
        E = Z[:, :3] + 1e-10 * E
    work = diag._IVDiagnosticWorkspace(E, C, Z)
    cache = [None] * E.shape[1]
    diag.first_stage_diagnostics(E, C, Z, _workspace=work, _inverse_cache=cache)
    assert cache == [None] * E.shape[1]
    expected = (diag.first_stage_diagnostics(E, C, Z),
                diag.sanderson_windmeijer_diagnostics(E, C, Z))
    _assert_same(diag._first_stage_suite(E, C, Z), expected)


def test_conditional_path_keeps_one_contrast_in_observation_space(monkeypatch):
    E, C, Z = _problem(k=9, n=320)
    work = diag._IVDiagnosticWorkspace(E, C, Z)
    original, shapes = diag._first_stage_statistics, []
    def capture(ep, *args, **kwargs):
        shapes.append((ep.shape, None if ep.base is None else ep.base.shape))
        return original(ep, *args, **kwargs)
    monkeypatch.setattr(diag, '_first_stage_statistics', capture)
    got = diag._conditional_from_reduced_form(
        work, vce='robust', clusters=None, df_absorbed=0, nested_adj=0,
        df1_override=Z.shape[1] - E.shape[1] + 1)
    assert got is not None
    assert shapes == [((len(E),), None)] * (2 * E.shape[1])


@pytest.mark.parametrize('noise_scale', [1, 30])
def test_first_stage_ap_sw_matches_exact_fraction_oracle(noise_scale):
    """Exact rational scalar regressions and HC1; no package covariance/solve.

    Small well-conditioned and weaker first stages check accuracy independently
    of either floating-point path. This is not a guarantee for arbitrary
    ill-conditioned or rank-boundary systems.
    """
    from fractions import Fraction as F
    rng = np.random.default_rng(982)
    n, k, q, dfabs = 24, 2, 3, 2
    c = np.column_stack([np.ones(n, dtype=int), rng.integers(-3, 4, n)])
    z = rng.integers(-4, 5, (n, q))
    e = z[:, :k] + noise_scale * rng.integers(-4, 5, (n, k))
    def rational(A):
        return [[F(int(x)) for x in row] for row in A]
    def transpose(A):
        return [list(row) for row in zip(*A)]
    def multiply(A, B):
        return [[sum(x*y for x, y in zip(row, col)) for col in zip(*B)] for row in A]
    def difference(A, B):
        return [[x-y for x,y in zip(a,b)] for a,b in zip(A,B)]
    def inverse(A):
        size = len(A)
        rows = [list(row)+[F(i==j) for j in range(size)] for i,row in enumerate(A)]
        for j in range(size):
            pivot = next(i for i in range(j, size) if rows[i][j])
            rows[j], rows[pivot] = rows[pivot], rows[j]
            scale = rows[j][j]
            rows[j] = [x/scale for x in rows[j]]
            for i in range(size):
                if i != j:
                    scale = rows[i][j]
                    rows[i] = [a-scale*b for a,b in zip(rows[i],rows[j])]
        return [row[size:] for row in rows]
    def least_squares(A, B):
        At = transpose(A)
        return multiply(inverse(multiply(At,A)), multiply(At,B))
    def partial(A, B):
        return difference(A, multiply(B,least_squares(B,A)))
    C, Z, E = map(rational, (c,z,e))
    Zp = partial(Z,C)
    bread = inverse(multiply(transpose(Zp),Zp))
    def stage(dep, df1):
        ep = partial(dep,C)
        beta = least_squares(Zp,ep)
        resid = difference(ep,multiply(Zp,beta))
        tss = sum(row[0]**2 for row in ep)
        rss = sum(row[0]**2 for row in resid)
        scores = [[x*r[0] for x in row] for row,r in zip(Zp,resid)]
        meat = multiply(transpose(scores),scores)
        scale = F(n,n-len(C[0])-q-dfabs)
        V = multiply(multiply(bread,meat),bread)
        V = [[x*scale for x in row] for row in V]
        wald = multiply(multiply(transpose(beta),inverse(V)),beta)[0][0]
        classic = (tss-rss)/df1/(rss/F(n-len(C[0])-q-dfabs))
        return {'partial_r2':float(1-rss/tss), 'f_classic':float(classic),
                'f_robust':float(wald/df1)}
    Q = [a+b for a,b in zip(C,Z)]
    Ehat = multiply(Q,least_squares(Q,E))
    X, Xhat = [a+b for a,b in zip(C,E)], [a+b for a,b in zip(C,Ehat)]
    actual_first, actual_conditional = diag._first_stage_suite(
        e, c, z, vce='robust', df_absorbed=dfabs)
    for j in range(k):
        dep = [[row[j]] for row in E]
        target = len(C[0])+j
        Xm = [[x for h,x in enumerate(row) if h!=target] for row in X]
        Xhm = [[x for h,x in enumerate(row) if h!=target] for row in Xhat]
        b = least_squares(Xhm,dep)
        ap, sw = difference(dep,multiply(Xhm,b)), difference(dep,multiply(Xm,b))
        for got, expected in ((actual_first[j],stage(dep,q)),
                              (actual_conditional[j]['ap'],stage(ap,q-k+1)),
                              (actual_conditional[j]['sw'],stage(sw,q-k+1))):
            for key in expected:
                np.testing.assert_allclose(got[key],expected[key],rtol=5e-11,atol=5e-12)
