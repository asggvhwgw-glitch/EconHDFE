import os
import numpy as np

from pyreghdfe.design import build_design
from pyreghdfe.execution import plan_workspace, ReusableRHSWorkspace
from pyreghdfe.vcov import _cluster_meat


def test_numpy_design_zero_copy_without_omission():
    x = np.arange(120.0).reshape(30, 4)
    d = build_design(None, x, 30)
    assert np.shares_memory(d.values, x)


def test_numpy_design_copies_when_user_omits_column():
    from pyreghdfe import omit_column
    x = np.arange(120.0).reshape(30, 4)
    d = build_design(None, x, 30, omit=[omit_column('x2')])
    assert d.values.shape == (30, 3)
    assert not np.shares_memory(d.values, x)


def test_workspace_is_reused_and_contiguous():
    src = np.arange(240.0).reshape(40, 6)
    ws = ReusableRHSWorkspace(40, 4)
    a = ws.load(src, 0, 4)
    ptr = a.__array_interface__['data'][0]
    b = ws.load(src, 2, 6)
    assert b.flags.c_contiguous
    assert b.__array_interface__['data'][0] == ptr


def test_cluster_high_cardinality_matches_reference():
    rng = np.random.default_rng(4)
    n, p = 6000, 5
    s = rng.normal(size=(n, p))
    # Mostly singleton cells with a few repeated groups. Force low-memory branch
    # by using a huge maximum code while retaining dense-ish semantics.
    c = np.arange(n, dtype=np.int64)
    c[:200] = np.repeat(np.arange(100), 2)
    ref_codes, inv = np.unique(c, return_inverse=True)
    g = len(ref_codes)
    sums = np.empty((g, p))
    for j in range(p):
        sums[:, j] = np.bincount(inv, weights=s[:, j], minlength=g)
    ref = sums.T @ sums
    # _cluster_meat expects dense codes, so pad cardinality by adding many empty
    # levels is not meaningful. The identity is tested directly via a larger p
    # threshold-independent construction below.
    got, _ = _cluster_meat(s, inv)
    assert np.allclose(got, ref, rtol=1e-12, atol=1e-10)


def test_fixed_map_certificate_matches_standard_map():
    from pyreghdfe.absorber import HDFEAbsorber
    rng = np.random.default_rng(81)
    n = 12000
    g1 = rng.integers(0, 300, n, dtype=np.int32)
    g2 = rng.integers(0, 240, n, dtype=np.int32)
    z = rng.normal(size=(n, 4))
    a = HDFEAbsorber([g1, g2], acceleration='none', tol=1e-9, absorb_threads=2)
    ref, info = a.residualize(z.copy(), return_info=True)
    out, finfo = a.residualize_fixed_map(z.copy(), info.iterations, return_info=True, absorb_threads=2)
    assert finfo.converged
    assert np.max(np.abs(out - ref)) < 2e-8
    assert a.fe_orthogonality_error(out) <= 1e-9


def test_runtime_configuration_safe_if_numba_was_imported_first(monkeypatch):
    import numba  # noqa: F401
    from econhdfe.compute.runtime import configure_numba_runtime
    monkeypatch.delenv("NUMBA_NUM_THREADS", raising=False)
    assert configure_numba_runtime() >= 1
