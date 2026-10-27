"""Matched end-to-end PPML/IV-PPML profile, including changing FE weights.

No timing claims from non-converged projections. Inputs and output hashes,
all repetitions, parity checks, and non-overlapping component times are saved.
Run from an installed source checkout. Optional within wheel is required.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
from importlib.machinery import EXTENSION_SUFFIXES
import json
import os
from pathlib import Path
import platform
try:
    import resource
except ImportError:  # Windows has no getrusage.
    resource = None
import time

import numpy as np
from threadpoolctl import threadpool_limits

from econhdfe import ExecutionConfig, PPMLConfig, IVPPMLConfig, ppmlhdfe, ivppmlhdfe
from econhdfe.hdfe.absorber import HDFEAbsorber
from econhdfe.hdfe.two_way import TwoWayFEAbsorber
from econhdfe.hdfe.weighted_projection import WeightedFEProjector
from econhdfe.hdfe.within_backend import WithinFEAbsorber


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).view(np.uint8)).hexdigest()



def native_identity(native):
    """Hash the loaded extension, not a maturin package's Python __init__.py.

    Both a directly imported extension and maturin's package/submodule layout
    are supported. Fail rather than label an unverified Python file as a binary.
    """
    entrypoint = Path(native.__file__).resolve()
    binary_module = getattr(native, "_econhdfe_within", native)
    binary = Path(binary_module.__file__).resolve()
    if not any(binary.name.endswith(suffix) for suffix in EXTENSION_SUFFIXES):
        raise RuntimeError("Cannot identify the loaded native within extension binary")
    return dict(
        native_identity_schema=2,
        native_module_file=str(binary),
        native_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        native_import_entrypoint_file=str(entrypoint),
        native_import_entrypoint_sha256=hashlib.sha256(entrypoint.read_bytes()).hexdigest(),
    )


def fixture(n, case, seed):
    rng = np.random.default_rng(seed)
    if case == 'repeated':
        m = (n + 7) // 8
        groups = [np.repeat(rng.integers(L, size=m), 8)[:n] for L in (12000, 2000, 48)]
    elif case == 'weak':
        # 40 communities, 98.5% local links; no redundant FE partitions.
        communities = 40
        c = rng.integers(communities, size=n)
        g1 = c * 250 + rng.integers(250, size=n)
        g2 = c * 50 + rng.integers(50, size=n)
        bridge = rng.random(n) < .015
        g2[bridge] = rng.integers(2000, size=np.sum(bridge))
        g3 = c * 2 + rng.integers(2, size=n)
        bridge = rng.random(n) < .015
        g3[bridge] = rng.integers(80, size=np.sum(bridge))
        groups = [g1, g2, g3]
    elif case == 'schur':
        groups = [rng.integers(L, size=n) for L in (20000, 32, 8)]
    elif case == 'crossed':
        groups = [rng.integers(L, size=n) for L in (20000, 4000, 48)]
    else:
        raise ValueError(case)
    # Both kernels require dense labels; formal array APIs also compile FEPlan.
    groups = [np.unique(g, return_inverse=True)[1].astype(np.int32) for g in groups]
    c = rng.normal(size=n)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    e = .8*z1 + .3*z2 + .25*c + rng.normal(scale=.5, size=n)
    off = .15*rng.normal(size=n)
    weights = np.exp(np.clip(.6*rng.normal(size=n), -4., 4.))
    eta = .4 + .2*c + .18*e + off
    for g in groups:
        eta += rng.normal(scale=.15, size=int(g.max()) + 1)[g]
    y = rng.poisson(np.exp(eta)).astype(float)
    return dict(y=y, c=c, e=e, instruments=np.column_stack([z1, z2]),
                offset=off, weights=weights, groups=groups)


@contextmanager
def profile():
    records = []
    changes = []
    def install(cls, name, wrap):
        old = getattr(cls, name)
        setattr(cls, name, wrap(old))
        changes.append((cls, name, old))
    def preparation(old):
        def wrapped(self, weights, *, tol):
            cold = not self.compiled
            t = time.perf_counter()
            out = old(self, weights, tol=tol)
            records.append(dict(part='prepare', seconds=time.perf_counter()-t,
                                cold=cold, nobs=len(weights), tol=float(tol),
                                backend=self._active_kind,
                                route=getattr(getattr(out, '_native', None), 'route', None)))
            return out
        return wrapped
    def projection(old):
        def wrapped(self, data, *args, **kwargs):
            t = time.perf_counter()
            out = old(self, data, *args, **kwargs)
            elapsed = time.perf_counter()-t
            info = out[1] if kwargs.get('return_info') else None
            if info is not None and (not info.converged or not np.isfinite(info.max_update)):
                raise RuntimeError(f'Uncertified projection: {type(self).__name__}: {info}')
            records.append(dict(part='solve', seconds=elapsed,
                                nobs=int(np.shape(data)[0]),
                                rhs=1 if np.ndim(data)==1 else int(np.shape(data)[1]),
                                backend=type(self).__name__, tol=float(self.tol),
                                iterations=None if info is None else int(info.iterations),
                                error=None if info is None else float(info.max_update)))
            return out
        return wrapped
    install(WeightedFEProjector, 'prepare', preparation)
    for cls in (HDFEAbsorber, TwoWayFEAbsorber, WithinFEAbsorber):
        install(cls, 'residualize', projection)
    try:
        yield records
    finally:
        for cls, name, old in reversed(changes):
            setattr(cls, name, old)


def fit(d, model, engine, budget):
    ex = ExecutionConfig(threads=1, memory_budget_mb=budget)
    common = dict(y=d['y'], absorb=d['groups'], weights=d['weights'],
                  offset=d['offset'], vce='robust', execution_config=ex)
    # Only FE separation is timed; simplex/ReLU correctness is tested separately.
    conf = dict(engine=engine, separation=('fe',), tolerance=1e-9,
                target_inner_tol=1e-10, max_iter=200)
    if model == 'ppml':
        return ppmlhdfe(x=np.column_stack([d['c'], d['e']]),
                        config=PPMLConfig(**conf), **common)
    return ivppmlhdfe(exog=d['c'], endog=d['e'], instruments=d['instruments'],
                     weight_type='pweight', config=IVPPMLConfig(**conf), **common)


def scores(d, r, model):
    keep = r.sample_mask
    u = d['weights'][keep] * (d['y'][keep] - r.mu[keep])
    Q = np.column_stack([d['c'], d['e']]) if model == 'ppml' else np.column_stack([d['c'], d['instruments']])
    Q = Q[keep]
    score = Q.T @ u / len(u)
    fe = 0.
    for g0 in d['groups']:
        g = g0[keep]
        mass = np.bincount(g, weights=d['weights'][keep]*r.mu[keep])
        s = np.bincount(g, weights=u)
        fe = max(fe, float(np.max(np.abs(s)/np.maximum(mass, 1e-300))))
    return dict(moment_max=float(np.max(np.abs(score))), relative_fe_score=fe)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rows', type=int, default=1000000)
    ap.add_argument('--case', choices=('crossed','repeated','weak','schur'), required=True)
    ap.add_argument('--model', choices=('ppml','ivppml'), required=True)
    ap.add_argument('--repeat', type=int, default=3)
    ap.add_argument('--seed', type=int, default=20261027)
    ap.add_argument('--budget-mb', type=int, default=2048)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    if a.rows < 2000 or a.repeat < 1 or a.budget_mb < 1:
        ap.error("rows >= 2000, repeat >= 1 and budget-mb >= 1 are required")
    import scipy, numba, _econhdfe_within as native
    record = dict(rows=a.rows, case=a.case, model=a.model, seed=a.seed,
                  repeat=a.repeat, threads=1, budget_mb=a.budget_mb,
                  platform=platform.platform(), python=platform.python_version(),
                  numpy=np.__version__, scipy=scipy.__version__, numba=numba.__version__,
                  upstream_rev=native.UPSTREAM_REV,
                  **native_identity(native),
                  outer_tolerance=1e-9, target_inner_tolerance=1e-10,
                  separation=['fe'], vce='robust',
                  source_ref=os.environ.get('ECONHDFE_SOURCE_REF', os.environ.get('GITHUB_SHA')),
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  timings=[], parity=[], status='running')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        a.output.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    save()
    try:
        with threadpool_limits(limits=1):
            # Exclude JIT/runtime initialization by warming both engines and model.
            small = fixture(4000, 'crossed', a.seed+1)
            small['groups'] = [np.arange(4000, dtype=np.int32)%L for L in (40,25,7)]
            for engine in ('optimized','within'):
                fit(small, a.model, engine, a.budget_mb)
            d = fixture(a.rows, a.case, a.seed)
            record['input_hashes'] = {k: digest(v) for k,v in d.items() if k != 'groups'}
            record['group_hashes'] = [digest(g) for g in d['groups']]
            record['fe_levels'] = [int(g.max())+1 for g in d['groups']]
            for rep in range(a.repeat):
                outputs = {}
                for engine in (('optimized','within') if rep%2==0 else ('within','optimized')):
                    with profile() as prof:
                        start = time.perf_counter()
                        r = fit(d, a.model, engine, a.budget_mb)
                        elapsed = time.perf_counter()-start
                    if not r.converged:
                        raise RuntimeError('Nonconverged outer fit')
                    total_prepare = sum(x['seconds'] for x in prof if x['part']=='prepare')
                    build = sum(x['seconds'] for x in prof if x['part']=='prepare' and x['cold'])
                    solve = sum(x['seconds'] for x in prof if x['part']=='solve')
                    t = dict(repeat=rep, engine=engine, total_seconds=elapsed,
                             initial_prepare_seconds=build, weight_update_seconds=total_prepare-build,
                             solve_seconds=solve, other_seconds=elapsed-total_prepare-solve,
                             iterations=r.iterations, nobs=int(r.sample_mask.sum()),
                             params=r.params.tolist(), stderr=r.stderr.tolist(),
                             deviance=float(r.deviance), loglike=float(r.loglike),
                             scores=scores(d,r,a.model), calls=prof,
                             peak_rss_kib=(None if resource is None else int(
                                 resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                                 / (1024 if platform.system() == 'Darwin' else 1))))
                    record['timings'].append(t)
                    outputs[engine] = r
                    print(json.dumps({k:v for k,v in t.items() if k!='calls'}), flush=True)
                    save()
                b, r = outputs['optimized'], outputs['within']
                assert np.array_equal(b.sample_mask, r.sample_mask)
                assert np.array_equal(b.separation_mask, r.separation_mask)
                assert b.names == r.names and b.df_absorbed == r.df_absorbed
                np.testing.assert_allclose(r.params, b.params, atol=2e-7, rtol=2e-7)
                np.testing.assert_allclose(r.vcov, b.vcov, atol=1e-10, rtol=2e-6)
                np.testing.assert_allclose(r.mu, b.mu, atol=2e-6, rtol=2e-6)
                parity = dict(repeat=rep, coef_max=float(np.max(np.abs(r.params-b.params))),
                              vcov_max=float(np.max(np.abs(r.vcov-b.vcov))),
                              mu_max=float(np.nanmax(np.abs(r.mu-b.mu))),
                              deviance_diff=float(abs(r.deviance-b.deviance)))
                record['parity'].append(parity)
                del outputs, b, r
                save()
    except Exception as exc:
        record['status'] = 'failed'
        record['error'] = repr(exc)
        save()
        raise
    record['status'] = 'passed'
    save()


if __name__ == '__main__':
    main()
