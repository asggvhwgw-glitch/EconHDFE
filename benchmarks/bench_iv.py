"""Synthetic IV timings and a separate diagnostic-stage profile for one revision."""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
import platform
import pstats
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info
from econhdfe import ivhdfe, factor, reg_interaction, InferenceConfig
from econhdfe.design import build_design


def fixture(n, groups):
    n = n // groups * groups
    rng = np.random.default_rng(20260913)
    g = np.repeat(np.arange(groups), n // groups)
    df = pd.DataFrame({'g': g})
    # Match the IV draw order in heterogeneous_spec_models.py (8 local slopes).
    for _ in range(8):
        rng.normal(size=n)
    for j in range(2):
        df[f'c{j}'] = rng.normal(size=n)
    rng.normal(size=n)  # OLS disturbance in the original benchmark.
    z = rng.normal(size=n)
    e = .8 * z + .15 * df.c0.to_numpy() + rng.normal(scale=.55, size=n)
    df['z'], df['e'] = z, e
    df['y'] = np.linspace(.15, .45, groups)[g] * e + .1 * df.c0.to_numpy() + rng.normal(size=n)
    df['cl'] = np.arange(n) % max(20, min(1200, n // 20))
    return df


def numeric_results(result):
    arrays = {name: np.asarray(getattr(result, name)) for name in
              ('params', 'vcov', 'stderr', 'residuals', 'nobs', 'nobs_raw', 'rank',
               'df_absorbed', 'df_resid', 'df_model', 'df_resid_fit', 'vcov_rank')}
    arrays['first_coefficients'] = result.first_stage['coefficients']
    arrays['first_fitted'] = result.first_stage['fitted_endog']

    def flatten(value, key):
        if isinstance(value, dict):
            for k, v in value.items():
                flatten(v, key + '/' + str(k))
        elif isinstance(value, (tuple, list)):
            for j, v in enumerate(value):
                flatten(v, key + '/' + str(j))
        elif isinstance(value, (float, int, np.number)):
            arrays[key] = np.asarray(value)
    flatten(result.first_stage['diagnostics'], 'first_diagnostics')
    flatten(result.diagnostics, 'diagnostics')
    return arrays


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n', type=int, default=96000)
    p.add_argument('--groups', type=int, default=12)
    p.add_argument('--reps', type=int, default=5)
    p.add_argument('--representation', choices=['structured', 'dense'], default='structured')
    p.add_argument('--vce', choices=['robust', 'cluster'], default='robust')
    p.add_argument('--estimator', choices=['2sls', 'liml', 'gmm2s'], default='2sls')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--save-numerics', action='store_true')
    a = p.parse_args()
    if min(a.n, a.groups, a.reps) < 1 or a.n < a.groups:
        p.error('positive dimensions/repetitions and n >= groups required')
    df = fixture(a.n, a.groups)
    C = ['c0', 'c1']
    E = [reg_interaction(factor('g', drop_base=False), 'e', name='ge')]
    Z = [reg_interaction(factor('g', drop_base=False), 'z', name='gz')]

    def fit():
        c, e, z = C, E, Z
        if a.representation == 'dense':
            c, e, z = (build_design(df, s, len(df), structural=False).values for s in (C, E, Z))
        return ivhdfe(
            df, y='y', exog=c, endog=e, instruments=z, absorb=['g'], vce=a.vce,
            cluster='cl' if a.vce == 'cluster' else None, collinearity='drop',
            estimator=a.estimator, inference_config=InferenceConfig(diagnostics='off'),
        )

    t = time.perf_counter()
    result = fit()
    cold = time.perf_counter() - t
    elapsed = []
    for _ in range(a.reps):
        t = time.perf_counter()
        result = fit()
        elapsed.append(time.perf_counter() - t)
    profile = cProfile.Profile()
    profile.runcall(fit)
    stats = pstats.Stats(profile)
    names = {'first_stage_diagnostics', 'sanderson_windmeijer_diagnostics',
             'kleibergen_paap_stats', 'cragg_donald_stat', 'fit_iv_kclass',
             'fit_iv_kclass_block', '_first_stage_test', '_first_stage_statistics',
             '_residualize_small', 'materialize', '_precisions', '_precision_cache',
             '_conditional_from_reduced_form', 'ols_vcov', 'sandwich_vcov_xe'}
    stages, lstsq = [], []
    for (path, line, name), (_, calls, self_s, total_s, _) in stats.stats.items():
        row = dict(file=Path(path).name, line=line, function=name, calls=calls,
                   self_seconds=self_s, cumulative_seconds=total_s)
        if name in names or (Path(path).name == 'diagnostics.py' and name in {'__init__', 'tests'}):
            stages.append(row)
        if name == 'lstsq' and 'scipy' in path:
            lstsq.append(row)
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss = rss / (1024 * 1024 if sys.platform == 'darwin' else 1024)
    except ImportError:
        peak_rss = None  # Windows: no extra dependency for process profiling.
    import scipy
    import numba
    import econhdfe
    source = Path(econhdfe.__file__).resolve().parent
    hashes = {name: hashlib.sha256((source / 'models' / 'linear_iv' / name).read_bytes()).hexdigest()
              for name in ('diagnostics.py', 'estimators.py')}
    data = dict(
        nobs=len(df), groups=a.groups, representation=a.representation, vce=a.vce,
        estimator=a.estimator, repetitions=a.reps, cold_seconds=cold,
        warm_seconds=elapsed, median_seconds=float(np.median(elapsed)),
        profile_total_seconds=stats.total_tt, stages=stages, lstsq=lstsq,
        peak_rss_mib=peak_rss, source_sha256=hashes,
        structured_path=bool(result.diagnostics.get('heterogeneous_spec_path')),
        diagnostics_mode=result.diagnostics_mode,
        environment=dict(
            python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
            numba=numba.__version__, econhdfe=econhdfe.__version__,
            threads={k: os.environ.get(k) for k in
                     ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMBA_NUM_THREADS')},
            blas=[{k: v for k, v in item.items() if k != 'filepath'} for item in threadpool_info()],
        ),
        notes=[
            'Synthetic development benchmark, not external-package parity.',
            'Cumulative stage times overlap; use unprofiled medians for comparisons.',
            'Peak RSS covers the process including warmup and the separate profile run.',
            'diagnostics=off still computes all existing diagnostics on both revisions.',
        ],
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(data, indent=2) + '\n')
    if a.save_numerics:
        np.savez_compressed(a.output.with_suffix('.npz'), **numeric_results(result))
    print(json.dumps({k: data[k] for k in ('nobs', 'groups', 'representation', 'vce',
                                         'median_seconds', 'peak_rss_mib', 'lstsq')}, indent=2))


if __name__ == '__main__':
    main()
