"""Million-record sample-peeling checks, not an estimator/solver speed comparison.

Inputs are generated, not user data. Timings include the validated peeling API
and its allocations, excluding generation/JIT. Known fixed points are the
independent oracle; small numerical tests separately check estimator inference.
"""
from __future__ import annotations
import argparse
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import numba
from econhdfe.hdfe.sample import peel_sample
from econhdfe.hdfe.plan import FEPlan
from econhdfe.models.ppml.config import PPMLConfig
from econhdfe.models.ppml.separation import initial_sample_selection


def run(n=1_000_000):
    if n < 1_000_000 or n % 100:
        raise ValueError('use >=1,000,000 records, divisible by 100')
    warm = [np.array([0,0,1,1]), np.array([0,1,1,2])]
    peel_sample(warm, positive=np.array([True,False,True,False]))
    records = []
    i = np.arange(n)
    chain = [i//2, (i+1)//2, i%1000, i%53]
    for name, fw, expected in [('chain_unit', None, 0),
                               ('chain_frequency_protected', np.ones(n), n)]:
        if fw is not None:
            fw[[0,-1]] = 2
        start = perf_counter()
        result = peel_sample(chain, frequency_weights=fw)
        elapsed = perf_counter()-start
        kept = int(result.keep.sum())
        assert kept == expected
        records.append(dict(case=name, rows=n, fe_levels=[int(a.max())+1 for a in chain],
                            expected_kept=expected, kept=kept, seconds=elapsed, passed=True))
    core = 9*n//10
    tail = n-core
    rng = np.random.default_rng(20261027)
    g = np.r_[np.repeat(np.arange(core//10),10), np.arange(core//10,core//10+tail)]
    groups = [g, i%40, rng.integers(0,5000,n), rng.integers(0,200,n)]
    fw = rng.integers(1,4,n).astype(float)
    fw[core:] = np.tile([1.,5.],tail//2)
    y = np.ones(n)
    zeros = tail//20
    y[-zeros:] = 0
    plan = FEPlan.from_arrays(groups)
    expected = np.ones(n,dtype=bool)
    expected[core:] = fw[core:]>1
    expected[-zeros:] = False
    for mode in ('stata_compat','unified'):
        start = perf_counter()
        result = initial_sample_selection(y,plan,PPMLConfig(singleton_mode=mode),
                                          weights=fw,weight_kind='fweight')
        elapsed = perf_counter()-start
        target = expected if mode=='unified' else i<core
        assert np.array_equal(~result.dropped,target)
        records.append(dict(case='four_fe_'+mode, rows=n,
                            fe_levels=[int(a.max())+1 for a in groups],
                            expected_kept=int(target.sum()), kept=int((~result.dropped).sum()),
                            singletons=int(result.singletons.sum()), separated=int(result.separated.sum()),
                            seconds=elapsed, passed=True))
    return dict(passed=True, scope='sample-peeling only; single-run timings, no relative speed claim',
                python=platform.python_version(), platform=platform.platform(),
                numpy=np.__version__, numba=numba.__version__, records=records)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=1_000_000)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.n)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
