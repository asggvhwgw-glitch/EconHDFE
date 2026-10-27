"""Warm coefficient-bootstrap comparison with the preserved per-batch route.

Run from a checkout with NUMBA_NUM_THREADS=1 and BLAS/OMP/MKL threads=1:
  python benchmarks/resampling/wild_bootstrap_scores.py --output results.json

The delegating projector selects the public function's unchanged legacy path;
the candidate is the production function, including design projection, memory
admission, scores, RNG and draws. Numerical oracles live in tests/numerics.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numba
import numpy as np
import scipy
from threadpoolctl import threadpool_limits

from econhdfe import ExecutionConfig
from econhdfe.hdfe.absorber import HDFEAbsorber
from econhdfe.models.ols_bootstrap import wild_cluster_bootstrap_ols


class LegacyProjector:
    """Delegate the same solve while selecting the custom-projector fallback."""

    def __init__(self, absorber):
        self.absorber = absorber

    def __getattr__(self, name):
        return getattr(self.absorber, name)


def run_case(n, weighted, reps, rounds):
    rng = np.random.default_rng(20261027)
    levels = (20_000, 4_000, 50) if n >= 1_000_000 else (80, 40, 10)
    groups = [rng.integers(L, size=n, dtype=np.int32) for L in levels]
    cluster = rng.integers(1000 if n >= 1_000_000 else 50, size=n, dtype=np.int32)
    w = np.exp(rng.normal(0, .5, n)) if weighted else None
    X = rng.normal(size=(n, 3))
    X[:, 0] += rng.normal(size=levels[0])[groups[0]]
    X[:, 1] += .3*rng.normal(size=levels[1])[groups[1]]
    y = X @ np.array([.7, -.4, .2]) + rng.normal(size=n)
    for g, L in zip(groups, levels, strict=True):
        y += rng.normal(size=L)[g]
    start = time.perf_counter()
    absorber = HDFEAbsorber(
        groups, weights=w, tol=1e-11, max_iter=2000, method="map",
        acceleration="cg", absorb_threads=1, projection_memory_budget_mb=2048,
    )
    transformed, info = absorber.residualize(np.column_stack([y, X]), return_info=True)
    assert info.converged, info
    yt, Xt = transformed[:, 0], transformed[:, 1:]
    sw = np.ones(n) if w is None else np.sqrt(w)
    beta = np.linalg.lstsq(Xt*sw[:, None], yt*sw, rcond=None)[0]
    u = yt-Xt @ beta
    setup_seconds = time.perf_counter()-start
    calls = []
    original = absorber.residualize

    def tracked(data, **kwargs):
        want_info = kwargs.pop("return_info", False)
        out, inf = original(data, return_info=True, **kwargs)
        calls.append(dict(converged=bool(inf.converged), iterations=int(inf.iterations),
                          metric=float(inf.max_update)))
        return (out, inf) if want_info else out

    absorber.residualize = tracked
    kwargs = dict(y_within=yt, X_within=Xt, beta=beta, residuals=u,
                  clusters=cluster, weights=w, reps=reps, batch_size=8,
                  seed=73451, n_jobs=1,
                  execution_config=ExecutionConfig(threads=1, memory_budget_mb=2048,
                                                   memory_policy="relaxed"))
    times = dict(legacy=[], scores=[])
    diagnostics = dict(legacy=[], scores=[])
    differences = []
    for repeat in range(rounds):
        outputs = {}
        for kind in ("legacy", "scores") if repeat % 2 == 0 else ("scores", "legacy"):
            gc.collect()
            calls.clear()
            start = time.perf_counter()
            outputs[kind] = wild_cluster_bootstrap_ols(
                absorber=LegacyProjector(absorber) if kind == "legacy" else absorber,
                **kwargs,
            )
            times[kind].append(time.perf_counter()-start)
            assert all(c["converged"] for c in calls), calls
            diagnostics[kind].append(dict(calls=len(calls),
                                           all_converged=True,
                                           iterations=[c["iterations"] for c in calls]))
        difference = outputs["scores"]-outputs["legacy"]
        differences.append(dict(max_abs=float(np.max(np.abs(difference))),
                                relative_l2=float(np.linalg.norm(difference) /
                                                  np.linalg.norm(outputs["legacy"]))))
        np.testing.assert_allclose(outputs["scores"], outputs["legacy"], rtol=1e-9, atol=1e-10)
        print(json.dumps(dict(rows=n, weighted=weighted, repeat=repeat,
                              seconds={k: v[-1] for k, v in times.items()},
                              difference=differences[-1])), flush=True)
    medians = {k: float(np.median(v)) for k, v in times.items()}
    return dict(rows=n, fe_levels=levels, controls=3, clusters=int(cluster.max())+1,
                weight_type="lognormal, log SD 0.5" if weighted else "unit",
                reps=reps, batch_size=8, threads=1, tol=1e-11,
                fit_setup_seconds_excluded=setup_seconds,
                timings_seconds=times, median_seconds=medians,
                speedup=medians["legacy"]/medians["scores"],
                differences=differences, absorption=diagnostics,
                cumulative_process_peak_rss_mib=(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss /
                                                 (1024**2 if sys.platform == "darwin" else 1024)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--reps", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        for weighted in (False, True):
            run_case(4000, weighted, 8, 1)
        result = dict(
            base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (ROOT/"econhdfe/models/ols_bootstrap.py", ROOT/"econhdfe/resampling/engine.py")},
            environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__,
                             numba=numba.__version__, platform=platform.platform()),
            scope="Warm coefficient draws only; initial fit, imports and JIT excluded. Production score setup included. Same sample, weights, seed batches, fit and absorber. No studentized inference or end-to-end estimator speedup claim.",
            cases=[run_case(args.rows, weighted, args.reps, args.rounds) for weighted in (False, True)],
        )
    args.output.write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
