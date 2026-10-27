"""New, reproducible native-within validation; NOT the historical composite 5M.

Every FE tuple occurs in pairs with equal positive weights.  The two rows have
opposite dyadic within signals and a common additive FE part.  All additions
are exactly representable in float64, so the stated within signals are the
projection of the *stored inputs*, not merely an unrounded DGP approximation.

The three FEs form weakly linked communities.  Controls share a large FE part
and differ by 2**-12 times a signal; a small excluded-instrument signal is
nearly absorbed.  The full-IV reference uses QR/SVD of known within data and
independent cluster-score aggregation.  It imports no EconHDFE solver helpers.

Run each backend/process separately, initially setting BLAS threads to one.
The public estimator manages its own configured thread budget.  Example:
  PYTHONPATH=<native-wheel-dir>:<deps>:<checkout> OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=3 python validate_native_dyadic_iv.py \
    --rows 1000000 --mode both --threads 2 --output result.json

The acceptance limits below are fixed before large-sample runs.  This is a
numerical validation corpus, not a claim about arbitrary real-data designs or
the historical 5M fixture whose generator has not been recovered.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import time
import traceback

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


CORPUS = "dyadic_paired_weak_community_iv_v1"
SEED = 20261027
DELTA = 2.0**-12
INSTRUMENT_SCALE = 2.0**-8
TOL = 1e-12
MAX_ITER = 16000
LIMITS = dict(projection_weighted_relative=1e-8,
              coefficient_scaled=1e-5, covariance_correlation_scaled=5e-4,
              stderr_relative=1e-4, residual_weighted_relative=1e-6,
              rss_relative=1e-6)
NAMES = ("y", "c1", "c2", "e", "z1", "z2")


@dataclass
class Fixture:
    raw: np.ndarray
    codes: np.ndarray
    weights: np.ndarray
    pair_codes: np.ndarray
    pair_weights: np.ndarray
    pair_within: np.ndarray
    row_pair: np.ndarray
    row_sign: np.ndarray
    levels: tuple[int, ...]
    metadata: dict


def make_fixture(n: int, seed: int = SEED) -> Fixture:
    if n < 8192 or n % 8:
        raise ValueError("rows must be >=8192 and divisible by eight")
    rng = np.random.default_rng(seed)
    q = n // 2
    communities = 32
    workers = np.arange(q, dtype=np.uint32) // 4
    community = workers % communities
    g2 = 32 * community + rng.integers(32, size=q, dtype=np.uint32)
    g3 = 8 * community + rng.integers(8, size=q, dtype=np.uint32)
    bridge2 = rng.random(q) < 1 / 64
    bridge3 = rng.random(q) < 1 / 96
    # Guarantee links around the whole community ring independently of RNG.
    bridge2[4 * np.arange(communities)] = True
    bridge3[4 * np.arange(communities) + 1] = True
    g2[bridge2] = 32 * ((community[bridge2] + 1) % communities) + g2[bridge2] % 32
    g3[bridge3] = 8 * ((community[bridge3] + communities - 1) % communities) + g3[bridge3] % 8
    pair_codes = np.column_stack((workers, g2, g3))
    pair_codes = np.column_stack([
        np.unique(g, return_inverse=True)[1] for g in pair_codes.T
    ]).astype(np.uint32)
    levels = tuple(int(g.max()) + 1 for g in pair_codes.T)
    pair_weights = np.exp2(2 * rng.integers(-2, 3, size=q)).astype(float)
    pair_weights[bridge2 | bridge3] *= 2.0**-6

    # Integer signals on a fixed binary grid keep every generated input exact.
    signal = rng.integers(-1024, 1025, size=(q, 6)).astype(float) * 2.0**-10
    e = .75 * signal[:, 2] + .25 * signal[:, 3] + .5 * signal[:, 4]
    c1 = signal[:, 0]
    c2 = c1 + DELTA * signal[:, 1]
    y = .75 * c1 - .25 * c2 + 1.5 * e + .25 * signal[:, 4] + .5 * signal[:, 5]
    pair_within = np.column_stack((y, c1, c2, e, signal[:, 2],
                                  INSTRUMENT_SCALE * signal[:, 3]))
    del signal, e, c1, c2, y
    fe_part = np.zeros((q, 6))
    for g, nlev in zip(pair_codes.T, levels, strict=True):
        load = rng.integers(-256, 257, size=(nlev, 6)).astype(float)
        load[:, 2] = load[:, 1]  # hidden weak control contrast before absorption
        load[:, 5] *= 2.0**-4
        fe_part += load[g]
    # Values are on a 2**-24 grid and |raw| < 2**10.  Thus each addition
    # uses fewer than 35 significant binary digits, safely inside float64's 53.
    assert np.max(np.abs(fe_part)) + np.max(np.abs(pair_within)) < 2.0**10
    assert np.array_equal(pair_within * 2.0**24,
                          np.rint(pair_within * 2.0**24))
    order = rng.permutation(n)
    row_pair = order // 2
    row_sign = (1 - 2 * (order % 2)).astype(np.int8)
    del order
    raw = np.empty((n, 6), order="F")
    for j in range(6):
        raw[:, j] = fe_part[row_pair, j] + row_sign * pair_within[row_pair, j]
        # Exact stored-float decomposition, checked on all observations.
        assert np.array_equal(raw[:, j] - fe_part[row_pair, j],
                              row_sign * pair_within[row_pair, j])
    del fe_part
    codes = np.asfortranarray(pair_codes[row_pair])
    weights = pair_weights[row_pair]
    metadata = dict(corpus=CORPUS, seed=seed, rows=n, rhs=6,
                    communities=communities, levels=list(levels),
                    bridge_pairs=int(np.count_nonzero(bridge2 | bridge3)),
                    weight_min=float(weights.min()), weight_max=float(weights.max()),
                    weak_control_delta=DELTA, instrument_within_scale=INSTRUMENT_SCALE,
                    stored_float_projection_exact=True,
                    historical_5m_reproduction=False)
    return Fixture(raw, codes, weights, pair_codes, pair_weights, pair_within,
                   row_pair, row_sign, levels, metadata)


def reference(f: Fixture) -> dict:
    """Observation-space QR/SVD reference in stable control-contrast units.

    Collapsing a +/- pair to its positive row with weight 2*w preserves all
    cross products and pair-cluster scores exactly in real arithmetic.  This
    independent reference uses that identity, not FE tuple compression code.
    """
    n = f.raw.shape[0]
    q = n // 2
    a = f.pair_within
    x = np.column_stack((a[:, 1], (a[:, 2] - a[:, 1]) / DELTA, a[:, 3]))
    z = np.column_stack((x[:, :2], a[:, 4], a[:, 5] / INSTRUMENT_SCALE))
    sw = np.sqrt(2 * f.pair_weights)
    xw = x * sw[:, None]
    zw = z * sw[:, None]
    qz, _ = np.linalg.qr(zw / np.linalg.norm(zw, axis=0), mode="reduced")
    del zw, z
    scale = np.linalg.norm(xw, axis=0)
    projected = qz @ (qz.T @ (xw / scale))
    del qz
    u, singular, vt = np.linalg.svd(projected, full_matrices=False)
    del projected
    assert singular[-1] > singular[0] * 1e-3, singular
    inverse = ((vt.T / singular) @ u.T) / scale[:, None]
    del u
    gamma = inverse @ (a[:, 0] * sw)
    residual_pair = a[:, 0] - x @ gamma
    scores = inverse.T * (residual_pair * sw)[:, None]
    cluster = f.pair_codes[:, 0]
    sums = np.column_stack([
        np.bincount(cluster, weights=scores[:, j], minlength=f.levels[0])
        for j in range(3)
    ])
    del scores, inverse
    # FE1 is nested in the FE1 clusters. FE2/FE3 are active and are not nested.
    for g, nlev in zip(f.pair_codes[:, 1:].T, f.levels[1:], strict=True):
        low = np.full(nlev, f.levels[0], dtype=np.int64)
        high = np.full(nlev, -1, dtype=np.int64)
        np.minimum.at(low, g, cluster)
        np.maximum.at(high, g, cluster)
        assert np.any(low != high), "unexpected nested FE changes DoF convention"
    l2, l3 = f.levels[1:]
    graph = coo_matrix((np.ones(q, dtype=np.int32),
                       (f.pair_codes[:, 1], l2 + f.pair_codes[:, 2])),
                      shape=(l2 + l3, l2 + l3)).tocsr()
    components = int(connected_components(graph, directed=False, return_labels=False))
    assert components == 1, "guaranteed community links must connect active FE graph"
    # Independent exact two-way dummy rank, with the public nested-FE anchor
    # convention subtracting one further degree of freedom.
    df_absorbed = l2 + l3 - components - 1
    ng = f.levels[0]
    df_fit = n - 3 - df_absorbed
    correction = ng / (ng - 1) * (n - 1) / df_fit
    transform = np.array([[1., -1 / DELTA, 0.], [0., 1 / DELTA, 0.], [0., 0., 1.]])
    covariance = transform @ ((sums.T @ sums) * correction) @ transform.T
    return dict(beta=transform @ gamma, covariance=covariance,
                residual_pair=residual_pair,
                units=np.sqrt(np.sum(2 * f.pair_weights[:, None] * a[:, 1:4]**2, axis=0) / n),
                rss=float(np.sum(2 * f.pair_weights * residual_pair**2)),
                df_absorbed=df_absorbed, df_fit=df_fit, df_resid=min(df_fit, ng - 1),
                projected_design_condition=float(singular[0] / singular[-1]),
                active_fe_graph_components=components)


def projection_check(f: Fixture, args) -> dict:
    from _econhdfe_within import NativeSolver
    start = time.perf_counter()
    solver = NativeSolver(f.codes, weights=f.weights, strategy=args.strategy,
                          threads=args.threads, workspace_bytes=args.memory_mb * 2**20)
    setup = time.perf_counter() - start
    start = time.perf_counter()
    out, ok, iterations, metric = solver.solve(f.raw, tol=TOL, max_iter=MAX_ITER)
    elapsed = time.perf_counter() - start
    errors = []
    original_row_metric = []
    group_mass = [np.bincount(g, weights=f.weights) for g in f.codes.T]
    for j in range(6):
        truth = f.row_sign * f.pair_within[f.row_pair, j]
        errors.append(float(np.sqrt(np.dot(f.weights, (out[:, j] - truth)**2) /
                                    np.dot(f.weights, truth**2))))
        # Recompute the certificate on returned original-row residuals without
        # trusting the native status/metric fields changed by this PR.
        weighted_scaled = f.weights * (out[:, j] / np.max(np.abs(f.raw[:, j])))
        means = [np.bincount(g, weights=weighted_scaled) / mass
                 for g, mass in zip(f.codes.T, group_mass, strict=True)]
        original_row_metric.append(float(np.max([np.max(np.abs(m)) for m in means])))
    gate = max(TOL, 64 * np.finfo(float).eps)
    passed = bool(all(ok) and np.isfinite(errors).all()
                  and np.max(errors) <= LIMITS["projection_weighted_relative"]
                  and np.isfinite(metric).all() and np.max(metric) <= gate
                  and np.isfinite(original_row_metric).all() and np.max(original_row_metric) <= gate
                  and all(0 <= i <= MAX_ITER for i in iterations))
    return dict(passed=passed, setup_seconds=setup, solve_seconds=elapsed,
                total_seconds=setup + elapsed, converged=list(ok),
                iterations=list(iterations), stationarity=list(metric),
                independent_original_row_stationarity=original_row_metric,
                relative_errors=errors, route=solver.route, cells=solver.n_cells,
                fallback_count=solver.fallback_count,
                schur_fallback_count=getattr(solver, "schur_fallback_count", solver.fallback_count))


def iv_check(f: Fixture, args) -> dict:
    import pandas as pd
    from econhdfe import ExecutionConfig, InferenceConfig, ivhdfe
    from threadpoolctl import threadpool_info
    start = time.perf_counter()
    expected = reference(f)
    reference_seconds = time.perf_counter() - start
    data = {name: f.raw[:, j] for j, name in enumerate(NAMES)}
    data.update({f"fe{j+1}": f.codes[:, j] for j in range(3)})
    data["w"] = f.weights
    frame = pd.DataFrame(data, copy=False)
    fit_threadpools = []
    def capture_threads(_event):
        if not fit_threadpools:
            fit_threadpools.extend({key: pool.get(key) for key in
                                   ("internal_api", "num_threads", "version", "architecture")}
                                  for pool in threadpool_info())
    start = time.perf_counter()
    fit = ivhdfe(frame, y="y", exog=["c1", "c2"], endog=["e"],
                 instruments=["z1", "z2"], absorb=["fe1", "fe2", "fe3"],
                 cluster=["fe1"], vce="cluster", weights="w", weight_type="generic",
                 method=args.method, tol=TOL, max_iter=MAX_ITER,
                 dof_method="exact", drop_singletons=False,
                 inference_config=InferenceConfig(iv_advanced_statistics=False),
                 on_stage=capture_threads,
                 execution_config=ExecutionConfig(threads=args.threads,
                     memory_budget_mb=args.memory_mb, memory_policy="relaxed", cache="off"))
    elapsed = time.perf_counter() - start
    units = expected["units"]
    coeferr = float(np.max(np.abs((fit.params - expected["beta"]) * units) /
                           np.maximum(1., np.abs(expected["beta"] * units))))
    se = np.sqrt(np.diag(expected["covariance"]))
    vceerr = float(np.max(np.abs(fit.vcov - expected["covariance"]) / np.outer(se, se)))
    seerr = float(np.max(np.abs(fit.stderr / se - 1)))
    truth_resid = f.row_sign * expected["residual_pair"][f.row_pair]
    residerr = float(np.sqrt(np.dot(f.weights, (fit.residuals - truth_resid)**2) /
                            expected["rss"]))
    rss = float(np.dot(f.weights, fit.residuals**2))
    rsserr = abs(rss / expected["rss"] - 1)
    metadata = bool(fit.converged and fit.rank == 3 and fit.names == ("c1", "c2", "e")
                    and fit.nobs == len(f.raw) and fit.nobs_raw == len(f.raw)
                    and fit.df_absorbed == expected["df_absorbed"]
                    and fit.df_resid_fit == expected["df_fit"]
                    and fit.df_resid == expected["df_resid"]
                    and fit.dropped_singletons == 0)
    passed = bool(metadata and coeferr <= LIMITS["coefficient_scaled"]
                  and vceerr <= LIMITS["covariance_correlation_scaled"]
                  and seerr <= LIMITS["stderr_relative"]
                  and residerr <= LIMITS["residual_weighted_relative"]
                  and rsserr <= LIMITS["rss_relative"])
    return dict(passed=passed, method=args.method, fit_seconds=elapsed,
                active_fit_threadpools=fit_threadpools,
                reference_seconds=reference_seconds, sample_rank_dof_passed=metadata,
                coefficient_scaled_error=coeferr, covariance_correlation_scaled_error=vceerr,
                stderr_relative_error=seerr, residual_weighted_relative_error=residerr,
                rss_relative_error=rsserr, beta=np.asarray(fit.params).tolist(),
                beta_reference=expected["beta"].tolist(), nobs=int(fit.nobs),
                rank=int(fit.rank), df_absorbed=int(fit.df_absorbed),
                df_resid_fit=float(fit.df_resid_fit), df_resid=float(fit.df_resid),
                iterations=int(fit.iterations), absorb_info=getattr(fit, "absorb_info", None),
                reference_projected_condition=expected["projected_design_condition"])


def provenance() -> dict:
    import econhdfe
    import _econhdfe_within as native
    import scipy
    import pandas
    import numba
    import psutil
    root = Path(econhdfe.__file__).resolve().parent.parent
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL)
    try:
        commit = git("rev-parse", "HEAD").decode().strip()
        diff_hash = hashlib.sha256(git("diff", "HEAD", "--binary")).hexdigest()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = diff_hash = None
    native_path = Path(native.__file__).resolve()
    if native_path.suffix == ".py":
        native_path = next(native_path.parent.glob("*.so"))
    return dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                native_sha256=hashlib.sha256(native_path.read_bytes()).hexdigest(),
                native_path=str(native_path), package_path=econhdfe.__file__,
                package_commit=commit, package_diff_sha256=diff_hash,
                upstream=native.UPSTREAM_REV, python=platform.python_version(),
                numpy=np.__version__, scipy=scipy.__version__, pandas=pandas.__version__,
                numba=numba.__version__, psutil=psutil.__version__, platform=platform.platform())


def environment_measurement() -> dict:
    """Require process-local RSS measurement; do not alter runtime admission."""
    from econhdfe.planner.resources import effective_cpu_count
    if platform.system() == "Linux":
        proc_pid = int(Path("/proc/self/stat").read_text().split()[0])
        if proc_pid != os.getpid():
            raise RuntimeError("PID namespace differs from /proc: psutil cannot safely identify "
                               "this process. Run validation with a matching process/proc namespace.")
    return dict(rss_measurement="unmodified psutil.Process",
                harness_rss_override=False, pid_namespace_mismatch=False,
                effective_cpu_count=effective_cpu_count(),
                blas_threads={key: os.environ.get(key) for key in
                              ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--mode", choices=("projection", "iv", "both"), default="both")
    parser.add_argument("--strategy", choices=("auto", "plain", "compressed"), default="auto")
    parser.add_argument("--method", choices=("within", "auto"), default="within")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-mb", type=int, default=2048)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = dict(provenance=provenance(), environment=environment_measurement(),
                  limits=LIMITS, tolerance=TOL,
                  max_iter=MAX_ITER, threads=args.threads, memory_mb=args.memory_mb)
    start = time.perf_counter()
    f = make_fixture(args.rows, args.seed)
    result["fixture"] = f.metadata
    result["generation_seconds"] = time.perf_counter() - start
    for name, run in (("projection", projection_check), ("iv", iv_check)):
        if args.mode not in (name, "both"):
            continue
        try:
            result[name] = run(f, args)
        except Exception as exc:
            result[name] = dict(passed=False, exception=type(exc).__name__,
                                message=str(exc), traceback=traceback.format_exc())
        print(json.dumps({name: result[name]}, default=str), flush=True)
    result["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if platform.system() == "Darwin" else 1024)
    result["passed"] = all(result[k]["passed"] for k in ("projection", "iv") if k in result)
    payload = json.dumps(result, indent=2, default=str) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, flush=True)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
