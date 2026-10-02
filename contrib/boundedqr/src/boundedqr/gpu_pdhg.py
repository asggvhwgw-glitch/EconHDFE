"""Bounded-memory GPU proposals for quantreg's finite bootstrap LP.

This is exploratory diagonally preconditioned PDHG, an established method.
Shared-matrix batching is prior art (Blin et al., arXiv:2601.21990), not a
novelty claim. Returned coefficients are proposals for CPU active-set repair;
neither the termination heuristic nor the diagnostics certify optimality.

Interface: beta_batch, diagnostics = solve_batch(X, y, tau, W, beta0).
X is dense n-by-p, W is p-by-B, beta0 is p or B-by-p, output is B-by-p.
Each solve retains the finite pseudo row d_b=W_b/tau and M=n*max(abs(y)).
No n-by-B array or block-diagonal design is constructed. Working duals have
n-by-batch_size entries. Sparse input is deliberately rejected, not densified.
"""
from pathlib import Path
import os
import time
import numpy as np

def _prepare_windows_import_paths():
    """Keep NVRTC include paths ASCII via existing Windows 8.3 aliases.

    This changes only this process's search paths, not installed packages.
    CuPy's NVRTC compiler cannot open some non-ASCII header paths on Windows.
    Respect an explicit CuPy cache. Otherwise use the configured BoundedQR
    cache root, or .cache/cupy under the working directory. Installed wheels
    never try to write into their Python installation directory.
    """
    explicit=os.environ.get('CUPY_CACHE_DIR')
    cache=(Path(explicit) if explicit else
           Path(os.environ.get('BOUNDEDQR_CACHE_DIR',str(Path.cwd()/'.cache'))) / 'cupy')
    cache.mkdir(parents=True,exist_ok=True)
    os.environ['CUPY_CACHE_DIR']=str(cache.resolve())
    if os.name != "nt":
        return
    import ctypes
    import sys
    get_short = ctypes.windll.kernel32.GetShortPathNameW
    get_short.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
    get_short.restype = ctypes.c_uint

    def short_path(path):
        if not path or not os.path.exists(path):
            return path
        size = get_short(path, None, 0)
        if not size:
            return path
        buffer = ctypes.create_unicode_buffer(size)
        return buffer.value if get_short(path, buffer, size) else path

    sys.path[:] = [short_path(entry) for entry in sys.path]
    os.environ["CUPY_CACHE_DIR"] = short_path(str(cache))


def _original_diagnostics(x, y, tau, w, beta, dual, pseudo_dual, m):
    """Original-FP64 diagnostics; stationarity is not silently assumed."""
    dual = np.clip(dual, tau - 1.0, tau)
    pseudo_dual = float(np.clip(pseudo_dual, tau - 1.0, tau))
    r = y - x @ beta
    d = w / tau
    r0 = float(m - d @ beta)
    h = np.where(r >= 0, (tau - dual) * r,
                 (dual - tau + 1.0) * (-r))
    h0 = ((tau - pseudo_dual) * r0 if r0 >= 0 else
          (pseudo_dual - tau + 1.0) * (-r0))
    eq = x.T @ dual + d * pseudo_dual
    loss = float(np.maximum(tau * r, (tau - 1.0) * r).sum())
    centered = loss - float(w @ beta) + max(float(d @ beta) - m, 0.0)
    return dict(centered_objective=centered, ordinary_loss=loss,
                fenchel_sum=float(h.sum() + h0),
                equality_inf=float(np.linalg.norm(eq, ord=np.inf)),
                equality_l2=float(np.linalg.norm(eq)),
                pseudo_residual=r0, pseudo_dual=pseudo_dual,
                box_violation=0.0, certificate=False)


def solve_batch(X, y, tau, W, beta0, *, batch_size=16, max_iter=1200,
                check_every=100, dtype="float32", primal_weight=None,
                heuristic_tol=1e-4, max_seconds=30.0,
                memory_fraction=0.65, return_dual=False):
    """Generate approximate bootstrap coefficients and explicit diagnostics.

    Diagonal PDHG uses T_j=theta*gamma/(sum_i|X_ij|+|d_j|),
    Sigma_i=theta/(gamma*sum_j|X_ij|), and the analogous pseudo-row
    Sigma. With theta=.95 the scaled-operator bound is strictly below one
    in real arithmetic, including the pseudo row. gamma changes the metric,
    not the objective. Zero row/column sums are protected with a denominator
    of one; those rows/columns make no contribution to the operator norm.

    Best centered primal objective among initial, checked last, and averaged
    iterates chooses the proposal. The final dual is a heuristic witness,
    evaluated together with that proposal on original FP64 data. Each block
    shares its iteration budget; completed columns are not yet compacted.
    max_seconds is checked only at check intervals and bounds exploration,
    not CUDA startup or the original-data diagnostic scan. Set it to None to
    disable it. Validation and device errors propagate; no CPU speedup or
    successful solve is silently substituted. OOM is reported per block.
    """
    start = time.perf_counter()
    if hasattr(X, "tocsr"):
        raise TypeError("This bounded prototype accepts dense X only.")
    x = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(W, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) == 0:
        raise ValueError("X must be a nonempty n-by-p dense matrix.")
    n, p = x.shape
    if y.shape != (n,) or w.ndim != 2 or w.shape[0] != p or w.shape[1] == 0:
        raise ValueError("Expected y(n,) and W(p,B), B>=1.")
    if not 0.0 < tau < 1.0 or dtype not in ("float32", "float64"):
        raise ValueError("Require 0<tau<1 and float32 or float64.")
    if min(batch_size, max_iter, check_every) < 1:
        raise ValueError("Iteration and batch parameters must be positive.")
    if not 0.0 < memory_fraction < 1.0 or heuristic_tol <= 0:
        raise ValueError("Invalid memory fraction or heuristic tolerance.")
    if max_seconds is not None and max_seconds <= 0:
        raise ValueError("max_seconds must be positive or None.")
    b = w.shape[1]
    initial = np.asarray(beta0, dtype=np.float64)
    if initial.shape == (p,):
        initial = np.broadcast_to(initial, (b, p)).copy()
    elif initial.shape != (b, p):
        raise ValueError("beta0 must have shape (p,) or (B,p).")
    for name, a in [("X", x), ("y", y), ("W", w), ("beta0", initial)]:
        if not np.isfinite(a).all():
            raise ValueError(f"{name} contains nonfinite values.")
    gamma = float(np.sqrt(n / p) if primal_weight is None else primal_weight)
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("primal_weight must be finite and positive.")
    m = float(n * np.max(np.abs(y)))
    if not np.isfinite(m) or not np.isfinite(w / tau).all():
        raise ValueError("Finite pseudo-observation overflowed in FP64.")

    _prepare_windows_import_paths()
    import cupy as cp
    stream = cp.cuda.get_current_stream()
    free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
    item = np.dtype(dtype).itemsize
    # Conservative live-array estimate, including temporary absolute values
    # of X during preconditioning. CuPy allocator reservations are additional.
    common_bytes = item * (2 * n * p + 5 * n + 5 * p)
    per_rep_bytes = item * (8 * n + 12 * p + 32)
    affordable = int((memory_fraction * free_bytes - common_bytes) // per_rep_bytes)
    if affordable < 1:
        raise MemoryError("Estimated bounded GPU working set exceeds free memory.")
    effective_batch = min(int(batch_size), b, affordable)
    diagnostics = dict(method="diagonal_PDHG_proposal", novelty_claim=False,
                       tau=float(tau), shape=[n, p, b], dtype=dtype,
                       primal_weight=gamma, batch_size=effective_batch,
                       pseudo_M=m, max_iter=int(max_iter),
                       heuristic_tol=heuristic_tol,
                       max_seconds=max_seconds, certificate=False,
                       initial_free_device_bytes=int(free_bytes),
                       total_device_bytes=int(total_bytes),
                       estimated_working_bytes=int(common_bytes +
                                                   effective_batch * per_rep_bytes),
                       timings={}, blocks=[], replicates=[])
    upload_start = time.perf_counter()
    gx = cp.asarray(x, dtype=dtype)
    gy = cp.asarray(y, dtype=dtype)
    colsum = cp.abs(gx).sum(axis=0)
    rowsum = cp.abs(gx).sum(axis=1)
    sigma = (.95 / gamma) / cp.where(rowsum > 0, rowsum, 1)
    stream.synchronize()
    diagnostics["timings"]["common_upload_and_metric_seconds"] = time.perf_counter() - upload_start
    result = initial.copy()
    duals = [] if return_dual else None
    solve_start = time.perf_counter()
    for first in range(0, b, effective_batch):
        last = min(b, first + effective_batch)
        block_start = time.perf_counter()
        history = []
        status = "iteration_limit"
        iteration = 0
        block = dict(first=first, last=last)
        try:
            gw = cp.asarray(w[:, first:last], dtype=dtype)
            gd = gw / tau
            c = cp.asarray(initial[first:last].T, dtype=dtype)
            bar = c.copy()
            average = c.copy()
            best = c.copy()
            r = gy[:, None] - gx @ c
            dual = cp.where(r >= 0, tau, tau - 1.0).astype(dtype)
            r0 = m - cp.sum(gd * c, axis=0)
            pseudo = cp.where(r0 >= 0, tau, tau - 1.0).astype(dtype)
            denom = colsum[:, None] + cp.abs(gd)
            eta = (.95 * gamma) / cp.where(denom > 0, denom, 1)
            pseudo_rowsum = cp.abs(gd).sum(axis=0)
            sigma0 = (.95 / gamma) / cp.where(pseudo_rowsum > 0, pseudo_rowsum, 1)
            scale = 1.0 + cp.sum(cp.abs(gy))
            eq_scale = 1.0 + cp.max(denom, axis=0)

            def objective(coeff):
                residual = gy[:, None] - gx @ coeff
                value = cp.maximum(tau * residual, (tau - 1.0) * residual).sum(axis=0)
                return value - cp.sum(gw * coeff, axis=0) + cp.maximum(cp.sum(gd * coeff, axis=0) - m, 0)

            best_value = objective(c)
            stream.synchronize()
            iterate_start = time.perf_counter()
            for iteration in range(1, max_iter + 1):
                residual = gy[:, None] - gx @ bar
                dual = cp.clip(dual + sigma[:, None] * residual, tau - 1.0, tau)
                pseudo = cp.clip(pseudo + sigma0 * (m - cp.sum(gd * bar, axis=0)), tau - 1.0, tau)
                nxt = c + eta * (gx.T @ dual + gd * pseudo[None, :])
                bar = 2 * nxt - c
                c = nxt
                average += (c - average) / iteration
                if iteration % check_every and iteration != max_iter:
                    continue
                for proposal in (c, average):
                    value = objective(proposal)
                    take = value < best_value
                    best = cp.where(take[None, :], proposal, best)
                    best_value = cp.minimum(best_value, value)
                residual = gy[:, None] - gx @ c
                r0 = m - cp.sum(gd * c, axis=0)
                h = cp.where(residual >= 0, (tau - dual) * residual,
                             (dual - tau + 1.0) * (-residual)).sum(axis=0)
                h += cp.where(r0 >= 0, (tau - pseudo) * r0,
                              (pseudo - tau + 1.0) * (-r0))
                eq = gx.T @ dual + gd * pseudo[None, :]
                eq_relative = cp.max(cp.abs(eq), axis=0) / eq_scale
                score = cp.maximum(h / scale, eq_relative)
                values = cp.asnumpy(score)
                history.append(dict(iteration=iteration,
                                    worst_heuristic=float(np.max(values)),
                                    median_heuristic=float(np.median(values))))
                if not np.isfinite(values).all():
                    status = "nonfinite_iteration"
                    break
                if (values <= heuristic_tol).all():
                    status = "heuristic_tolerance"
                    break
                if max_seconds is not None and time.perf_counter() - solve_start >= max_seconds:
                    status = "time_limit"
                    break
            stream.synchronize()
            block["iteration_seconds"] = time.perf_counter() - iterate_start
            download_start = time.perf_counter()
            best_cpu = cp.asnumpy(best).T.astype(np.float64)
            dual_cpu = cp.asnumpy(dual).astype(np.float64)
            pseudo_cpu = cp.asnumpy(pseudo).astype(np.float64)
            block["download_seconds"] = time.perf_counter() - download_start
            audit_start = time.perf_counter()
            for j in range(last - first):
                result[first + j] = best_cpu[j]
                row = _original_diagnostics(x, y, tau, w[:, first + j],
                                            best_cpu[j], dual_cpu[:, j], pseudo_cpu[j], m)
                row.update(replicate=first + j, status=status, iterations=iteration)
                diagnostics["replicates"].append(row)
                if return_dual:
                    duals.append(np.r_[np.clip(dual_cpu[:, j], tau - 1, tau),
                                       np.clip(pseudo_cpu[j], tau - 1, tau)])
            block["original_fp64_diagnostic_seconds"] = time.perf_counter() - audit_start
        except cp.cuda.memory.OutOfMemoryError as exc:
            status = "out_of_memory"
            block["error"] = str(exc)
            for j in range(first, last):
                diagnostics["replicates"].append(dict(replicate=j, status=status,
                                                      iterations=iteration, certificate=False))
                if return_dual:
                    duals.append(None)
        block.update(status=status, iterations=iteration, history=history,
                     total_seconds=time.perf_counter() - block_start)
        diagnostics["blocks"].append(block)
        if status in ("time_limit", "out_of_memory", "nonfinite_iteration"):
            for j in range(last, b):
                diagnostics["replicates"].append(dict(replicate=j,
                                                      status="not_run_after_" + status,
                                                      iterations=0, certificate=False))
                if return_dual:
                    duals.append(None)
            break
    diagnostics["timings"]["all_blocks_seconds"] = time.perf_counter() - solve_start
    diagnostics["timings"]["total_seconds"] = time.perf_counter() - start
    diagnostics["device_pool_reserved_bytes"] = int(cp.get_default_memory_pool().total_bytes())
    if return_dual:
        # Opt-in only: returning all ordinary duals has n-by-B host storage.
        diagnostics["dual_batch"] = duals
    return result, diagnostics
