"""Streamed, batched full-data verification around small finite QR dual LPs.

This composes known residual-sign QR preprocessing and shared-X GEMM. It makes
no originality or speed claim. LPs remain independent; only full-data products
are batched. All GPU acceptance products use binary64, never FP32/TF32.

Main API:
    for beta, a_aug, diag in iter_repair_batches(preprocessor, W, proposal):
        # beta: k-by-p; a_aug: k-by-(n+1); global indices in diag.
        # Consume a_aug (e.g. coefficient_radius) before requesting next block.

The convenience fit_batch(..., on_batch=callback) accumulates only B-by-p beta
and small diagnostics. It deliberately does not collect n-by-B duals. CPU/GPU
work arrays are bounded by batch_size<=64 and max_block_bytes; callers can
defeat that bound by retaining yielded blocks, which this module never does.
"""
from __future__ import annotations

import time
import numpy as np
from .positive_pseudo import pseudo_residual_lower

try:
    from .lp import _lp
    from .gpu_pdhg import _prepare_windows_import_paths
except ImportError:
    from .lp import _lp
    from .gpu_pdhg import _prepare_windows_import_paths


class _Products:
    """Synchronized full-data FP64 products, instrumented by operation type."""
    def __init__(self, preprocessor, backend):
        self.x = preprocessor.x
        self.y = preprocessor.y
        self.base_residual = preprocessor.r0
        self.backend = backend
        self.records = []
        start = time.perf_counter()
        if backend == "gpu":
            _prepare_windows_import_paths()
            import cupy as cp
            self.cp = cp
            self.stream = cp.cuda.get_current_stream()
            self.gx = cp.asarray(self.x, dtype=cp.float64)
            self.gy = cp.asarray(self.y, dtype=cp.float64)
            self.gr0 = cp.asarray(self.base_residual, dtype=cp.float64)
            self.stream.synchronize()
            if self.gx.dtype != cp.float64:
                raise AssertionError("GPU verification requires FP64 X.")
        self.setup_seconds = time.perf_counter() - start

    def residual(self, coefficients, *, centered=False):
        coefficients = np.asarray(coefficients, dtype=np.float64)
        start = time.perf_counter()
        if self.backend == "gpu":
            cp = self.cp
            gc = cp.asarray(coefficients, dtype=cp.float64)
            gy = self.gr0 if centered else self.gy
            out = cp.asnumpy(gy[:, None] - self.gx @ gc)
            self.stream.synchronize()
        else:
            response = self.base_residual if centered else self.y
            out = response[:, None] - self.x @ coefficients
        self.records.append(dict(kind="proposal_residual" if centered else "acceptance_residual",
                                 columns=coefficients.shape[1], dtype="float64",
                                 seconds=time.perf_counter() - start))
        return out

    def transpose(self, columns, kind):
        columns = np.asarray(columns, dtype=np.float64)
        start = time.perf_counter()
        if self.backend == "gpu":
            cp = self.cp
            out = cp.asnumpy(self.gx.T @ cp.asarray(columns, dtype=cp.float64))
            self.stream.synchronize()
        else:
            out = self.x.T @ columns
        self.records.append(dict(kind=kind, columns=columns.shape[1], dtype="float64",
                                 seconds=time.perf_counter() - start))
        return out


def iter_repair_batches(preprocessor, W, proposal, *, backend="cpu", batch_size=16,
                        active_multiplier=4.0, max_rounds=10,
                        max_block_bytes=512 * 1024**2, allow_full_fallback=True,
                        positive_pseudo=False):
    """Yield repaired beta, augmented duals, and diagnostics in bounded blocks.

    preprocessor is an existing CFMPreprocessor with shared original FP64 X/y,
    baseline beta0/residual, and leverage. Its setup cost is reported but not
    rerun. W is p-by-B and proposal is B-by-p. Initial active sets have at least
    4p ordinary rows, selected by |proposal residual|/shared leverage. The true
    pseudo row is free by default. Optional positive_pseudo first tries its
    upper face with an outward strict pseudo-residual guard and the original
    full-data checks. Only that phase has an actual doubling growth cap;
    rejected trials restart the original free strategy from the initial state.

    Every repair round solves small CPU LPs for pending draws, then uses two
    batched FP64 products to check the successfully solved draws: X@beta and
    X.T@a. All tail signs, dual bounds, and Fenchel sums are checked. Passing
    draws freeze; failed draws alone expand. Small equality/Fenchel residuals
    are numerical checks, not mathematically exact certificates.

    Returns NaN dual rows with explicit failed status if even a full LP fails
    or if a required fallback was disabled. No failed draw is silently dropped.
    An OOM/device failure propagates rather than silently changing backend.
    """
    if backend not in ("cpu", "gpu"):
        raise ValueError("backend must be cpu or gpu.")
    if not isinstance(positive_pseudo, (bool, np.bool_)):
        raise ValueError('positive_pseudo must be boolean')
    if not isinstance(batch_size, (int, np.integer)) or not 1 <= batch_size <= 64:
        raise ValueError("batch_size must be an integer from 1 to 64.")
    if active_multiplier < 4 or max_rounds < 1 or max_block_bytes <= 0:
        raise ValueError("Require active_multiplier>=4 and positive memory/round limits.")
    pre = preprocessor
    x = np.asarray(pre.x, dtype=np.float64)
    n, p = x.shape
    tau = float(pre.tau)
    w = np.asarray(W, dtype=np.float64)
    candidate = np.asarray(proposal, dtype=np.float64)
    if w.ndim != 2 or w.shape[0] != p or w.shape[1] < 1:
        raise ValueError("W must have shape p-by-B, B>=1.")
    count = w.shape[1]
    if candidate.shape != (count, p):
        raise ValueError("proposal must have shape B-by-p.")
    if not np.isfinite(w).all() or not np.isfinite(candidate).all():
        raise ValueError("W and proposal must be finite.")
    # Original-data validation was already done by the supplied preprocessor.
    # Conservative live array allowance: residuals, anchors, scores, duals,
    # check temporaries and masks. Existing X and GPU X copies are separate.
    bytes_per_column = 8 * 9 * n + 3 * n + 8 * 20 * p
    affordable = int(max_block_bytes // bytes_per_column)
    if affordable < 1:
        raise MemoryError("One verification column exceeds max_block_bytes.")
    block_size = min(batch_size, affordable, count)
    operator = _Products(pre, backend)
    total_start = time.perf_counter()
    initial_size = min(n, int(np.ceil(active_multiplier * p)))
    for first in range(0, count, block_size):
        last = min(count, first + block_size)
        k = last - first
        block_start = time.perf_counter()
        products_start = len(operator.records)
        wb = w[:, first:last]
        d = wb / tau
        pseudobase = pre.pseudo_y - np.sum(d * pre.beta0[:, None], axis=0)
        guess = candidate[first:last]
        residual_guess = operator.residual((guess - pre.beta0).T, centered=True)
        anchor = np.where(residual_guess < 0, tau - 1, tau)
        anchor_total = operator.transpose(anchor, "anchor_total")
        score = np.abs(residual_guess) / pre.leverage[:, None]
        del residual_guess
        active = np.zeros((n, k), dtype=bool)
        for j in range(k):
            active[np.argpartition(score[:, j], initial_size - 1)[:initial_size], j] = True
        beta = guess.copy()
        # Yielded k-by-(n+1) rows are contiguous for the scalar radius API.
        augmented_dual = np.full((n + 1, k), np.nan, dtype=np.float64, order="F")
        records = [dict(replicate=first+j, active_sizes=[], sign_repairs=[], lp_failures=[],
                        fallback=False, numerically_checked=False, status="pending",
                        lp_seconds=0.0, certificate=False,
                        positive_pseudo=dict(enabled=bool(positive_pseudo), attempts=0,
                            lp_failures=0, accepted=False, fallback_reason=None,
                            residual_lower_bound=None, active_sizes=[], free_restart=False)) for j in range(k)]
        pending = list(range(k))
        rounds = []

        def expand(j, target, bounded=False):
            target = min(n, max(int(target), int(active[:, j].sum())))
            if target == n:
                active[:, j] = True
            else:
                selected = np.argpartition(score[:, j], target - 1)[:target]
                if bounded:
                    slots = max(0, int(target)-int(active[:, j].sum()))
                    selected = selected[~active[selected, j]][:slots]
                active[selected, j] = True

        deferred_free = []
        for positive_face in ((True, False) if positive_pseudo else (False,)):
            if positive_pseudo and not positive_face:
                pending = sorted(set(pending + deferred_free))
                for j in pending:
                    # Restore the original free strategy, including its full
                    # round budget and proposal-derived tail partition.
                    active[:, j] = False
                    expand(j, initial_size)
                    beta[j] = guess[j]
                    info = records[j]['positive_pseudo']
                    info['free_restart'] = True
                    if info['fallback_reason'] is None:
                        info['fallback_reason'] = 'positive_round_limit'
            for round_id in range(max_rounds if positive_face else max_rounds + 1):
                if not pending:
                    break
                full_round = round_id == max_rounds
                if full_round and not allow_full_fallback:
                    for j in pending:
                        records[j]["status"] = "fallback_disabled"
                    break
                if full_round:
                    for j in pending:
                        active[:, j] = True
                        records[j]["fallback"] = True
                round_start = time.perf_counter()
                solved, solutions = [], {}
                still_pending = []
                for j in pending:
                    indices = np.flatnonzero(active[:, j])
                    records[j]["active_sizes"].append(int(indices.size))
                    if positive_face:
                        records[j]["positive_pseudo"]["active_sizes"].append(int(indices.size))
                    xx = x[indices]
                    # Homogeneous finite equality before any optional face trial.
                    rhs = -anchor_total[:, j] + xx.T @ anchor[indices, j]
                    if indices.size == n:
                        # Exact zero prevents cancellation of two full anchor
                        # products from polluting the fully expanded LP equality.
                        rhs = np.zeros(p)
                    begin_lp = time.perf_counter()
                    try:
                        info = records[j]['positive_pseudo']
                        if positive_face:
                            info['attempts'] += 1
                            step, ordinary_a = _lp(xx, pre.r0[indices], tau,
                                                   rhs-d[:, j]*tau, limit=pre.solver_limit)
                            lower = pseudo_residual_lower(d[:, j], pre.pseudo_y, pre.beta0+step)
                            info['residual_lower_bound'] = lower
                            if lower <= 0:
                                info['fallback_reason'] = 'pseudo_residual_not_strictly_positive'
                                deferred_free.append(j)
                                continue
                            a = np.r_[ordinary_a, tau]
                        else:
                            step, a = _lp(np.vstack((xx, d[:, j])),
                                          np.r_[pre.r0[indices], pseudobase[j]],
                                          tau, rhs, limit=pre.solver_limit)
                    except RuntimeError as exc:
                        records[j]["lp_failures"].append(str(exc))
                        if positive_face:
                            info['lp_failures'] += 1
                        if indices.size == n:
                            if positive_face:
                                info['fallback_reason'] = 'positive_lp_failed_at_full_design'
                                deferred_free.append(j)
                            else:
                                records[j]["status"] = "full_lp_failed"
                        else:
                            expand(j, 2 * indices.size, bounded=positive_face)
                            still_pending.append(j)
                    else:
                        beta[j] = pre.beta0 + step
                        solved.append(j)
                        solutions[j] = (indices, a)
                    finally:
                        records[j]["lp_seconds"] += time.perf_counter() - begin_lp
                round_summary = dict(round=round_id, pending=len(pending), solved=len(solved),
                                     frozen=0, expanded=len(still_pending), full_round=full_round,
                                     pseudo_mode="positive_face" if positive_face else "free")
                if solved:
                    # All full-data acceptance products in this round are GEMMs.
                    # Compaction excludes already frozen and failed-LP draws.
                    a_block = anchor[:, solved].copy()
                    pseudo_a = np.empty(len(solved))
                    for at, j in enumerate(solved):
                        indices, small_a = solutions[j]
                        a_block[indices, at] = small_a[:-1]
                        pseudo_a[at] = small_a[-1]
                    residual = operator.residual(beta[solved].T)
                    equality = operator.transpose(a_block, "acceptance_equality")
                    equality += d[:, solved] * pseudo_a[None, :]
                    pseudo_r = pre.pseudo_y - np.sum(d[:, solved] * beta[solved].T, axis=0)
                    local = np.where(residual >= 0, (tau - a_block) * residual,
                                     (a_block - tau + 1) * (-residual))
                    local0 = np.where(pseudo_r >= 0, (tau - pseudo_a) * pseudo_r,
                                      (pseudo_a - tau + 1) * (-pseudo_r))
                    gaps = np.maximum(local, 0).sum(axis=0) + np.maximum(local0, 0)
                    raw_minima = np.minimum(local.min(axis=0), local0)
                    eqmax = np.max(np.abs(equality), axis=0)
                    boxes = np.maximum.reduce([np.max(a_block - tau, axis=0),
                        np.max(tau - 1 - a_block, axis=0), pseudo_a - tau,
                        tau - 1 - pseudo_a, np.zeros(len(solved))])
                    wrong = (~active[:, solved]) & (
                        ((anchor[:, solved] == tau) & (residual < -pre.sign_tolerance)) |
                        ((anchor[:, solved] == tau - 1) & (residual > pre.sign_tolerance)))
                    mismatch_counts = wrong.sum(axis=0)
                    for at, j in enumerate(solved):
                        bad = int(mismatch_counts[at])
                        records[j]["sign_repairs"].append(bad)
                        record = dict(gap=float(gaps[at]), fenchel_sum=float(gaps[at]),
                                      minimum_raw_fenchel=float(raw_minima[at]),
                                      equality_residual=float(eqmax[at]),
                                      equality_l2=float(np.linalg.norm(equality[:, at])),
                                      box_violation=float(boxes[at]),
                                      pseudo_residual=float(pseudo_r[at]), pseudo_dual=float(pseudo_a[at]))
                        records[j].update(record)
                        checked = (bad == 0 and gaps[at] <= pre.gap_tolerance and
                                   eqmax[at] <= pre.equality_tolerance and
                                   boxes[at] <= pre.equality_tolerance and
                                   all(np.isfinite(v) for v in record.values()))
                        if checked:
                            augmented_dual[:-1, j] = a_block[:, at]
                            augmented_dual[-1, j] = pseudo_a[at]
                            records[j].update(numerically_checked=True, status="numerically_checked")
                            records[j]['positive_pseudo']['accepted'] = bool(positive_face)
                            round_summary["frozen"] += 1
                        elif full_round or active[:, j].all():
                            if positive_face:
                                records[j]['positive_pseudo']['fallback_reason'] = 'positive_full_data_check_failed'
                                deferred_free.append(j)
                            else:
                                records[j]["status"] = "full_data_check_failed"
                            records[j]["lp_failures"].append("Full-data FP64 checks failed after full expansion")
                        else:
                            old_size = int(active[:, j].sum())
                            if not positive_face:
                                active[wrong[:, at], j] = True
                            if not bad:
                                records[j]["lp_failures"].append("Full-data FP64 numerical checks failed")
                            if positive_face or not bad or bad > .1 * old_size:
                                expand(j, max(2 * old_size, int(active[:, j].sum())), bounded=positive_face)
                            still_pending.append(j)
                            round_summary["expanded"] += 1
                pending = still_pending
                round_summary["seconds"] = time.perf_counter() - round_start
                rounds.append(round_summary)
        for record in records:
            if record["status"] == "pending":
                record["status"] = "round_limit"
        block_seconds = time.perf_counter() - block_start
        diag = dict(backend=backend, acceptance_dtype="float64", certificate=False,
                    replicate_indices=list(range(first, last)), fits=records, rounds=rounds,
                    all_checked=all(r["numerically_checked"] for r in records),
                    fallback_count=sum(r["fallback"] for r in records),
                    failed_count=sum(not r["numerically_checked"] for r in records),
                    actual_batch_size=k, estimated_array_bytes=bytes_per_column*k,
                    max_block_bytes=max_block_bytes, block_seconds=block_seconds,
                    preprocessor_setup_seconds=pre.setup_seconds,
                    shared_backend_setup_seconds=operator.setup_seconds if first == 0 else 0.0,
                    products=operator.records[products_start:],
                    elapsed_repair_seconds=time.perf_counter()-total_start)
        yield beta, augmented_dual.T, diag


def fit_batch(preprocessor, W, proposal, *, on_batch=None, **options):
    """Collect B-by-p beta; optional callback consumes each bounded dual block.

    callback(beta[k,p], a_aug[k,n+1], block_diag) may compute radii immediately.
    If on_batch is None, full duals are discarded after numerical verification.
    The returned dictionary contains only small diagnostics, never n-by-B data.
    Callback cost is included in total_seconds and listed separately.
    """
    start = time.perf_counter()
    w = np.asarray(W)
    beta = np.full((w.shape[1], preprocessor.p), np.nan)
    blocks = []
    callback_seconds = 0.0
    for values, duals, diag in iter_repair_batches(preprocessor, W, proposal, **options):
        beta[diag["replicate_indices"]] = values
        if on_batch is not None:
            begin = time.perf_counter()
            on_batch(values, duals, diag)
            callback_seconds += time.perf_counter() - begin
        blocks.append(diag)
    return beta, dict(blocks=blocks, all_checked=all(d["all_checked"] for d in blocks),
                      failed_count=sum(d["failed_count"] for d in blocks),
                      fallback_count=sum(d["fallback_count"] for d in blocks),
                      total_seconds=time.perf_counter()-start,
                      callback_seconds=callback_seconds, certificate=False)
