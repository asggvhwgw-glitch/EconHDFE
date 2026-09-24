"""Bounded FP64 batching of the model-qualified coefficient enclosure.

This is a computational rearrangement of ``error_radius.coefficient_radius``;
it changes neither its theorem nor its proof level. It is NOT a formal interval
proof. Both CPU BLAS and GPU cuBLAS products are assumed to obey the classical
binary64 componentwise gamma_k model, with round-to-nearest, gradual underflow,
and the scalar assumptions documented in error_radius. These assumptions are
not established by the smoke tests. FP32/TF32 products are never requested.

Cache X and abs(X) once. Four shared GEMMs evaluate X@beta.T,
abs(X)@abs(beta.T), X.T@a, and abs(X).T@abs(a). Clip a inward BEFORE any
product. CPU scalar/vector operations construct exactly the original outward
envelopes and verify the small selected-row left inverse. Different BLAS
reduction orders can change the reported bound; bitwise agreement with a
sequence of GEMVs is neither required nor claimed.

The original supplied binary64 X/y/W, including the rounded finite pseudo row,
are the numerical data being enclosed. Upstream gradient/data error is excluded.
The auditor takes host arrays even with backend='gpu'. Its snapshots of X and y
are private so a later caller mutation cannot invalidate cached abs(X).

API: RadiusAuditor(X,y,tau,backend='cpu').audit(W[p,k],beta[k,p],a_aug[k,n+1])
returns a list of RadiusCertificate. k <= max_batch (default 32) is enforced;
there is no retained n-by-total-bootstrap array. Call repeatedly for chunks.
Diagnostics report setup, scan/transfer, envelope and inverse-check times plus
conservative explicit-array budgets, not process RSS or measured peak memory.

Run the bounded existing-data checks (no performance claim):
  python experiments/batched_radius.py --backend both
Optional one-off n=50,000,k=2 smoke: add --small-50k.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .certification import (
    RadiusCertificate, _U, _add_up, _down, _gamma, _lower_endpoint,
    _mul_up, _nonnegative_up, _norm_upper, _positive_reduction_upper,
    _residual_envelope, _select_rows, _sum_upper, _underflow, _up,
    _verified_kappa,
)


class RadiusAuditor:
    """Reuse a dense operator across small blocks of independently audited fits.

    GPU errors propagate; there is no silent CPU fallback. Inputs must be finite
    and have the documented two-dimensional batch shapes (including k=1).
    Invalid inputs raise ValueError; valid but uninformative bounds return inf.
    ``diagnostics`` retains cumulative scalar counters and the last block only.
    ``max_batch`` is a memory contract, not an internal n-by-B expansion.
    """

    def __init__(self, X, y, tau, backend="cpu", *, interior_tol=32 * _U,
                 max_candidates=None, max_batch=32):
        started = time.perf_counter()
        x = np.asarray(X, dtype=np.float64)
        yy = np.asarray(y, dtype=np.float64).reshape(-1)
        if x.ndim != 2 or min(x.shape, default=0) <= 0:
            raise ValueError("X must be a nonempty dense two-dimensional array")
        self.n, self.p = x.shape
        if yy.size != self.n:
            raise ValueError("Expected y of length n")
        self.tau = float(tau)
        if not np.isfinite(self.tau) or not 0 < self.tau < 1:
            raise ValueError("tau must be finite and strictly between zero and one")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(yy)):
            raise ValueError("All supplied arrays must be finite")
        if not np.isfinite(interior_tol) or interior_tol < 0:
            raise ValueError("interior_tol must be nonnegative and finite")
        self.interior_tol = float(interior_tol)
        if max_candidates is None:
            max_candidates = max(32, 8 * self.p)
        if not isinstance(max_candidates, (int, np.integer)) or max_candidates < self.p:
            raise ValueError("max_candidates must be an integer at least p")
        if not isinstance(max_batch, (int, np.integer)) or max_batch < 1:
            raise ValueError("max_batch must be a positive integer")
        self.max_candidates, self.max_batch = int(max_candidates), int(max_batch)
        if backend not in ("cpu", "gpu"):
            raise ValueError("backend must be 'cpu' or 'gpu'")
        self.backend = backend
        self.x = np.array(x, dtype=np.float64, order="C", copy=True)
        self.y = np.array(yy, dtype=np.float64, copy=True)
        self.abs_x = np.abs(self.x)
        for snapshot in (self.x, self.y, self.abs_x):
            snapshot.setflags(write=False)
        self.low, self.high = _lower_endpoint(self.tau)
        with np.errstate(over="ignore"):
            self.t0 = float(self.n * np.max(np.abs(self.y)))
        self.cp = None
        device_cache_bytes, upload_seconds = 0, 0.0
        if backend == "gpu":
            # Existing project helper handles Windows non-ASCII NVRTC paths.
            from .gpu_pdhg import _prepare_windows_import_paths
            _prepare_windows_import_paths()
            import cupy as cp
            self.cp = cp
            upload_started = time.perf_counter()
            self.device_x = cp.asarray(self.x, dtype=cp.float64)
            # Binary64 abs only clears the sign bit (including signed zero and
            # subnormals). Derive the same cache on device, avoiding a second
            # design-sized pinned host-transfer allocation. Inputs are finite.
            self.device_abs_x = cp.abs(self.device_x)
            cp.cuda.get_current_stream().synchronize()
            upload_seconds = time.perf_counter() - upload_started
            device_cache_bytes = self.device_x.nbytes + self.device_abs_x.nbytes
        self.diagnostics = dict(
            backend=backend, dtype="float64", max_batch=self.max_batch,
            setup_seconds=time.perf_counter() - started,
            setup_upload_seconds=upload_seconds,
            persistent_host_array_bytes=self.x.nbytes + self.abs_x.nbytes + self.y.nbytes,
            persistent_device_array_bytes=device_cache_bytes,
            memory_accounting="explicit array budget; excludes caller arrays, BLAS/CUDA workspaces and allocator reserves; not RSS/peak measurement",
            proof_level="FP64 model-qualified numerical enclosure; not formal interval",
            gamma_model_assumed_for="CPU BLAS" if backend == "cpu" else "CPU BLAS and GPU cuBLAS FP64",
            calls=0, draws_audited=0, cumulative_audit_seconds=0.0,
            last_audit=None,
        )

    def _products(self, beta, used_a):
        """Return host FP64 products; pseudo-row products stay on the CPU."""
        if self.backend == "cpu":
            return (self.x @ beta.T, self.abs_x @ np.abs(beta.T),
                    self.x.T @ used_a[:-1], self.abs_x.T @ np.abs(used_a[:-1]))
        cp = self.cp
        device_beta = cp.asarray(beta.T, dtype=cp.float64)
        device_a = cp.asarray(used_a[:-1], dtype=cp.float64)
        # Each product is transferred before the next one, so only one result
        # matrix is live on the device. CuPy may reserve freed allocator blocks;
        # this class does not clear a process-global memory pool.
        dot = cp.asnumpy(self.device_x @ device_beta)
        absolute_dot = cp.asnumpy(self.device_abs_x @ cp.abs(device_beta))
        equality = cp.asnumpy(self.device_x.T @ device_a)
        absolute_equality = cp.asnumpy(self.device_abs_x.T @ cp.abs(device_a))
        return dot, absolute_dot, equality, absolute_equality

    def audit(self, W, beta, a_aug) -> list[RadiusCertificate]:
        """Audit one bounded block; never discard uninformative draws.

        Runtime diagnostics include host validation/clipping, all four shared
        scans, transfers for GPU, and the complete per-column verification.
        Input array storage is caller-owned and excluded from array budgets.
        """
        started = time.perf_counter()
        w, b, a = [np.asarray(v, dtype=np.float64) for v in (W, beta, a_aug)]
        if w.ndim != 2 or w.shape[0] != self.p:
            raise ValueError("W must have shape (p,k)")
        k = w.shape[1]
        if k < 1 or k > self.max_batch:
            raise ValueError(f"Expected 1 <= k <= max_batch={self.max_batch}; pass separate blocks")
        if b.shape != (k, self.p) or a.shape != (k, self.n + 1):
            raise ValueError("Expected beta shape (k,p) and a_aug shape (k,n+1)")
        if any(not np.all(np.isfinite(v)) for v in (w, b, a)):
            raise ValueError("All supplied arrays must be finite")
        # high is at or ABOVE the exact real lower endpoint tau-1. All later
        # products use this inward-clipped witness, never the caller's a.
        used_a = np.clip(a.T, self.high, self.tau)
        validation_seconds = time.perf_counter() - started
        scan_started = time.perf_counter()
        with np.errstate(over="ignore", invalid="ignore", under="ignore", divide="ignore"):
            dot, absolute_dot, equality, absolute_equality = self._products(b, used_a)
            scan_seconds = time.perf_counter() - scan_started
            certificate_started = time.perf_counter()
            results, inverse_seconds = [], 0.0
            for j in range(k):
                result, elapsed = self._certificate(
                    w[:, j], b[j], a[j], used_a[:, j], dot[:, j],
                    absolute_dot[:, j], equality[:, j], absolute_equality[:, j])
                results.append(result)
                inverse_seconds += elapsed
        certificate_seconds = time.perf_counter() - certificate_started
        elapsed = time.perf_counter() - started
        # Conservative budgets for this implementation's explicit array
        # allocations. Each column's nonlinear vectors are released before the
        # next column, and only scalar/dataclass diagnostics survive this call.
        # Includes possible input dtype conversions, abs(a) for the fourth
        # product, and a contiguous staging copy when uploading strided a.
        host_block = 8 * (6 * (self.n + 1) * k + 4 * self.p * k)
        host_scratch = 8 * (40 * (self.n + 1) +
                           20 * self.max_candidates * self.p + 10 * self.p**2)
        device_block = 8 * (4 * self.n * k + 4 * self.p * k) if self.cp is not None else 0
        last = dict(k=k, audit_seconds=elapsed, validation_clipping_seconds=validation_seconds,
                    shared_products_transfer_seconds=scan_seconds,
                    certificate_seconds=certificate_seconds,
                    inverse_verification_seconds=inverse_seconds,
                    nonlinear_envelope_seconds=max(0.0, certificate_seconds - inverse_seconds),
                    finite_count=sum(r.status == "finite" for r in results),
                    host_block_array_budget_bytes=host_block,
                    host_per_column_scratch_budget_bytes=host_scratch,
                    device_block_array_budget_bytes=device_block)
        self.diagnostics["calls"] += 1
        self.diagnostics["draws_audited"] += k
        self.diagnostics["cumulative_audit_seconds"] += elapsed
        self.diagnostics["last_audit"] = last
        return results

    def _certificate(self, w, beta, original_a, used_a, dot, absolute_dot,
                     ordinary_e, ordinary_absolute_e):
        """Original per-column envelope, supplied with batched dot products."""
        result = RadiusCertificate()
        result.dual_lower_endpoint_lower = self.low
        result.dual_lower_endpoint_upper = self.high
        result.clipping_adjustment_max = float(np.max(np.abs(used_a - original_a)))
        margins = np.maximum(0.0, np.minimum(_down(used_a - self.high),
                                            _down(self.tau - used_a)))
        all_interior = np.flatnonzero(margins > self.interior_tol)
        selected = _select_rows(all_interior, margins, self.max_candidates)
        result.interior_count, result.selected_count = int(all_interior.size), int(selected.size)
        result.selected_indices = tuple(int(i) for i in selected)
        z0 = w / self.tau
        if not np.isfinite(self.t0) or not np.all(np.isfinite(z0)):
            result.reason = "nonfinite_augmented_data"
            return result, 0.0

        # Identical gamma_p residual envelope; the only changed operation is
        # the rounded GEMM dot supplied here instead of a standalone GEMV.
        r = self.y - dot
        magnitude = _positive_reduction_upper(absolute_dot, self.p)
        dot_error = _mul_up(_gamma(self.p), magnitude)
        subtraction_error = _mul_up(_up(_U / (1 - _U)), np.abs(r))
        r_error = _add_up(_add_up(dot_error, subtraction_error), _underflow(self.p))
        r0, r0_error = _residual_envelope(z0[None, :], np.array([self.t0]), beta)
        residual, residual_error = np.r_[r, r0], np.r_[r_error, r0_error]
        result.residual_error_max = float(np.max(residual_error))
        positive = np.maximum(0.0, _up(residual + residual_error))
        negative = np.maximum(0.0, _up(-residual + residual_error))
        positive_coefficient = _nonnegative_up(self.tau - used_a)
        negative_coefficient = _nonnegative_up(used_a - self.low)
        terms = _add_up(_mul_up(positive, positive_coefficient),
                        _mul_up(negative, negative_coefficient))
        result.gap_upper = _sum_upper(terms)

        e = ordinary_e + z0 * used_a[-1]
        absolute_e = ordinary_absolute_e + np.abs(z0 * used_a[-1])
        magnitude_upper = _positive_reduction_upper(absolute_e, self.n + 1)
        e_error = _add_up(_mul_up(_gamma(self.n + 1), magnitude_upper),
                          _underflow(self.n + 1))
        result.stationarity_norm_upper = _norm_upper(_add_up(np.abs(e), e_error))
        selected_residual = _add_up(np.abs(residual[selected]), residual_error[selected])
        result.interior_residual_upper = _sum_upper(_mul_up(margins[selected], selected_residual))
        if not np.isfinite(result.gap_upper + result.stationarity_norm_upper + result.interior_residual_upper):
            result.reason = "nonfinite_forward_error_envelope"
            return result, 0.0
        if selected.size < self.p:
            result.reason = "fewer_selected_interior_rows_than_coefficients"
            return result, 0.0
        zrows = np.empty((selected.size, self.p), dtype=np.float64)
        ordinary = selected < self.n
        zrows[ordinary] = self.x[selected[ordinary]]
        zrows[~ordinary] = z0
        verification_started = time.perf_counter()
        kappa, eta, inverse_norm, verification = _verified_kappa(zrows, margins[selected])
        inverse_seconds = time.perf_counter() - verification_started
        result.kappa_lower = kappa
        result.inverse_residual_frobenius_upper = eta
        result.left_inverse_frobenius_upper = inverse_norm
        denominator = float(_down(kappa - result.stationarity_norm_upper))
        result.denominator_lower = max(0.0, denominator)
        if verification != "verified":
            result.reason = verification
            return result, inverse_seconds
        if denominator <= 0:
            result.reason = "stationarity_envelope_not_below_sharpness_bound"
            return result, inverse_seconds
        numerator = float(_add_up(result.gap_upper, result.interior_residual_upper))
        result.radius = float(_up(numerator / denominator))
        if not np.isfinite(result.radius):
            result.reason = "radius_overflow"
            return result, inverse_seconds
        result.status, result.reason = "finite", "model_qualified_enclosure"
        return result, inverse_seconds
