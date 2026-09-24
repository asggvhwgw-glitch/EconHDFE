"""Independent array API for the unmodified PyFixest 0.60.0 FN kernel.

This is a reuse adapter, not a new quantile solver.  The pinned upstream file is
MIT licensed, Copyright (c) 2022 pyfixest authors; its complete license and source
distribution are retained under vendor/pyfixest.  Only NumPy and SciPy are
needed: importing this module does not import or install the PyFixest package,
its Rust extension, formula framework, numba, or R.

The scope is dense, finite, full-column-rank X with an explicitly supplied
intercept if wanted, one quantile, and no observation weights.  The same
Cholesky setup as Quantreg.fit_qreg_fn is used.  Upstream convergence is an
absolute complementarity-gap check; it is not a coefficient-error certificate.
The returned QR dual is diagnostic and MUST NOT replace the frozen residual-
indicator psi in an existing quantreg bootstrap comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, solve_triangular


UPSTREAM_VERSION = "0.60.0"
UPSTREAM_FILE = Path(__file__).with_name("_vendor") / "pyfixest_fn.py"
UPSTREAM_SHA256 = "54cb36544e902b549d344637698d82e3686eb99059df32073cf89d32eee50d38"
_KERNEL = None


def _load_kernel():
    global _KERNEL
    if _KERNEL is None:
        if not UPSTREAM_FILE.is_file():
            raise FileNotFoundError(
                "The installed boundedqr package is incomplete: its pinned "
                "PyFixest numerical source is missing. Reinstall boundedqr."
            )
        digest = hashlib.sha256(UPSTREAM_FILE.read_bytes()).hexdigest()
        if digest != UPSTREAM_SHA256:
            raise RuntimeError("Pinned PyFixest FN source hash mismatch")
        spec = importlib.util.spec_from_file_location(
            "_project_reference_pyfixest_060_fn", UPSTREAM_FILE
        )
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load pinned PyFixest FN source")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _KERNEL = module.frisch_newton_solver
    return _KERNEL


@dataclass
class ReferenceFit:
    beta: np.ndarray
    residual: np.ndarray
    dual: np.ndarray
    converged: bool
    upstream_iteration_index: int
    complementarity_gap: float
    stationarity_inf: float
    box_violation: float
    check_loss: float
    tolerance: float
    source_sha256: str = UPSTREAM_SHA256


def fn_fit(
    X: np.ndarray,
    y: np.ndarray,
    tau: float = 0.5,
    *,
    tol: float = 1e-6,
    maxiter: int | None = None,
) -> ReferenceFit:
    """Fit one QR by reusing the pinned upstream FN kernel without changes.

    ``beta`` and ``residual`` are available directly.  A caller requiring
    convergence must inspect ``converged``; the original upstream flag and
    zero-based iteration index are preserved.  Rank deficiency raises the
    SciPy Cholesky error; this adapter does not silently drop regressors.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if X.ndim != 2 or y.ndim != 1 or len(y) != len(X):
        raise ValueError("Require X of shape (n,p) and y of shape (n,) or (n,1)")
    n, p = X.shape
    if n < p or p < 1:
        raise ValueError("Require n >= p >= 1 and full column rank")
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("X and y must be finite")
    tau, tol = float(tau), float(tol)
    if not np.isfinite(tau) or not 0 < tau < 1:
        raise ValueError("tau must lie strictly between 0 and 1")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    if maxiter is None:
        maxiter = n
    if isinstance(maxiter, bool) or int(maxiter) != maxiter or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")

    # This setup is the public Quantreg.fit_qreg_fn setup, with array validation.
    chol, _ = cho_factor(X.T @ X, lower=True, check_finite=False)
    chol = np.atleast_2d(chol)
    P = solve_triangular(chol, X.T, lower=True, check_finite=False)
    beta, converged, iteration, x, s, z, w, _ = _load_kernel()(
        A=X.T, b=(1 - tau) * X.T @ np.ones(n), c=-y, u=np.ones(n),
        q=tau, tol=tol, max_iter=int(maxiter), chol=chol, P=P,
        backoff=0.9995, beta_init=None,
    )
    residual = y - X @ beta
    dual = x + (tau - 1.0)
    return ReferenceFit(
        beta=beta, residual=residual, dual=dual,
        converged=bool(converged), upstream_iteration_index=int(iteration),
        complementarity_gap=float(x @ z + s @ w),
        stationarity_inf=float(np.max(np.abs(X.T @ dual))),
        box_violation=float(max(0.0, np.max(tau - 1.0 - dual), np.max(dual - tau))),
        check_loss=float(np.sum(np.maximum(tau * residual, (tau - 1.0) * residual))),
        tolerance=tol,
    )
