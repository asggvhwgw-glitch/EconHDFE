"""Experimental dual-kernel absorber; never changes EconHDFE estimator defaults.

Positive weights, categorical intercept FE, unchanged sample only. No singleton
removal, rank/DoF, separation, inference or heterogeneous slopes policy here.
"""
from __future__ import annotations
import json
from typing import Literal
import numpy as np
from ._native import NativeDualSolver, WITHIN_REV, FEREG_REV

__all__ = ["DualAbsorber", "WITHIN_REV", "FEREG_REV"]
__version__ = "0.1.0a1"

class DualAbsorber:
    """Batched FE projection with a common independent backward-error gate.

    The auto policy is an uncalibrated conservative heuristic, frozen after its
    first solve. Explicit backend choices can still fail over unless
    allow_fallback=False. scratch_bytes limits estimated per-block workspace,
    NOT total process memory or third-party factorization fill.
    """
    def __init__(self, codes: np.ndarray, *, weights: np.ndarray | None = None,
                 backend: Literal["auto", "within", "fereg"] = "auto",
                 threads: int = 2, scratch_bytes: int = 512 << 20,
                 allow_fallback: bool = True) -> None:
        codes = np.asarray(codes)
        if codes.ndim != 2 or codes.dtype.kind not in "iu":
            raise ValueError("codes must be a two-dimensional integer array")
        if codes.size and (codes.min() < 0 or codes.max() > np.iinfo(np.uint32).max):
            raise ValueError("FE labels must fit uint32; encode missing values first")
        if isinstance(threads, bool) or int(threads) != threads or threads < 1:
            raise ValueError("threads must be a positive integer")
        self._n = codes.shape[0]
        self._native = NativeDualSolver(
            np.ascontiguousarray(codes, dtype=np.uint32),
            None if weights is None else np.asarray(weights, dtype=np.float64),
            backend=backend, threads=int(threads), scratch_bytes=int(scratch_bytes),
            allow_fallback=allow_fallback,
        )

    def residualize(self, data: np.ndarray, *, tol: float = 1e-10,
                    max_iter: int = 6000, return_info: bool = False):
        """Return original-order weighted residuals; raise on failed certification.

        A vector returns a vector, a matrix returns a matrix. The input is never
        modified. max_iter is PER projection attempt (at most one correction per
        kernel); attempt timings/iterations are retained in diagnostics.
        """
        data = np.asarray(data, dtype=np.float64)
        vector = data.ndim == 1
        if vector:
            data = data[:, None]
        if data.ndim != 2 or data.shape[0] != self._n:
            raise ValueError("data must have shape (n,) or (n, rhs)")
        output, info = self._native.solve_block(data, tol=tol, max_iter=max_iter)
        if vector:
            output = output[:, 0]
        return (output, json.loads(info)) if return_info else output

    def update_weights(self, weights: np.ndarray) -> None:
        """Update positive numerical weights without changing FE/sample structure.

        Invalid weights leave the old state unchanged. fereg set_weights keeps
        its structure and eligible factors; within reuses Design and rebuilds
        numerical Solver state. No warm FE coefficient seed is used in v0.1.
        """
        self._native.update_weights(np.asarray(weights, dtype=np.float64))

    @property
    def diagnostics(self) -> dict:
        return json.loads(self._native.diagnostics_json)

    @property
    def backend(self) -> str:
        return self._native.selected_backend

    @property
    def weight_generation(self) -> int:
        return self._native.weight_generation
