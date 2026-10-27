"""Shared Poisson-family primitives for PPML and IV-PPML models."""
from __future__ import annotations
import math
import numpy as np
from numba import njit, prange
from ..frontend.validate import require_numeric
from ..frontend.roles import VariableRole
from ..errors import SpecificationError


def prepare_poisson_base(y, *, offset=None, exposure=None, weights=None):
    """Validate the common Poisson outcome/offset/weight inputs."""
    if offset is not None and exposure is not None:
        raise SpecificationError("only one of offset or exposure may be specified", code="specification.offset_exposure", stage="frontend")
    y = require_numeric(y, name="y", role=VariableRole.OUTCOME, ndim=1, nonnegative=True)
    n = len(y)
    w = np.ones(n, dtype=np.float64) if weights is None else require_numeric(weights, name="weights", role=VariableRole.WEIGHT, ndim=1, positive=True)
    if w.shape != (n,):
        from ..errors import ShapeError
        raise ShapeError("weights must have nobs rows", details={"expected": n, "shape": w.shape})
    off = np.zeros(n, dtype=np.float64) if offset is None else require_numeric(offset, name="offset", role=VariableRole.OFFSET, ndim=1)
    if off.shape != (n,):
        from ..errors import ShapeError
        raise ShapeError("offset must have nobs rows", details={"expected": n, "shape": off.shape})
    if exposure is not None:
        expv = require_numeric(exposure, name="exposure", role=VariableRole.EXPOSURE, ndim=1, positive=True)
        if expv.shape != (n,):
            from ..errors import ShapeError
            raise ShapeError("exposure must have nobs rows", details={"expected": n, "shape": expv.shape})
        off = off + np.log(expv)
    return y, off, w


@njit(cache=True, nogil=True, parallel=True)
def exp_clip_into(eta, mu):
    """Stable elementwise exponentiation with in-place predictor clipping."""
    for i in prange(len(eta)):
        x = eta[i]
        if x < -745.0:
            x = -745.0
        elif x > 709.0:
            x = 709.0
        eta[i] = x
        mu[i] = math.exp(x)


@njit(cache=True, nogil=True, parallel=True)
def working_state(y, mu, eta, offset, base_w, irls_w, z):
    """Fill Poisson IRLS weights and working response in one observation pass."""
    for i in prange(len(y)):
        m = mu[i]
        irls_w[i] = base_w[i] * m
        zi = eta[i] - offset[i] - 1.0
        if y[i] > 0.0:
            zi += y[i] / m
        z[i] = zi


@njit(cache=True, nogil=True)
def _poisson_deviance_sum(y, mu, weights, eta):
    acc = 0.0
    for i in range(len(y)):
        yi = y[i]
        term = mu[i] - yi
        if yi > 0.0:
            term += yi * (math.log(yi) - eta[i])
        acc += weights[i] * term
    return 2.0 * acc


def poisson_deviance(y, mu, weights, eta=None):
    """Poisson deviance without N-sized temporary arrays."""
    y = np.asarray(y, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    eta_arr = np.log(mu) if eta is None else np.asarray(eta, dtype=np.float64)
    dev = float(_poisson_deviance_sum(y, mu, w, eta_arr))
    if dev / max(len(y), 1) < np.finfo(float).eps:
        dev = 0.0
    return max(dev, 0.0)


def prepare_ppml_weights(weights, n, weight_type=None):
    """Validate positive PPML weights without renormalizing their likelihood."""
    from ..compute.weights import prepare_weights
    info = prepare_weights(weights, n, weight_type, normalize_ap=False)
    if info.kind == "aweight":
        raise SpecificationError("PPML supports fweight/pweight, not aweight", code="specification.weight_type")
    w = np.ones(n, dtype=np.float64) if info.estimation is None else info.estimation
    return w, info.kind
