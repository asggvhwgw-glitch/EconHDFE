"""Shared Poisson-family primitives for PPML and IV-PPML models."""
from __future__ import annotations
import math
import numpy as np
from numba import njit, prange
from ..frontend.validate import require_numeric
from ..frontend.roles import VariableRole
from ..errors import SpecificationError, InputError
from ..compute.clusters import normalize_vce_clusters
from ..compute.vcov import finalize_cluster_vcov, normalize_vce


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


_AWEIGHT_REJECTION = {
    "ppml": "PPML supports fweight/pweight, not aweight",
    "ivppml": "IV-PPML supports upstream-compatible fweight/pweight, not aweight",
}


def prepare_ppml_weights(weights, n, weight_type=None, *, family="ppml"):
    """Validate positive PPML weights without renormalizing their likelihood."""
    from ..compute.weights import prepare_weights
    info = prepare_weights(weights, n, weight_type, normalize_ap=False)
    if info.kind == "aweight":
        raise SpecificationError(_AWEIGHT_REJECTION[family], code="specification.weight_type")
    w = np.ones(n, dtype=np.float64) if info.estimation is None else info.estimation
    return w, info.kind


def initial_active_rows(initial, *, require_rows):
    """Rows kept by the initial singleton/FE-zero stage.

    An empty result is an error once rows were dropped. ``require_rows``
    (IV-PPML) also rejects an empty input here; PPML leaves that case to the
    separation check, which reports ``identification.no_finite_estimate``.
    """
    dropped = initial.dropped
    active = ~dropped
    if (require_rows or np.any(dropped)) and not np.any(active):
        raise InputError(
            "no observations remain after recursive singleton removal",
            code="input.empty_after_singletons", stage="sample",
            details={"nobs_raw": int(len(dropped)), "n_singletons": int(np.sum(dropped)), "reason": "all_rows_pruned_as_singletons"},
            suggestion="Simplify the FE specification or retain levels with repeated support; no FE coefficient can be recovered from an empty sample.",
        )
    return active


def separation_survivors(y, sep, n_fe_initial):
    """Fold initial FE-zero removals into ``sep`` and return the surviving rows.

    Raise when no row with a positive outcome survives: no finite MLE exists.
    """
    if "fe" in sep.by_method:
        sep.by_method["fe"] += n_fe_initial
    keep = ~(sep.separated | sep.singletons)
    if not np.any(keep) or not np.any(y[keep] > 0):
        raise InputError(
            "PPML has no finite estimable sample after separation",
            code="identification.no_finite_estimate", stage="separation",
            details={
                "nobs_after_singletons": int(len(y)),
                "n_separated": int(np.sum(sep.separated)),
                "separation_by_method": dict(sep.by_method),
                "reason": "ppml_separation_no_finite_mle",
            },
            suggestion="Inspect all-zero FE levels and separated support; changing normalization cannot create a finite PPML estimate.",
        )
    return keep


def poisson_clusters(clusters, sample, vce):
    """Cluster arrays on the estimation sample, normalized for cluster VCE."""
    cls = None
    if clusters is not None:
        raw = clusters if isinstance(clusters, (list, tuple)) else [clusters]
        cls = [np.asarray(c)[sample] for c in raw]
    if normalize_vce(vce) == "cluster":
        cls = normalize_vce_clusters(cls, nobs=int(np.sum(sample)))
    return cls


def poisson_inference(V, *, kind, clusters, scale, constant_index=None):
    """Reported covariance (multiway-cluster repaired) and standard errors."""
    V = finalize_cluster_vcov(V, kind, clusters, scale, constant_index)
    return V, np.sqrt(np.maximum(np.diag(V), 0.0))


def poisson_df_resid(clusters, effective_n, n_params, df_absorbed):
    """Clustered: min(G) - 1; otherwise N - K - df_a, floored at zero."""
    if clusters:
        return min([max(len(np.unique(c)) - 1, 0) for c in clusters])
    return max(effective_n - n_params - df_absorbed, 0.0)
