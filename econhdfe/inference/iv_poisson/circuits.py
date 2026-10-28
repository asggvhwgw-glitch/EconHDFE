"""Experimental product moments on disjoint balanced FE circuits.

These are NEW moment functions under STRONGER conditional independence and
conditional-IV validity assumptions. They do not replace additive-IV PPML.
A three-way gravity 2x2x2 cube is an example, not arbitrary network coverage.
"""
from __future__ import annotations
import numpy as np
from .panel import _matrix
from .statistic import moment_test


def circuit_s_test(y,x,instruments,beta,positive,negative,*,absorb,offset=None,
                   assume_independent_rows=False,assume_conditional_iv=False):
    """Independent circuit S statistic, no FE estimates or target Jacobian.

    Each row of positive/negative specifies one disjoint circuit. Rows cannot be
    reused across circuits. Signed counts must cancel in EVERY supplied FE
    partition. Conditional on ALL instruments and FE, rows must be independent
    and E[y_i-exp(D_i alpha)r_i(beta0)|Z,FE]=0. Unconditional E[Z u]=0 alone does
    not justify multiplying by arbitrary instrument contrasts/products.
    """
    if assume_independent_rows is not True or assume_conditional_iv is not True:
        raise ValueError("circuit moments require explicit independent-row and conditional-IV assumptions")
    y=np.asarray(y,dtype=float)
    if y.ndim!=1 or not np.all(np.isfinite(y)) or np.any(y<0): raise ValueError("invalid y")
    n=len(y);X=_matrix(x,n,"x");Z=_matrix(instruments,n,"instruments")
    b=np.asarray(beta,dtype=float).reshape(-1)
    if b.shape!=(X.shape[1],) or not np.all(np.isfinite(b)):raise ValueError("all slopes must be fixed")
    plus=np.asarray(positive);minus=np.asarray(negative)
    if plus.ndim!=2 or plus.shape!=minus.shape or plus.shape[0]<2 or plus.shape[1]<1:
        raise ValueError("positive/negative must be equal-shape circuit-index matrices")
    for idx in (plus,minus):
        if not np.issubdtype(idx.dtype,np.integer) or np.any(idx<0) or np.any(idx>=n):
            raise ValueError("invalid circuit indices")
    used=np.concatenate([plus.ravel(),minus.ravel()])
    if len(np.unique(used))!=len(used):
        raise ValueError("overlapping circuits are not independent units; refused")
    groups=list(absorb)
    if not groups:raise ValueError("explicit FE partitions are required for circuit balance verification")
    for arr in groups:
        arr=np.asarray(arr)
        if arr.shape!=(n,):raise ValueError("invalid FE labels")
        if not np.all(np.sort(arr[plus],axis=1)==np.sort(arr[minus],axis=1)):
            raise ValueError("signed counts do not cancel in every FE partition")
    off=np.zeros(n) if offset is None else np.asarray(offset,dtype=float)
    if off.shape!=(n,) or not np.all(np.isfinite(off)):raise ValueError("invalid offset")
    eta=off+X@b
    if not np.all(np.isfinite(eta)):raise ValueError("nonfinite null predictor")
    logy=np.full(n,-np.inf);positive_y=y>0;logy[positive_y]=np.log(y[positive_y])
    a=logy[plus].sum(axis=1)+eta[minus].sum(axis=1)
    c=logy[minus].sum(axis=1)+eta[plus].sum(axis=1)
    finite=np.concatenate([a[np.isfinite(a)],c[np.isfinite(c)]])
    shift=float(finite.max()) if len(finite) else 0.
    # One global scale preserves the statistic; per-circuit random scales would
    # generally invalidate the zero-mean moment and are deliberately not used.
    delta=np.exp(a-shift)-np.exp(c-shift)
    contrast=Z[plus].sum(axis=1)-Z[minus].sum(axis=1)
    scores=contrast*delta[:,None]
    d=dict(method="disjoint_balanced_FE_circuits",circuits=len(plus),rows_used=len(used),
           rows_available=n,rows_discarded=n-len(used),circuit_size=2*plus.shape[1],
           nonzero_product_contrasts=int(np.count_nonzero(delta)),
           FE_balance_verified=True,changes_moment_functions=True,
           independence_assumed_not_tested=True,conditional_instrument_validity_required=True,
           no_incidental_FE_estimates=True,may_have_poor_power_and_heavy_tails=True,
           unbalanced_network_coverage_certified=False)
    out=moment_test(scores,diagnostics=d);out.null_value=b
    return out
