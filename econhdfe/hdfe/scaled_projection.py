"""RHS equilibration adapter for strict weighted projection callers.

This is the numerically identical adapter already used by the local dev1
IV profile model, exposed at its proper lower layer for inference clients.
The adapter is not a new FE solver. Callers must select an appropriate solver.
"""
from __future__ import annotations
import numpy as np
from ..errors import NumericalError


def residualize_scaled(projector, data, weights, *, tol, copy=False):
    """Equilibrate RHS before projection; never accept nonconverged FE output.

    Tiny included-exogenous residual columns can otherwise trigger an absolute
    breakdown threshold in a fused multi-RHS backend and leave other columns
    unprojected. This adapter changes neither FE space nor projection target.
    """
    data=np.array(data,dtype=float,copy=True) if copy else np.asarray(data,dtype=float)
    scales=np.max(np.abs(data),axis=0)
    scales=np.where(scales>0,scales,1.)
    data/=scales
    try:
        out,info=projector.residualize(data,weights,tol=tol,return_info=True,copy=False)
    except Exception:
        # A rejected trial must not leave the persistent X/Z workspace in
        # equilibrated units before the next trial reuses it.
        data*=scales
        raise
    out*=scales
    if info is not None and not info.converged:
        raise NumericalError("FE projection failed convergence certification", code="numerical.iv_profile_projection")
    return out,info

