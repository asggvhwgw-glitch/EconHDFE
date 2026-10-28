"""Honest discrete inversion: no invented finite endpoints or filled gaps."""
from __future__ import annotations
import numpy as np


def invert_scalar_grid(test, grid, *, alpha=0.05):
    """Evaluate a point-null test on an ordered scalar grid.

    This returns a discrete approximation, NOT a certified continuum confidence
    interval. A rejected local fit/overflow/rank failure is unknown, not a test
    rejection. Boundary acceptance means search-truncated, not proven infinity.
    """
    points = np.asarray(grid, dtype=float)
    if points.ndim != 1 or len(points) < 2 or not np.all(np.isfinite(points)) or np.any(np.diff(points) <= 0):
        raise ValueError("grid must be finite, one-dimensional and strictly increasing")
    if not 0 < alpha < 1: raise ValueError("alpha must lie in (0,1)")
    records = []
    for b in points:
        try:
            r = test(float(b))
            p = r.pvalue
            state = "unknown" if p is None or not np.isfinite(p) else ("accepted" if p >= alpha else "rejected")
            records.append(dict(value=float(b), status=state, statistic=r.statistic,
                                pvalue=p, diagnostic=r.status))
        except (ValueError, RuntimeError, FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
            records.append(dict(value=float(b), status="unknown", statistic=None, pvalue=None,
                                diagnostic=f"{type(exc).__name__}: {exc}"))
    runs=[]; i=0
    while i < len(records):
        if records[i]["status"] != "accepted": i+=1; continue
        j=i
        while j+1 < len(records) and records[j+1]["status"] == "accepted": j+=1
        runs.append(dict(first_grid_point=records[i]["value"], last_grid_point=records[j]["value"],
                         left_search_boundary=(i==0), right_search_boundary=(j==len(records)-1)))
        i=j+1
    return dict(alpha=float(alpha), records=records, accepted_grid_runs=runs,
                unresolved_points=sum(r["status"]=="unknown" for r in records),
                continuum_set_certified=False, gaps_between_grid_points="not_certified",
                domain=[float(points[0]), float(points[-1])],
                empty_grid_is_not_proof_of_empty_continuum_set=True)
