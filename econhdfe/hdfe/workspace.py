from __future__ import annotations
from .projection import estimate_index_bytes


def absorption_build_cost(groups, kind):
    """Bound known constructor arrays without first constructing FE indexes.

    Two-way construction holds sorted unique keys/inverses, edge arrays, CSR
    storage and component/diagonal work concurrently (edges <= observations).
    Generic construction can retain group indexes plus numerical-core copies.
    These byte inventories exclude caller-owned groups/weights and allocator
    overhead; they are cooperative admission estimates, not hard RSS limits.
    """
    n = len(groups[0]) if groups else 0
    levels = [int(g.max()) + 1 if len(g) else 0 for g in groups]
    if kind == "twoway":
        return int(96 * n + 64 * sum(levels))
    indexes = estimate_index_bytes(n, levels)
    return int(indexes + 24 * n * len(groups) + 32 * sum(levels) + 32 * n)


def slope_build_bytes(groups, slopes):
    """Slope copies, centered vectors, means and Gram/inverse work of slope FEs."""
    total = 0
    for codes, slope in zip(groups, slopes, strict=True):
        if slope is not None:
            width = 1 if slope.ndim == 1 else slope.shape[1]
            levels = int(codes.max()) + 1 if len(codes) else 0
            total += 8 * (len(codes) * (width + 5) + levels * (3 * width**2 + width))
    return total
