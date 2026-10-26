"""External projection audit; deliberately independent of within's stop flag."""
from __future__ import annotations
import numpy as np


def audit(codes, weights, rhs, residual, oracle):
    codes = np.asarray(codes)
    weights = np.asarray(weights, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    if residual.shape != rhs.shape:
        raise ValueError(f"Residual shape {residual.shape} != RHS shape {rhs.shape}")
    if not np.all(np.isfinite(residual)):
        raise ValueError("Nonfinite residuals")
    sqrtw = np.sqrt(weights)
    normal = np.zeros(rhs.shape[1])
    oracle_normal_squared = 0.0
    for q in range(codes.shape[1]):
        code = codes[:, q]
        degree = np.bincount(code, weights=weights)
        if np.any(degree <= 0):
            raise ValueError("Expected dense observed codes and positive weights")
        oracle_sum = np.bincount(code, weights=weights * oracle)
        oracle_normal_squared += float(np.sum(oracle_sum ** 2 / degree))
        for j in range(rhs.shape[1]):
            sums = np.bincount(code, weights=weights * residual[:, j])
            normal[j] += np.sum(sums ** 2 / degree)
    denominator = np.linalg.norm(sqrtw[:, None] * rhs, axis=0)
    scaled_normal = np.sqrt(normal) / np.maximum(denominator, np.finfo(float).tiny)
    truth_norm = np.linalg.norm(sqrtw * oracle)
    if truth_norm == 0:
        raise ValueError("Oracle residual has zero norm")
    forward = np.linalg.norm(sqrtw * (residual[:, -1] - oracle)) / truth_norm
    x, y = residual[:, 1], residual[:, 0]
    bread_denom = float(x @ (weights * x))
    if bread_denom > 0:
        beta = float(x @ (weights * y) / bread_denom)
        scores = weights * x * (y - x * beta)
        cluster_scores = np.bincount(codes[:, 0], weights=scores)
        se = float(np.linalg.norm(cluster_scores) / bread_denom)
    else:
        beta, se = None, None
    return {
        "relative_scaled_normal_residual": scaled_normal.tolist(),
        "oracle_error_relative_to_residual": float(forward),
        "oracle_orthogonality_check": float(np.sqrt(oracle_normal_squared) / truth_norm),
        "fwl_beta": beta,
        "fwl_cluster_cr0_se": se,
        "projection_gate_passed": bool(np.max(scaled_normal) < 1e-8 and forward < 1e-8),
        "scope": "One exact-oracle RHS; normal equations for all RHS; CR0, no external DoF parity claim",
    }
