from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from econhdfe.design import build_design, factor, reg_interaction
from econhdfe.hdfe.plan import FEPlan
from econhdfe.models.ppml.block_experimental import fit_structured_dataframe_experimental
from econhdfe.models.ppml.config import PPMLConfig
from econhdfe.models.ppml.estimator import fit_arrays


def run(*, blocks=6, per_block=12_000, local_terms=18, shared_terms=6, seed=9060):
    rng = np.random.default_rng(seed)
    n = int(blocks * per_block)
    g = np.repeat(np.arange(blocks), per_block)
    within = np.tile(np.arange(per_block), blocks)
    fe1 = g * 100_000 + within // 20
    fe2 = g * 100_000 + within % 20
    data = {"g": g, "fe1": fe1, "fe2": fe2}
    for j in range(local_terms):
        data[f"x{j}"] = rng.normal(size=n)
    for j in range(shared_terms):
        data[f"z{j}"] = rng.normal(size=n)
    df = pd.DataFrame(data)
    specs = [
        reg_interaction(factor("g", drop_base=False), f"x{j}")
        for j in range(local_terms)
    ] + [f"z{j}" for j in range(shared_terms)]

    eta = np.zeros(n)
    for j in range(local_terms):
        eta += np.linspace(-0.025, 0.03, blocks)[g] * df[f"x{j}"].to_numpy() / (j + 2)
    for j in range(shared_terms):
        eta += 0.025 * (j + 1) * df[f"z{j}"].to_numpy() / max(shared_terms, 1)
    y = rng.poisson(np.exp(np.clip(eta, -0.7, 0.7))).astype(float)
    df["y"] = y
    plan = FEPlan.from_arrays([fe1, fe2])
    cfg = PPMLConfig(
        separation=(), engine="optimized", standardize=True, max_iter=100,
        tolerance=1e-9, target_inner_tol=1e-10, fast_partial=False,
    )

    t0 = time.perf_counter()
    dense = build_design(df, specs, n, structural=False)
    dense_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    dense_result = fit_arrays(
        y, dense.values, plan, offset=np.zeros(n), true_w=np.ones(n),
        vce="model", clusters=None, names=dense.names, config=cfg,
    )
    dense_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    structured_result, execution_design = fit_structured_dataframe_experimental(
        df, y="y", x=specs, plan=plan, vce="model", config=cfg,
        structural_collinearity=False,
    )
    structured_total = time.perf_counter() - t0

    return {
        "experimental": True,
        "nobs": n,
        "ncols": int(dense.values.shape[1]),
        "blocks": int(blocks),
        "local_terms_per_block": int(local_terms),
        "shared_terms": int(shared_terms),
        "dense_build_seconds": dense_build,
        "dense_fit_seconds": dense_fit,
        "dense_total_seconds": dense_build + dense_fit,
        "structured_total_seconds": structured_total,
        "speedup_dense_total_over_structured": (dense_build + dense_fit) / structured_total,
        "dense_design_mib": dense.values.nbytes / (1024**2),
        "partitioned_payload_mib": execution_design.values.payload_bytes / (1024**2),
        "payload_reduction_fraction": 1.0 - execution_design.values.payload_bytes / dense.values.nbytes,
        "max_abs_coef_diff": float(np.max(np.abs(dense_result.params - structured_result.params))),
        "execution_reason": execution_design.execution_structure.reason,
        "detected_shared_columns": len(execution_design.execution_structure.shared_columns),
        "storage_plan": execution_design.storage_plan.as_dict(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path)
    ap.add_argument("--per-block", type=int, default=12_000)
    args = ap.parse_args()
    result = run(per_block=args.per_block)
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
