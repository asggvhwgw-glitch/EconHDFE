#!/usr/bin/env python3
"""End-to-end PPML-family parity and raw-score audit on a large three-FE corpus.

A single run is an accuracy check with timings, not a performance conclusion.
Use --repeat 3 or more for alternating same-machine timing repetitions.
The IV case is exactly identified so all raw additive moment equations apply.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import _econhdfe_within as native
from econhdfe.config import ExecutionConfig
from econhdfe.models.ppml.api import ppmlhdfe
from econhdfe.models.ppml.config import PPMLConfig
from econhdfe.models.ppml_iv.api import ivppmlhdfe
from econhdfe.models.ppml_iv.config import IVPPMLConfig


def corpus(rows, seed):
    rng = np.random.default_rng(seed)
    levels = (min(20_000, max(20, rows // 50)), min(4_000, max(10, rows // 250)), 50)
    a = rng.integers(levels[0], size=rows, dtype=np.int32)
    b = np.asarray(a % levels[1], dtype=np.int32)
    bridges = rng.random(rows) < .10
    b[bridges] = rng.integers(levels[1], size=int(bridges.sum()), dtype=np.int32)
    c = rng.integers(levels[2], size=rows, dtype=np.int32)
    C = rng.normal(size=(rows, 2))
    I = rng.normal(size=(rows, 1))
    E = .8 * I + .2 * C[:, :1] + .4 * rng.normal(size=(rows, 1))
    off = .1 * rng.normal(size=rows)
    eta = 1 + .12 * C[:, 0] - .18 * C[:, 1] + .2 * E[:, 0] + off
    groups = (a, b, c)
    for g, L in zip(groups, levels):
        eta += rng.normal(scale=.15, size=L)[g]
    # Positive continuous outcomes are allowed by PPML. Zeros/separation are
    # checked in separate native integration tests rather than this timing DGP.
    y = np.exp(eta + .2 * rng.normal(size=rows) - .02)
    w = np.exp(.6 * rng.normal(size=rows))
    return dict(y=y, C=C, E=E, I=I, off=off, w=w, groups=groups)


def run(kind, data, engine, execution):
    cfg_cls = PPMLConfig if kind == "ppml" else IVPPMLConfig
    cfg = cfg_cls(engine=engine, tolerance=1e-10, target_inner_tol=1e-12,
                  start_inner_tol=1e-5, max_iter=300, min_ok=2)
    kw = dict(y=data["y"], absorb=data["groups"], weights=data["w"],
              offset=data["off"], config=cfg, execution_config=execution, vce="robust")
    start = perf_counter()
    if kind == "ppml":
        result = ppmlhdfe(x=np.column_stack([data["C"], data["E"]]), **kw)
    else:
        result = ivppmlhdfe(exog=data["C"], endog=data["E"], instruments=data["I"],
                            weight_type="pweight", **kw)
    elapsed = perf_counter() - start
    assert result.converged
    assert result.nobs == len(data["y"])
    assert np.all(result.sample_mask)
    assert not np.any(result.separation_mask)
    resources = result.diagnostics["projection_resources"]
    if engine == "within":
        assert resources["backend"] == "within"
        assert resources["within"]["weight_generation"] >= result.iterations
    return result, dict(engine=engine, seconds=elapsed, iterations=result.iterations,
                        config=asdict(cfg), resources=resources)


def audit_scores(result, data, kind):
    y, mu, w = data["y"], result.mu, data["w"]
    score = w * (y - mu)
    scale = max(1., float(np.average(y, weights=w)))
    Q = np.column_stack([data["C"], data["E"] if kind == "ppml" else data["I"]])
    rms = np.sqrt(np.average(Q**2, axis=0, weights=w))
    moments = Q.T @ score / (w.sum() * np.maximum(rms, 1e-15) * scale)
    fe_error = 0.
    for g in data["groups"]:
        mean = np.bincount(g, weights=score) / np.bincount(g, weights=w)
        fe_error = max(fe_error, float(np.max(np.abs(mean))) / scale)
    moment_error = float(np.max(np.abs(moments)))
    assert moment_error < 1e-6, ("raw_regressor_or_instrument_score", moment_error)
    assert fe_error < 1e-6, ("raw_FE_score", fe_error)
    return dict(max_scaled_raw_score=moment_error, max_scaled_FE_score=fe_error)


def parity(a, b):
    assert a.names == b.names and a.df_absorbed == b.df_absorbed
    assert_array_equal(a.sample_mask, b.sample_mask)
    assert_array_equal(a.separation_mask, b.separation_mask)
    assert_allclose(a.params, b.params, atol=5e-8, rtol=2e-7)
    assert_allclose(a.mu, b.mu, atol=3e-7, rtol=2e-7)
    # V is O(1/N): report a relative max-norm as well as per-entry tolerances.
    assert_allclose(a.vcov, b.vcov, atol=2e-11, rtol=2e-5)
    v_error = float(np.max(np.abs(a.vcov - b.vcov))) / max(float(np.max(np.abs(b.vcov))), 1e-30)
    assert v_error < 2e-5, ("relative_vcov_max_norm", v_error)
    assert_allclose(a.deviance, b.deviance, atol=2e-7, rtol=2e-8)
    return dict(max_abs_coef_difference=float(np.max(np.abs(a.params - b.params))),
                max_scaled_mu_difference=float(np.max(np.abs(a.mu - b.mu) / (1 + np.abs(b.mu)))),
                relative_vcov_max_norm=v_error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--memory-mb", type=float, default=2048)
    parser.add_argument("--seed", type=int, default=20261027)
    args = parser.parse_args()
    if args.rows < 8000 or args.repeat < 1 or args.threads < 1:
        parser.error("rows >= 8000, repeat >= 1 and threads >= 1 are required")
    execution = ExecutionConfig(threads=args.threads, memory_budget_mb=args.memory_mb)
    execution.validate()
    small = corpus(8000, args.seed)
    for kind in ("ppml", "ivppml"):
        for engine in ("replica", "within"):
            run(kind, small, engine, execution)
    data = corpus(args.rows, args.seed)
    h = hashlib.sha256()
    for key in ("y", "C", "E", "I", "off", "w"):
        h.update(np.ascontiguousarray(data[key]).view(np.uint8))
    for g in data["groups"]:
        h.update(g.view(np.uint8))
    report = dict(arguments=vars(args), python=sys.version, platform=platform.platform(),
                  numpy=np.__version__, input_sha256=h.hexdigest(),
                  source_sha=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  extension_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
                  upstream_revision=native.UPSTREAM_REV,
                  timing_scope="complete estimator call; excludes data generation, imports, warmup and audits",
                  interpretation="single-repeat timings are not a performance claim", results=[])
    for kind in ("ppml", "ivppml"):
        for repetition in range(args.repeat):
            order = ("replica", "within") if repetition % 2 == 0 else ("within", "replica")
            estimates, records = {}, {}
            for engine in order:
                estimates[engine], records[engine] = run(kind, data, engine, execution)
                records[engine]["score_audit"] = audit_scores(estimates[engine], data, kind)
            checks = parity(estimates["within"], estimates["replica"])
            report["results"].append(dict(estimator=kind, repetition=repetition,
                                           run_order=order, routes=records, parity=checks))
            print(json.dumps({"progress": kind, "repetition": repetition, "passed": True,
                              "routes": records, "parity": checks}), file=sys.stderr, flush=True)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
