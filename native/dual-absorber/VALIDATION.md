# Dual absorber v0.1: validation and performance boundaries

Date: 2026-10-29. Status: **numerically validated experimental adapter; auto routing is not production-ready**.

## Implemented scope

The independent `econhdfe_dual` package exposes `DualAbsorber` with explicit `within`, `fereg` and experimental `auto` choices. It calls pinned Rust solvers, not a new independent reimplementation of fereg. Within uses `Solver::solve_batch`; fereg uses its fused residualization and stateful `set_weights`. Both return original-order residuals through the same scaling, certification and error/fallback interface. Within retains its Design across numerical weight updates. Fereg retains its structural state and lets upstream policy decide eligible numerical-factor reuse. FE coefficient warm starts are not used.

No existing EconHDFE estimator API, default, singleton/separation rule, rank/DoF or inference policy was changed. This prototype is not merged into main. Categorical intercept FE and strictly positive finite weights only. Heterogeneous slopes, zero-weight sample handling and production PPML/IV-PPML integration remain outside this version.

## Source and evidence

- Tested runtime code: `ca26258eb5f575805ba33fd0a03b9315be922ac1`.
- Baseline: unchanged `native/within-accel` from EconHDFE `44dae3f41f978a3ce78e0d780feab42b49a352f3`, using `NativeSolver(strategy="auto")`.
- within revision: `fc44282cea6124f5f18bb142908605a656d35a7c`.
- fereg-core revision: `0fa59e3b539addbddaee8bd71acd031c497166a0`.
- Build, unit tests and first complete cases: [run 40813893262](https://github.com/asggvhwgw-glitch/EconHDFE/actions/runs/40813893262).
- Remaining large cases and complete evidence validation: [run 40815059605](https://github.com/asggvhwgw-glitch/EconHDFE/actions/runs/40815059605).
- Complete artifact: `dual-absorber-prototype-validated-40815059605`, artifact ID `12872091657`.
- Artifact SHA-256: `4f555ac0f96fee5655cb2a5d703300a2f8ef8aa8a53453431e046beb13e1f584`.
- Prototype wheel SHA-256: `82c6d3eac6669d92bd680cc9c1381d62598b9c4dec3a4b92f8fa62fef9f648e4`.

The initial run timed out during a slow serial four-method 5M comparison, not because its recorded projections failed their numerical gates. The completion workflow re-ran the three remaining cases independently with the **same binary**, checked source SHAs and joined only complete case JSONs. No slow case was dropped and no numerical tolerance was relaxed. Subsequent documentation and CI-budget-only changes do not change the tested runtime code; the serial benchmark timeout is now 900s rather than 300s.

## Validation results

- **47 tests passed**, zero failures, errors or skips.
- **16 large configurations x 4 paths = 64 benchmark records**, all valid.
- **128 projection rounds**: first solve and reuse, including changed-weight cases.
- Maximum relative weighted forward error against the known residual: **2.2376386453408764e-11**, below the common `2e-8` benchmark acceptance threshold.
- Maximum original-row group-mean certificate for the new adapter: **2.460192843799378e-11**, below the requested `1e-10` gate.
- 36 of 96 new-adapter rounds used a corrective projection. Kernel fallback is disabled in the performance grid, so every explicit backend must pass independently. Failover is tested separately, including an intentionally iteration-limited within solve followed by an exact fereg solve.

Coverage includes dense SVD/lstsq residual oracles; weighted/unweighted 1/2/8 RHS; changed weights; invalid-update rollback; sparse labels; non-contiguous inputs; unit scaling; constant/zero/empty RHS; nested and duplicate FE; disconnected components; singletons without sample deletion; forced blocking; failed convergence; fallback; and stable initialized routes.

A common test-only safeguarded **PPML IRLS loop on 1,200 observations** is compared with a full-dummy Newton/WLS oracle under each backend. It checks coefficients, fitted means and raw scores. This is not production PPML/IV-PPML integration, covariance parity or a large end-to-end estimator benchmark.

## Numerical issue found and corrected

On a million-row sparse-group case, the core's own convergence flag did not imply the adapter's maximum weighted group-mean criterion. Repeating projection at the same inner tolerance left some groups above the gate. The corrective projection now tightens the inner tolerance in proportion to the observed violation, while keeping the external requested gate unchanged. `tests/test_sparse_group_gate.py` retains that regression.

The original-row criterion is a backward/stationarity certificate, not a condition-number-independent forward-error theorem. Large paired fixtures provide a separate analytically known residual for checking forward accuracy.

## Timing scope and benchmark limitations

Times are seconds. **Cold = constructor + first numerical weight update + first residualization**, including lazy kernel setup, copies, row reordering, scaling and certification. Reuse includes the second projection and any changed-weight setup.

All large fixtures use **positive heterogeneous weights and paired FE tuples**. Each complete tuple occurs at least twice, with residuals `e,-e` and equal weights inside the pair; the repeated case uses 16 rows per base tuple. This makes the true weighted residual known. It also favors compression: these are not fully unique-row data or a representative survey of all economic designs. Earlier unique-row/unweighted speed rankings need not carry over.

All four methods in a configuration run on the same runner with two solver threads and one BLAS thread. Initial cases used an Intel Xeon Platinum 8573C runner; resumed large cases used AMD EPYC 7763 or 9V45. Compare methods within a row, not absolute times across differently hosted rows. Only two rounds and fixed method order were used; no timing confidence interval is claimed. Reported process peak RSS is cumulative across methods, not isolated per-method memory.

## One million observations, eight RHS

| Case | Existing auto cold | Forced fereg cold | Prototype auto cold | Existing reuse | Prototype auto reuse | Auto route |
|---|---:|---:|---:|---:|---:|---|
| one_big | 0.665 | 0.441 | 0.446 | 0.493 | 0.384 | fereg |
| two_big | 1.419 | 0.830 | 0.756 | 1.226 | 0.655 | fereg |
| three_big | 17.287 | 1.882 | 14.325 | 15.821 | 14.651 | within |
| weak | 2.463 | 0.788 | 0.778 | 1.927 | 0.725 | fereg |
| repeated | 0.459 | 0.470 | 0.475 | 0.401 | 0.412 | fereg |
| reweight | 1.389 | 0.825 | 0.813 | 1.262 | 0.730 | fereg |

## Five million observations, two RHS

| Case | Existing auto cold | Forced fereg cold | Prototype auto cold | Existing reuse | Prototype auto reuse | Auto route |
|---|---:|---:|---:|---:|---:|---|
| one_big | 1.712 | 1.268 | 1.275 | 0.784 | 0.865 | fereg |
| three_big | 31.659 | 6.005 | 35.774 | 22.541 | 26.813 | within |
| weak | 9.615 | 1.986 | 1.921 | 4.294 | 1.507 | fereg |
| repeated | 1.262 | 1.486 | 1.462 | 0.613 | 0.977 | fereg |

## Integration decision

**The adapter is working; the chooser is not ready to replace production auto.**

Measured cold gains include about 3.17x for the 1M/eight-RHS weak network and 5.01x for the 5M/two-RHS weak network. One-big, two-big/eight-RHS and reweight/eight-RHS also improve. These are conditional gains, not universal speedups.

Important regressions are retained:

- Three-big weighted paired data can be badly misrouted: at 1M/eight RHS, auto takes 14.325s while forced fereg takes 1.882s; at 5M/two RHS, auto takes 35.774s while forced fereg takes 6.005s.
- Existing compressed-within is better on the 5M repeated case: cold 1.262s versus new auto 1.462s; reuse 0.613s versus 0.977s.
- For 1M/two-RHS two-big, legacy cold/reuse are 0.374s/0.282s versus new auto 0.625s/0.578s. Reweight is 0.398s/0.330s versus 0.667s/0.644s.
- The small duplicate-sample hint is not a population U/N estimator and can miss paired duplicates entirely. Size counts alone do not capture compression or weighted spectral difficulty.

Before integration, preserve three execution routes: **batched within, existing compressed-within, and fereg reduced PCG**. Improve admission using compression information, weights, measured convergence/setup costs and RHS count. Add mostly unique-row fixtures, independent large reference solves, multiple seeds and production estimator/inference parity. Do not fit thresholds only to this paired-oracle corpus.

## Usage

```sh
python -m pip install ./native/dual-absorber
python -m pytest native/dual-absorber/tests -q
```

```python
from econhdfe_dual import DualAbsorber

# codes: integer (n, FE dimensions); Y: (n, RHS); w: strictly positive.
solver = DualAbsorber(codes, weights=w, backend="fereg", threads=2)
residual, info = solver.residualize(Y, tol=1e-10, return_info=True)
solver.update_weights(w_next)
residual_next = solver.residualize(Y_next, tol=1e-10)
```

Explicit backends are recommended for controlled experiments. The package is separately installed; current estimator defaults are unchanged. `scratch_bytes` controls estimated batching, not a hard RSS or factor-fill bound. The supplied binary was validated on Linux x86-64; Windows/macOS builds have not been certified here.
