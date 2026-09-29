# HDFE technical documentation

**Status:** Current project technical documentation  
**Scope:** HDFE mathematics, projection, exact rank/DoF, numerical solver design, and exact-arithmetic backends.  
**Claim boundary:** Requested inference topology and mathematical targets remain separate from numerical execution choices.

## Exact structural rank and absorbed DoF

[`exact-multiway-dof/`](exact-multiway-dof/) contains the EconHDFE Project Technical Documentation for exact structural DoF with arbitrary finite numbers of intercept-only categorical fixed-effect partitions.

```text
exact-multiway-dof/exact_multiway_hdfe_dof.{tex,pdf}
        -> econhdfe/hdfe/rank.py
        -> econhdfe/hdfe/dof.py
        -> exact-rank / DoF tests
        -> benchmarks/hdfe/exact_rank.json
```

The mathematical core is machine-checked in Lean 4. It concerns the actual characteristic-zero rank of the requested categorical FE design; it does not certify heterogeneous slopes, group-individual multi-membership, cluster finite-sample conventions, or the production backend as formally verified software.

## Exact residual-core reduction

[`numerical-residual-core/`](numerical-residual-core/) contains the EconHDFE Project Technical Documentation for exact arbitrary-G categorical residual-core reduction.

```text
numerical-residual-core/exact_multiway_hdfe_residual_core.{tex,pdf}
        -> econhdfe/hdfe/numerical_core.py
        -> econhdfe/hdfe/absorber.py
        -> residual-core numerical tests
        -> benchmarks/hdfe/solver_v044_integration.json
```

Degree-one graph pruning is established prior art. The project document records the exact arbitrary-G projection theorem, reconstruction rule, and positive-weight validity conditions.

## Numerical solver implementation notes

[`solver/`](solver/) documents the broader absorption implementation: MAP/CG routing, fused weighted multi-RHS projection, buffer reuse, memory planning, and backend selection. These are engineering implementation notes rather than separate mathematical originality claims.

Numerical canonicalization and core reduction operate on execution state only. Requested FE topology remains the source of truth for structural DoF, nesting, reporting, and inference.

## Exact rank backends

[`rank-backends.md`](rank-backends.md) documents optional exact-arithmetic backends and the dependency-free certification/fallback path.
