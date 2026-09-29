# HDFE technical documentation

HDFE technical material is split by mathematical target. This separation is deliberate: numerical rewrites must not silently change inference conventions.

## Structural rank and absorbed degrees of freedom

`exact-multiway-dof/` contains the project technical document for exact structural DoF with three or more intercept-only categorical fixed-effect partitions. The implementation correspondence is:

```text
exact-multiway-dof/exact_multiway_hdfe_dof.{tex,pdf}
        -> econhdfe/hdfe/rank.py
        -> econhdfe/hdfe/dof.py
        -> exact-rank / DoF regression tests
        -> benchmarks/hdfe/exact_rank.json
```

The project technical document concerns characteristic-zero rank of the requested categorical FE design. It does not certify numerical convergence, heterogeneous slopes, group-individual multi-membership, or cluster finite-sample conventions.

## Exact numerical residual-core reduction

`numerical-residual-core/` contains the project technical document for the solver-side technical contribution that passes the package-wide novelty audit: exact arbitrary-G hypergraph leaf elimination for categorical HDFE projection. Its implementation correspondence is:

```text
numerical-residual-core/exact_multiway_hdfe_residual_core.{tex,pdf}
        -> econhdfe/hdfe/numerical_core.py
        -> econhdfe/hdfe/absorber.py
        -> tests/test_hdfe_multiway_solver_opt.py
        -> tests/test_hdfe_complex_fe_corpus.py
        -> benchmarks/hdfe/solver_v044_integration.json
```

The novelty claim is deliberately narrow. Degree-one pruning already appears in two-way graph-based HDFE work; the registered result is the exact arbitrary-G categorical-hypergraph projection theorem, recursive zero-residual reconstruction, and the positive/zero-weight topology conditions.

## Numerical solver implementation notes

`solver/` documents the broader numerical absorption layer introduced through the 0.4 line. Adaptive MAP/CG selection, fused weighted multi-RHS projection, buffer reuse, memory planning and backend routing are retained as engineering implementation notes rather than separate originality claims.

Numerical canonicalization/core reduction operates on execution state only. Requested FE topology remains the source of truth for DoF, nesting, reporting, and inference.

## Exact rank backends

`rank-backends.md` describes optional exact-arithmetic backends and the dependency-free certification/fallback path used by `rank.py`.
