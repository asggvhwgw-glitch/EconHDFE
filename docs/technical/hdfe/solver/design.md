# HDFE numerical solver design

**Status:** Current HDFE numerical-solver design  
**Scope:** Requested topology, canonical numerical plans, projection cores, solver routes, and execution-state invariants.  
**Claim boundary:** Solver optimizations may change execution representation but must not redefine requested inference topology.
The numerical solver is a distinct layer from structural DoF/rank. Its job is to compute the within transformation efficiently while preserving the column space requested by the estimator.

## Three representations

1. **Requested/inference topology**: the FE terms supplied by the user. This representation governs absorbed DoF, nesting, reporting, and compatibility semantics.
2. **Canonical numerical plan**: algebraically equivalent FE representation chosen for efficient execution.
3. **Transient numerical residual core**: optional row/level reduction used only during projection when degree-one structure proves some residual rows exactly zero.

Only layers 2 and 3 are solver optimizations. They must not rewrite layer 1.

## Main execution paths

- specialized two-way paths where their structural assumptions hold;
- generic MAP/CG absorption for arbitrary categorical FE systems;
- adaptive `acceleration="auto"` in optimized FE plans;
- weighted multi-RHS grouped projection for large pooled residualization;
- optional 3+ FE numerical core reduction for eligible pure-intercept NumPy systems.

## Numerical core reduction

For an intercept-only categorical FE level appearing in exactly one active observation, orthogonality to that dummy requires the residual at the corresponding row to be zero. Recursively peeling such rows can reduce the iterative system without changing the exact within result. Unlike structural-rank logic, this operation never deduplicates observations: each row carries its own RHS and weight.

The topology can be reused across strictly positive weight updates. Zero weights change the effective numerical graph and invalidate that reuse.

## Adaptive MAP/CG

`acceleration="auto"` performs real symmetric-MAP sweeps on the actual RHS block and estimates contraction from those iterates. Fast-contraction systems continue plain MAP; difficult systems switch to CG from the partially residualized state. Direct low-level `HDFEAbsorber` retains the conservative historical default `acceleration="cg"`; optimized `FEPlan` may select `auto` internally.

## Memory rule

Core reduction may allocate the reduced residual-core block, but `copy=False` must reuse the caller/workspace output buffer rather than materializing a second full `N x RHS` result. Memory planning is therefore based on the full workspace plus bounded reduced-core and grouped-projection scratch.
