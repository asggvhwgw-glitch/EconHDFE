# Multiway HDFE solver-opt2 integration (v0.4.4)

> **Document status:** Historical solver integration record for v0.4.4; not the current source of truth.


Version 0.4.4 integrated the solver-opt2 module into the shared HDFE layer without changing estimator equations, result schemas, DoF semantics, configuration dataclasses, or structured error codes.

## Innovation boundary

The package-wide technical-innovation audit classifies only the **exact arbitrary-G residual-core reduction theorem** as a solver-side technical innovation. Its formal manuscript is `../numerical-residual-core/exact_multiway_hdfe_residual_core.{tex,pdf}`.

The remaining solver-opt2 features in this note—adaptive MAP/CG routing, fused multi-RHS kernels, output-buffer reuse, memory budgeting and runtime planner choices—are engineering optimizations around established numerical methods. They are important performance work but are not advertised as new mathematical or econometric contributions. Degree-one pruning itself is also prior art in two-way graph-based HDFE solvers; the formal paper limits the claim to the exact arbitrary-G categorical-hypergraph generalization and reconstruction theorem.

The main additions were:

- numerical degree-one peeling/residual-core reduction for eligible 3+ FE pure-intercept systems;
- adaptive two-sweep MAP/CG planning through `acceleration="auto"`;
- fused weighted multi-RHS projection;
- positive-weight numerical-core topology reuse with zero-weight invalidation;
- caller/workspace output-buffer reuse on the core path to avoid an additional full `N x RHS` allocation.

The direct `HDFEAbsorber` default remains `acceleration="cg"`. The additive advanced controls are `core_reduction`, `core_min_peel_fraction`, `auto_plain_limit`, `auto_polish_limit`, and `fused_rhs_memory_mb`.

Correctness is guarded by parity tests against the unreduced solver, weighted reference projections, exact-zero peeled-row checks, in-place/copy semantics, weight lifecycle tests, adaptive routing tests, and complex stable/mobility FE corpora.

Performance is workload dependent. Leaf-rich and large multi-RHS weighted systems can benefit substantially; irreducible mobility systems should be expected to remain near parity. See `benchmarks.md` and `benchmarks/hdfe/solver_v044_integration.json`.
