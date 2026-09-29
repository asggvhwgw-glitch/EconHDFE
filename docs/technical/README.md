# EconHDFE Technical Documentation

This directory is the source of truth for current mathematical/algorithmic documentation, technical boundaries, and theorem-backed project documents.

## Technical overview

- [`overview.md`](overview.md) — estimator semantics, HDFE absorption, structural compilation, data layer, execution planning, repeated workflows and validation boundaries.
- [`performance-architecture.md`](performance-architecture.md) — where large HDFE workloads spend time and why representation/reuse precede kernel optimization.
- [`identified-fixed-effects.md`](identified-fixed-effects.md) — FE recovery, connected components, normalization and identification boundaries.
- [`formal-verification.md`](formal-verification.md) — machine-checked mathematics versus software/numerical/originality claims.

## Theorem-backed project technical documents

The project currently maintains three theorem-backed technical document lines:

1. [Exact arbitrary-G categorical HDFE structural rank / absorbed DoF](hdfe/exact-multiway-dof/)
2. [Exact arbitrary-G residual-core reduction](hdfe/numerical-residual-core/)
3. [Exact partition-refinement structural design reduction](structural-design/)

These are **EconHDFE Project Technical Documentation**, not anonymous manuscripts. Their mathematical claims, implementation mappings and prior-art boundaries are kept explicit. The core mapped mathematical statements have separate Lean 4 verification under [`formal/`](../../formal/README.md).

## Mathematical verification

- [Coverage table](../../formal/mathematical-coverage.md)
- [Exact-rank structural correspondence](../../formal/rank-correspondence.md)
- [Exact-rank arithmetic correspondence](../../formal/arithmetic-correspondence.md)
- [Formal-verification appendices](../../formal/appendices/)

Machine-checked status applies to explicitly mapped mathematical statements and assumptions. It does not certify Python/Numba program correctness, floating-point convergence, benchmark performance, external-package parity, or historical originality.

## Numerical and inference documentation

- [HDFE documentation](hdfe/)
- [Structural design](structural-design/)
- [Cluster inference](cluster-inference.md)
- [Heterogeneous specification optimization](heterogeneous-specification-optimization.md)
- [Partitioned WLS](partitioned-wls.md)

## Innovation policy

- [`innovation-audit.md`](innovation-audit.md) — package-wide distinction between established methods, engineering and registered theorem-backed work.
- [`innovation-registry.json`](innovation-registry.json) — machine-readable registry and artifact mapping.

A theorem-backed result is not automatically a novelty claim. The registry continues to use `originality_status: not_independently_established` until a separate prior-art review supports a stronger statement.

## Historical reviews

Version-specific mathematical and resource reviews are retained under [`history/`](history/) for provenance. They are not the current source of truth for mathematical coverage or runtime behavior.
