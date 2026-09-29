# EconHDFE technical documentation

**Status:** Current project technical documentation  
**Scope:** Econometric contracts, mathematical foundations, numerical architecture, implementation correspondence, validation boundaries, and technical-claim governance.  
**Claim policy:** Mathematical correctness, software correctness, numerical evidence, and historical originality are tracked separately.

## Start here

- [Technical overview](overview.md) — end-to-end econometric and computational contract.
- [Performance architecture](performance-architecture.md) — where large HDFE workloads spend time and how repeated work is removed.
- [Identified fixed effects](identified-fixed-effects.md) — identification, normalization, singleton/separation diagnostics, and recovery boundaries.
- [Machine-checked mathematical verification](formal-verification.md) — Lean scope, theorem mapping, audit boundary, and evidence.

## Mathematical foundations

Three theorem-backed project technical documents are maintained as the current mathematical core:

1. [Exact arbitrary-G categorical HDFE structural rank / absorbed DoF](hdfe/exact-multiway-dof/)
2. [Exact arbitrary-G residual-core reduction](hdfe/numerical-residual-core/)
3. [Exact partition-refinement structural design reduction](structural-design/)

Each directory contains canonical TeX, a compiled PDF, implementation/evidence mapping, and a Lean appendix. The independent Lean library lives under [`formal/`](../../formal/README.md), with theorem-by-theorem coverage in [`formal/mathematical-coverage.md`](../../formal/mathematical-coverage.md).

Machine-checked status applies only to explicitly mapped mathematical statements and assumptions. It does not certify Python/Numba control flow, floating-point convergence, benchmark performance, external Stata parity, or historical novelty.

## Numerical and inference documentation

- [HDFE documentation](hdfe/) — exact rank/DoF, residual-core projection, solver notes, and exact-rank backends.
- [Cluster inference](cluster-inference.md) — CRV/WCR behavior and support boundaries.
- [Partitioned WLS](partitioned-wls.md) — structured least-squares design and numerical scope.
- [Heterogeneous specification optimization](heterogeneous-specification-optimization.md) — interaction-rich specification acceleration and fallback boundaries.

## Technical-claim governance

- [Innovation audit](innovation-audit.md) — distinguishes theorem-backed contributions from established methods and engineering.
- `innovation-registry.json` — machine-readable source of truth for registered technical contributions and required artifacts.

The current registry contains the same three theorem-backed lines listed above. A machine-checked theorem is **not** automatically a novelty claim. `originality_status` remains independent and must be supported by separate prior-art review.

## Historical technical reviews

Version-specific reviews and superseded technical assessments are archival material rather than current specifications. They are indexed from the documentation history after repository cleanup.
