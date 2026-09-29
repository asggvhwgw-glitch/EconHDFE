# Data layer repeated-workflow checkpoint

Base: `econhdfe 0.4.10.1` development tree. No version bump.

## Scope

This checkpoint deliberately focuses on medium/large, wide, repeatedly used empirical datasets rather than extreme out-of-core workloads.

## Implemented

- immutable `EncodedEconometricDataset` with compact identifier-only encoding, one-time column signatures, packed validity metadata, lazy numeric cache and privacy-minimized source metadata;
- union-of-specifications raw-column/role compilation through `prepare_repeated_dataset()`;
- OLS-HDFE and linear-IV Sessions can consume encoded datasets directly;
- OLS-HDFE and linear-IV Sessions can consume supported DataSource/file inputs lazily;
- `fit_many()` performs one projected/encoded source materialization for the whole specification set;
- sequential DataSource-backed `fit()` expands the snapshot only when new columns/roles require it and invalidates dependent caches exactly;
- encoded FE identifiers prime the existing Session/HDFE component cache, avoiding repeated factorization of identifier-only FE columns;
- direct DataFrame and ordinary estimator behavior remain available.

## Cache hierarchy

1. source projection / role union;
2. immutable encoded columns;
3. FE component codes;
4. FE-specific absorber/sample/DoF entries;
5. FE-specific within-transformed columns.

No transformed variable is reused across a different FE/sample signature.

## Validation boundary

Targeted tests cover:

- encoded snapshot immutability;
- validity/sample-mask caching;
- role-union safety when one column is an FE in one specification and an explicit regressor in another;
- OLS and IV parity between DataFrame and encoded datasets;
- OLS and IV parity for direct CSV DataSource Sessions;
- one-materialization `fit_many()` behavior;
- exact cache invalidation when a sequential source-backed fit introduces a new column.

## Benchmark boundary

The main benchmark is the total cost of a regression-table workflow, not a raw `read_csv` microbenchmark. Current development evidence shows useful gains for wide projected sources and sequential interactive fits. The additional gain is small when `fit_many()` already batches the entire in-memory workflow, which is treated as a reason **not** to add more caching complexity.

## Stop line

Do not build persistent cache, distributed ingestion or out-of-core HDFE merely because the abstraction can support them. Add those only if real workflow measurements demonstrate that repeated source parsing or RAM capacity remains the dominant constraint.

## Final local gate

- **447 tests passed**;
- architecture map: current;
- public-contract compatibility: **0 changes**;
- technical-innovation validation: **PASS, 3 registered innovations (unchanged)**.

The separate-process RSS benchmark at 100k × 64 with 30 OLS-HDFE specifications measured about 1.34x end-to-end speedup and about 9.9% lower peak process RSS for the projected encoded source session. This is development evidence only and reinforces the design boundary that CSV projection primarily removes retained/duplicated work rather than parser scan cost.
