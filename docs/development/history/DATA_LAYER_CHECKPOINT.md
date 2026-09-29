# econhdfe data-layer repeated-workflow checkpoint

Base package version: **0.4.10.1**. This is an internal development checkpoint, not a release.

## Objective

Optimize the common empirical case of a medium/large, wide dataset reused across many related specifications. The checkpoint does **not** target extreme out-of-core execution as its primary goal.

## Delivered

- specification-union raw-column/role planning before file materialization;
- immutable `EncodedEconometricDataset` snapshots;
- safe `int32` encoding of identifier-only FE/cluster/group columns;
- one-time column signatures and validity metadata;
- lazy reusable numeric arrays;
- OLS and linear-IV Sessions consuming encoded datasets directly;
- OLS and linear-IV Sessions consuming CSV/Stata/Parquet/DataSource inputs lazily;
- one projected/encoded materialization for `fit_many()`;
- exact source expansion + downstream cache invalidation for sequential fits that later request new columns or broaden variable roles;
- pre-seeding of existing HDFE FE-component caches from encoded identifiers;
- privacy-minimized cache/source diagnostics.

## Cache semantics

The reusable layers are source projection → encoded raw columns → FE component codes → FE-specific sample/absorber/DoF → FE-specific within-transformed columns. A change in FE structure never reuses incompatible within transforms. A change in the immutable source snapshot invalidates every downstream layer.

## What the benchmark says

The data layer is valuable when the raw source is wide and/or the researcher repeatedly changes outcomes, controls or FE specifications. It adds little when the researcher already has a narrow in-memory dataset and uses `fit_many()`, because the existing Session already batches most repeated HDFE work.

Development examples:

- 120k × 64 CSV, 40 specifications, 14 required columns: source-backed encoded `fit_many()` was about **1.76x** faster end-to-end than full-width CSV + `fit_many()` in one run, with retained encoded payload about **79%** smaller.
- Independent-process 100k × 64 CSV, 30 specifications: about **1.34x** end-to-end speedup and **9.9%** lower peak RSS. The modest RSS change is expected because CSV parsing remains row-text scanning.
- Interactive sequential fits show larger gains because immutable cached signatures/FE codes avoid repeated N-row preprocessing, while batched `fit_many()` already eliminates much of that work.

These are development measurements, not release guarantees.

## Phase 3: cross-process research-session checkpoint

A narrow persistent cache was tested because exploratory empirical work often stops and restarts between specifications. It is explicit opt-in, disposable, and limited to file-backed OLS/linear-IV sessions. It persists safely encoded columns, FE/sample/weight-signature-scoped linear within columns, and completed specification results. It does **not** persist HDFE absorber/topology objects, PPML/IV-PPML iteration states, or solver iteration vectors.

Two source-validation modes are available. `strict` hashes full source bytes and is the conservative default; at medium CSV sizes this often removes most changed-specification speedup. `metadata` uses local filesystem identity/size/mtime/ctime plus parser options and can make changed-y/add-control reuse modestly worthwhile, but is intentionally opt-in because it is not a cryptographic content certificate.

The strongest measured use case is exact completed-specification resume and partial `fit_many()` recovery. Cross-process within-column reuse is retained for sufficiently large/repeated linear-HDFE work, but the evidence does not justify a persistent FE-topology layer or a general out-of-core runtime.

Completed-result cache entries currently retain the full serializable `RegressionResult`, including observation-level fitted/residual arrays when present. Disk usage can therefore grow with the number and size of cached specifications. No eviction policy is implemented in this checkpoint; the cache remains user-disposable.

## Stop line

Do not add persistent FE topology, automatic disk caching, distributed ingestion, GPU ingestion, or fully streaming HDFE solely because the abstractions permit them. Further persistence should require real workflow evidence that cache storage/I/O is still dominated by avoided computation.

## Final validation

- **454 tests passed**;
- architecture map current;
- public contract: **0 changes**;
- technical innovation registry: **3 entries, unchanged**;
- dependency-boundary test confirms `data` does not depend on HDFE/models;
- package version unchanged.
