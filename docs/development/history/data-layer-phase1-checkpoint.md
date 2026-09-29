# Data layer phase-1/2 foundation checkpoint

Base: `econhdfe 0.4.10.1` heterogeneous-specification closeout. No version bump.

## Implemented

- `econhdfe/data/` source, planner, dataset/materialization, stable categorical encoding and estimation-sample state.
- `econhdfe/frontend/columns.py` raw-column and semantic-role discovery from economic specifications without design expansion.
- OLS-HDFE and linear-IV can consume supported file/DataSource inputs while retaining their existing signatures and dense estimator fallback paths.
- Identifier-only source columns (e.g. firm/year used only as FE/cluster IDs) are safely encoded to stable `int32`; columns that also enter an explicit factor/regressor keep their original values and reference-level semantics.
- Standard OLS/linear-IV singleton pruning is recorded in `EstimationSampleState`; the HDFE algorithm still decides exactly the same sample as before.
- Optional high-performance Stata backend hook through `pyreadstat`; pandas remains the compatibility fallback.
- Privacy-minimized ingestion profiles.
- Economics-facing architecture and module map updated.

## Validation

- 438/438 local tests passing.
- Public-contract compatibility: 0 changes.
- Technical-innovation registry unchanged; no innovation claim is attached to the data layer.

## Development evidence

- Wide CSV benchmark (120k × 64, 6 required columns): retained DataFrame payload falls by about 90.6%; parser-time improvement is material in this environment, while peak parser RSS falls much less because CSV remains row-text.
- Wide Stata/pandas fallback benchmark (100k × 48, 6 required columns): retained payload falls by 87.5%, but parser peak RSS is not improved reliably; selected-column pandas timing is environment-sensitive. This path is therefore compatibility/memory-retention evidence, not a certified Stata performance claim.
- Identifier-only encoding benchmark (500k rows, 50k firm IDs + 20 years): payload falls from about 59.6 MiB to 7.63 MiB (87.2%) with about 0.12 s encoding cost in the development environment.

## Important boundary

Integrated OLS/IV source paths still perform projected **in-memory materialization**. The data layer also provides a chunked `scan()` API and stable cross-batch encoders, but HDFE/estimators are not yet fully out-of-core. Existing missing-value semantics are unchanged.

The next stage should build a reusable encoded/columnar source cache and connect PPML/repeated-spec workflows before attempting direct batch-fed HDFE.
