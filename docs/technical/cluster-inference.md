# Cluster inference boundary

> **Document status:** Current EconHDFE project technical documentation.


`econhdfe` separates three concerns:

- `compute/vcov.py`: standard sandwich covariance, including one-/multi-way CRV1 and finite-sample/nesting conventions.
- `resampling/`: generic draw generation, cluster-row grouping, seeds, parallel execution and failure aggregation.
- `inference/cluster/`: post-estimation cluster diagnostics and advanced one-way OLS WCR11/WCU11 tests.

## v0.4.6 scope

Public advanced cluster inference consists of `cluster_diagnostics()` and `wild_cluster_test_ols()`. WCR11 uses restricted residuals; WCU11 uses unrestricted residuals. Each bootstrap replication is re-absorbed because cluster multipliers generally destroy FE orthogonality, then re-estimated in the same WLS metric and studentized with the package CRV1 convention.

For sufficiently small G, requested Rademacher repetitions that cover all `2**G` signs trigger exact enumeration. Otherwise Monte Carlo p-values use a finite-replication correction and never report zero.

## Deliberate exclusions

The Library CRV3 prototype is not public in v0.4.6. It performs full delete-cluster HDFE refits and requires raw outcome/design state in addition to within-transformed state. Before integration it needs a large-data planner, bounded memory/state semantics, weighted/group-individual coverage and external parity. Multi-way WCB, IV WCB and PPML score/bootstrap inference are also separate future work.
