# Cluster inference boundary

`econhdfe` separates three concerns:

- `compute/vcov.py`: standard sandwich covariance, including one-/multi-way CRV1 and finite-sample/nesting conventions.
- `resampling/`: generic draw generation, cluster-row grouping, seeds, parallel execution and failure aggregation.
- `inference/cluster/`: post-estimation cluster diagnostics and advanced one-way OLS WCR11/WCU11 tests.

## v0.4.6 scope

Public advanced cluster inference consists of `cluster_diagnostics()` and `wild_cluster_test_ols()`. WCR11 uses restricted residuals; WCU11 uses unrestricted residuals. Each bootstrap replication is re-absorbed because cluster multipliers generally destroy FE orthogonality, then re-estimated in the same WLS metric and studentized with the package CRV1 convention.

For sufficiently small G, requested Rademacher repetitions that cover all `2**G` signs trigger exact enumeration. Otherwise Monte Carlo p-values use a finite-replication correction and never report zero.

## Coefficient-only wild bootstrap

`wild_bootstrap()` / `wild_cluster_bootstrap_ols()` return coefficient draws.
For a fixed sample, FE projector and WLS metric they can reuse cluster scores.
Write `M` for the weighted FE residual maker, `X` for the supplied design,
`B = pinv(X' W X)`, `u` for the supplied residual, and `v_g` for a cluster's
wild multiplier. Weighted self-adjointness, `M' W = W M`, gives

```text
Z       = M X
center  = B Z' W (X beta)
score_g = sum_{i in g} w_i Z_i u_i
draw    = center + B sum_g score_g v_g
```

The design is projected once, and each draw then uses only cluster-level
scores. Clusters need not nest in the fixed effects. `clusters=None` still
generates one multiplier per input row; frequency-weighted rows are not
expanded. The general center is retained because the low-level API permits
raw or imperfectly absorbed designs, rank deficiency and arbitrary `beta`.
The existing Gram pseudoinverse and its rank cutoff are unchanged.

The optimization admits the built-in float64 NumPy generic and native within
absorbers only when their weights exactly match the bootstrap WLS metric and
`0 < number of columns <= reps`. A failed one-time projection raises the
existing HDFE convergence error. The two-way Schur absorber retains the legacy
route: re-projecting an already-within design can fail its relative PCG check
when the Schur RHS is nearly zero. Custom/subclassed residualizers, non-float64
projectors, other backends, unequal metrics and wider designs also keep the
legacy per-batch absorption route.

Reordering products can amplify floating-point error in weak retained design
directions. If `||X'WX||_inf * ||B||_inf > 1/sqrt(float64 epsilon)` or is
nonfinite, the implementation keeps the legacy arithmetic. This is a
performance eligibility rule, not a new rank or omission rule. It does not
repair the existing normal-Gram numerical limitations in that fallback.

Seeds, batch dimensions/order and multiplier distributions are preserved,
including serial/parallel reproducibility. Memory admission charges the
whole one-time design projection (all columns, even when wider than one draw
batch), scores and retained draws, then budgets cluster-sized worker arrays.
Fallback admission is recomputed before dispatch if the condition screen
rejects the score route. If the one-time design projection does not fit the
budget, feasible narrow legacy batches remain available; a call is rejected
only when its selected route cannot be admitted.

This identity does not remove the per-draw residuals and covariance needed by
`wild_cluster_test_ols()`; its WCR11/WCU11 studentization is unchanged.
Independent dense dummy-SVD and legacy arithmetic checks are in
`tests/numerics/test_wild_score_bootstrap.py`. The reproducible warm benchmark
is `benchmarks/resampling/wild_bootstrap_scores.py`.

## Deliberate exclusions

The Library CRV3 prototype is not public in v0.4.6. It performs full delete-cluster HDFE refits and requires raw outcome/design state in addition to within-transformed state. Before integration it needs a large-data planner, bounded memory/state semantics, weighted/group-individual coverage and external parity. Multi-way WCB, IV WCB and PPML score/bootstrap inference are also separate future work.

## Multiway covariance repair (all estimators)

The signed Cameron-Gelbach-Miller multiway sandwich can be indefinite. OLS,
linear IV/GMM, the linear sessions, PPML and IV-PPML (dense and block paths)
form it without repair and then call `compute.vcov.finalize_cluster_vcov`,
which follows reghdfe 6.12.5 / ppmlhdfe 2.3.0: negative eigenvalues are clipped
in the standardized coordinates of the original regressors, i.e. after scaling
`V` by `s s'`, where `s` is the unweighted sample standard deviation (N-1) of
each regressor on the estimation sample before FE demeaning, floored at 1e-3.
A reported constant has scale 1; when present, the full matrix is clipped and
its slope block is replaced by the separately clipped slope block
(`reghdfe_fix_psd`). Collinear omitted columns do not participate, and the
zero-variance `_cons` posted by IV-PPML is appended afterwards. The repair is
thus independent of user units and of every internal solver scaling or
conditioning basis. Up to 0.10.0 the clip used reported (or internal) units,
which can differ from reghdfe by orders of magnitude when regressor scales
differ; see the 0.10.1 changelog. Under `singleton_mode="stata_compat"` the
scale is reghdfe's unweighted stdev of the stored rows, so a compact
frequency-weighted fit and its expanded copy can clip differently. Under
`"unified"` with fweights the scale is the frequency-weighted stdev
(`sum f` observations, N-1 = `sum f - 1`), identical to the expanded data;
pweights/aweights are never used for the scale
(`SamplePolicy.psd_frequency_scale`).

ivreg2/ivreghdfe do not repair by default (only with `psd0`/`psda`); econhdfe
keeps the repair for linear IV/GMM with the reghdfe coordinates. The
first-stage robust F diagnostics keep their repair in the max-abs-equilibrated
coordinates they already use. PPML still has an unrelated, registered
small-sample constant difference from ppmlhdfe.

The Poisson regression reference uses explicit weighted FE dummy projection and
independently aggregated cluster scores. It covers a genuinely indefinite
CGM covariance, frequency-weight expansion, standardization on/off, and
dense/block PPML and IV-PPML paths. IV-PPML keeps its per-subset G/(G-1)
correction; PPML keeps its existing N, absorbed-DF and nesting correction.
