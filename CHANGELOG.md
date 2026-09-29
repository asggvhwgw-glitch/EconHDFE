# 0.6.2 — nonlinear and resource-boundary validation candidate

## Unreleased — IV correctness closeout

- Reject deficient IV cross moments with `UnderidentifiedError` and numerically
  unsafe moment systems or solver rank loss with `NumericalError`. This is a
  numerical identification policy, not a weak-instrument F-test.
- Use a compressed instrument-coordinate QR/SVD solve for sensitive 2SLS
  systems; do not silently truncate projected coefficient directions.
- Report weighted first-stage fitted endogenous values in unweighted within
  coordinates consistently across dense, block and two-step GMM paths.
- Fixed k-class covariance now uses H=(1-kappa)X+kappa PzX scores; kappa=0
  matches OLS. This treats kappa as fixed, retains conventional LIML inference,
  and does not assert consistency for arbitrary kappa with endogenous X.
- Avoid a redundant joint instrument materialization for exactly identified
  block overidentification diagnostics. End-to-end memory budgeting and the
  full-score KP fallback remain open; this is not an out-of-core implementation.

## 0.6.5 — unreleased

- Reorganize tests around current contracts, behavior, numerical oracles and release
  tooling; replace redundant historical smoke checks and reduce repeated parameter
  combinations without relaxing retained tolerances. **Plain pytest now selects
  the daily core; use `scripts/run_tests.py --suite full -- -q` for full acceptance.**
- Align installed-wheel test selection, active technical-registry references, and
  source-archive validation with the nested layout. Avoid duplicate branch-push/PR
  matrices while keeping all 16 full-suite CI environments.
- No estimator formula, runtime API, numerical tolerance, result schema or runtime
  dependency changes are part of this test-system release.

## 0.6.4 — development baseline

- Bound native modular/rational exact-rank fallback with shared per-component
  work accounting, conservative sparse-storage admission and pre-multiplication
  integer-growth guards. Resource exhaustion fails explicitly; no finite-field
  lower bound is returned as exact. Optional external calls and process RSS/time
  are not covered by these cooperative guards.
- Bound `equilibrated_lstsq` validation/scaling scratch by the existing row-block
  budget and divide directly into the private QR workspace. QR/SVD, coefficient
  units, rank thresholds, public APIs and estimator formulas are unchanged.
- Add independent rational/SVD, layout/scale, allocation and resource-failure
  guards plus an isolated-process memory/timing comparison driver.
- Reconcile the roadmap with the published 0.6.3 release; reset 0.6.4 execution
  evidence. Cross-platform, clean-install and final release acceptance are
  separate checks, not implied by this entry.


## 0.6.3 — local candidate, 2026-09-14

- Fuse NumPy weighted/unweighted column products and row reductions in the shared
  CG absorber, avoiding full N-by-RHS product temporaries. GPU and projection
  buffer implementations are unchanged; no solver, tolerance or VCE changes.
- Add independent layout, dtype, empty-array, extreme-scale, dynamic-weight and
  near-stopping-boundary regression guards. Retain all earlier test files.
- Add an execution-evidence gate distinct from maintenance review, a unified
  TODO, a full-functional-test thread preflight and detached formal authorization.
- Candidate builds may retain truthful blocked/not_run checks. Formal release
  mode refuses them and checks final artifact hashes without rebuilding files.
- This is not cross-platform, clean-install, GPU or external-reference certification.


- Reuse final IV-PPML solver bread in dense/block VCE; correct original-unit
  first-stage diagnostics and coordinate-dependent divergence checks.
- Keep initial mean/eta consistent with offset; protect constant-offset-shift
  invariance in dense and block nonlinear-IV paths.
- Factorize observed cluster labels before DoF/VCE in PPML/IV-PPML. Reject every
  single-cluster dimension rather than dropping it or silently using robust VCE.
- Report IV-PPML legacy model/IID aliases as their actual robust convention;
  suppress confidence intervals for the normalized, zero-SE `_cons` coordinate.
- Balance block PPML covariance and use common dimensionless rank/bread in block
  2SLS. Validate integer iteration budgets; bound dense-code certification
  allocations by sample size instead of the largest raw label.
- Reduce QR temporary allocations without replacing stable QR/SVD with raw
  normal equations. Add independent nonlinear roots, explicit weighted-GLM
  references, mathematical fallback/resource guards and local benchmark tooling.
- Freeze compatibility and generated architecture state before BOTH source
  distribution and wheel builds. Verify selected independent numerical suites
  against the installed wheel outside the source tree using host dependencies.
- Current release remains a local candidate, with clean dependency installation,
  remote platform matrix and licensed external comparisons explicitly deferred.

# 0.6.1 — correctness and mathematical-review candidate

## Numerical and inference corrections

- Balance dense design columns before rank decisions; retain exact original
  dependencies across numerical FE residualization. Use common QR/SVD factors
  for dense OLS/WLS coefficients, covariance bread and effective rank.
- Balance 2SLS/LIML/k-class/GMM2S roles and restore original coefficient,
  covariance, first-stage and GMM metadata units. Balance the augmented LIML
  outcome/endogenous design before generalized-eigenvalue calculations.
- Correct the PPML model/IID covariance y-standardization factor in both engines
  and the heterogeneous path; robust/cluster sandwiches keep their own scaling.
- Reject one-cluster inference (including small_sample=False). Invalid t
  degrees of freedom no longer silently use a normal reference distribution.
- Reject non-finite recovery targets/weights and lossy/noncontiguous dense codes.
  Respect LSMR/LSQR condition-limit and iteration-limit stop codes; preserve raw
  stop diagnostics. Fix residual-core Hestenes post-convergence checking.
- Replace FE recovery's private two-way solver coupling with an ordered
  coefficient-recovery method.

## Mathematical review

- Review all 24 formal results across the three technical manuscripts under
  explicit assumptions; add an independent Fraction-elimination oracle and
  exhaustive tests of all 4,095 nonempty 2x2x3 supports.
- Clarify finite-field lower bounds versus characteristic-zero rank and generic
  sparsity rank; positive-weight residual-core scope; functional-dependency
  graph cycles/direction; active/reference columns; common slope multipliers;
  fitted-space versus named-coefficient invariance; and row-deletion semantics.
- Revise LaTeX/PDF manuscripts and classify the three entries as theorem-backed
  frameworks/applications, without certifying novelty or publication priority.

## Release engineering and compatibility

- Extend compatibility snapshots to the existing `econhdfe.effects` namespace.
  Append defaulted stop diagnostics; preserve top-level estimator signatures.
- Add Python 3.10–3.13 / Linux-Windows-macOS CI configuration, standard sdist
  manifests, clean-venv dependency-install validation, contributing/security
  guidance, and exact SHA256 manifest coverage checks. No remote publication.
- Retain an explicit offline diagnostic build mode; host-dependency smoke is not
  described as clean installation. See test-status for actual local limitations.
- The numerical safeguards add measured dense-kernel overhead; the small
  same-host probe is recorded rather than claiming universal performance parity.

# 0.6.0

Version 0.6.0 adds a thin, estimator-agnostic identified fixed-effect recovery layer for additive categorical/indicator fixed effects. Existing OLS-HDFE, linear-IV-HDFE, PPML-HDFE, IV-PPML-HDFE estimators and their top-level signatures are unchanged; recovery is exposed through the separate `econhdfe.effects` module.

## Identified fixed-effect recovery

- Added level-sized recovery of categorical intercept fixed effects from an already estimated additive FE contribution `g = D gamma`.
- Reuses the existing two-way Schur solver and exact categorical rank engine; generic K-way recovery uses matrix-free LSMR.
- Adds raw-level mapping, connected-component metadata, exact rank/nullity, ordinary component-shift nullity, and `extra_nullity` diagnostics.
- Adds canonical, reference, mean-zero, and observation-mass-weighted mean-zero reporting normalizations without re-estimating the underlying model.
- Normalization is solver-independent: Schur/LSMR decompositions are transformed to a common reporting convention while preserving `D gamma`.
- Results are stored by FE level rather than by observation; observation-sized FE contribution arrays are not retained by default.

## Estimator adapters and identification diagnostics

- Added thin post-estimation adapters for OLS/linear IV and PPML/IV-PPML. PPML recovery uses the converged linear predictor and final IRLS mass.
- Added strict identification diagnostics on the realized estimation sample, including disconnected support, nesting/redundancy, recursive-singleton pruning, realized-data extra rank deficiency, and PPML separation/finite-MLE support loss.
- Additional null-space directions are diagnosed component-by-component. Fully identified independent components are recovered normally; FE levels in unidentified components are returned as unavailable (`NaN`) with the causal diagnostic retained. A specification with no identified component raises `FixedEffectIdentificationError`. The package does not manufacture arbitrary restrictions to force uniqueness.
- Pure categorical interactions such as `firm#year` are in scope. Continuous/varying-slope FE recovery such as `firm#c.age`, FE standard errors, leave-out corrections, and structural constraint solving remain explicit non-goals for this release.
- Advanced structural workflows may inspect an unresolved decomposition with `strict_identification=False`; a richer user-constraint interface is reserved for later work.

## Compatibility

- Automated public-contract diff against v0.5.0 remains empty for the established top-level estimator/result/config surface.
- The new functionality lives under `econhdfe.effects`; estimator equations, statistical defaults, cache/planner behavior, and existing result objects are unchanged.

# 0.5.0

Version 0.5.0 is the architecture/performance milestone that integrates the heterogeneous-specification execution layer, estimator-agnostic econometric data layer, repeated-workflow encoded datasets, unified execution planner, real-runtime automatic thread calibration, and privacy-minimized planner feedback. The estimator equations and top-level public estimator signatures remain backward compatible with the canonical v0.4 baseline 0.4.10.2; the release changes how large and repeated empirical workloads are represented, ingested, resumed, planned, and executed.

## Privacy-safe support reporting

- Forward-ported the complete 0.4.10.2 support-report layer onto the 0.5 architecture line: `econhdfe.support_reports`, `econhdfe-report error|benchmark`, packaged canonical templates, isolated-wheel support-report smoke tests, and release-verifier identity checks.
- Error payload/report generation remains strict allowlist based and parameter-only; it does not serialize empirical data, coefficient/fitted/residual arrays, variable identifiers, arbitrary error strings, paths, commands, logs or full traceback text.
- The canonical v0.4 compatibility baseline for 0.5.0 is now 0.4.10.2 rather than 0.4.10.1.

## Runtime calibration and developer feedback

- Added cached real-runtime synthetic calibration for `threads="auto"` on sufficiently large HDFE workloads; small jobs stay single-threaded to avoid one-time calibration and parallel overhead.
- Added anonymous calibration IDs/timing curves to planner contracts without changing public estimator signatures or statistical semantics.
- Added privacy-minimized planner/performance developer reports in Markdown/JSON plus a standalone skill helper; reports never upload automatically and exclude data values, variable names, paths, commands, hostnames, logs and tracebacks.
- Added alternating multi-round real-runtime thread validation and release/skill governance checks for the planner report template.
- Added a conservative small-mixed-design dense fast path after canonical v0.4.10.2 A/B profiling showed that topology setup could dominate a one-shot complex event-study whose structurally surviving design was only six columns. The shortcut applies only to small, low-pass, memory-safe mixed designs with at least two globally supported controls and at most four localized columns; genuinely block-separable/localized and wider heterogeneous designs continue through the exact block planner.

## Post-architecture validation

- Completed cross-check consistency/performance validation after the Data Layer and unified Execution Planner refactors; no estimator/statistical semantics changed.
- Added deterministic pre-/post-planner model snapshots covering OLS, IV, PPML and IV-PPML with robust and multiway-cluster inference; results are exactly identical on the validation matrix.
- Added generic four-FE, heterogeneous-design, complex event-study, planner-overhead and representation-operator performance evidence.
- Repaired the complex-event-study benchmark harness to use the current DataFrame/DataSource input contract.
- The production storage policy remains conservative, with one narrow release hardening: small low-pass mixed designs can bypass topology analysis when the dense payload is already cheap and the localized column count is too small to justify setup. Wider or genuinely localized designs still use the exact block planner.

## Unified execution planner

- Added an estimator-agnostic `econhdfe.planner` layer with explicit exactness certificates, memory envelopes, physical-representation candidates, parallel budgets and a composite explainable `ExecutionPlan`.
- Moved canonical CPU/cgroup memory resource discovery into `planner.resources`; `compute.runtime` remains a compatibility re-export.
- Data ingestion and HDFE workspace planning now consume the same runtime memory envelope instead of maintaining unrelated budget logic.
- HDFE automatic thread resolution consumes the shared parallel contract, while already-resolved integer thread counts stay on a constant-time hot-kernel path.
- Dense/block design planning now validates block execution through the shared exact-certificate contract. The established conservative one-shot/repeated-pass production policy is intentionally unchanged pending hardware/operator calibration.
- Partitioned WLS border/fallback memory checks now respect the same runtime memory envelope.
- Added planner-specific governance tests: the planner cannot import data/compute/HDFE/IV/model implementations, and an uncertified candidate can never win because it is cheaper.
- Development representation benchmarks confirm that structural byte savings alone do not map one-for-one into runtime speedups, especially as BLAS thread count changes. The byte-traffic cost model therefore remains advisory in this checkpoint.
- No estimator equations, statistical defaults, public estimator signatures, or technical-innovation registry entries changed.

## Repeated-workflow data reuse

- Added immutable `EncodedEconometricDataset` snapshots for repeated empirical specifications: identifier-only FE/cluster columns are encoded once, column validity/content signatures are computed once, and numeric arrays are reused across session fits.
- Added union-of-specification role planning so a regression table can project the complete required column set before parsing a wide source. A column used as an explicit regressor/factor in any specification is never silently treated as identifier-only.
- `OLSHDFESession` and `IVHDFESession` can consume encoded datasets or supported DataSource/file inputs without signature changes. `fit_many()` materializes the projected/encoded union once; sequential fits rebuild only when a newly requested column/role requires it.
- Encoded FE identifiers seed the existing HDFE component cache; FE/sample/within caches remain exact and are invalidated when the immutable source snapshot changes.
- Workflow benchmarks now compare complete regression-table execution rather than parser-only timings. The checkpoint explicitly records that `fit_many()` already captures much of the repeated-specification benefit, so no duplicate cache hierarchy is added.
- At this repeated-workflow checkpoint no persistent cache was introduced; the later persistent-session checkpoint below adds a narrow opt-in cross-process cache. Distributed ingestion and out-of-core HDFE remain out of scope.

## Persistent research sessions

- Added an opt-in, disposable persistent cache for cross-process linear-HDFE research workflows.
- Reuses safe encoded source columns and FE/sample/weight-signature-scoped within transformations across later OLS/linear-IV sessions.
- Completed OLS/linear-IV specifications can resume without rerunning HDFE/model estimation; `fit_many()` skips completed specifications before prewarming unfinished work.
- Persistent cache files use versioned JSON + NumPy arrays and atomic replacement; no absorber/result pickle is used.
- Conservative `strict` complete-source validation is the default; optional `metadata` validation is an explicit local-workflow speed/assurance tradeoff.
- The evidence does not support persisting FE absorber topology, PPML/IV-PPML iterative states, or enabling cross-process caching automatically.

## Econometric data layer

- Added an estimator-agnostic `econhdfe.data` layer for projected/batched DataFrame, CSV, Stata and optional Parquet access, stable categorical encoding across batches, bounded-memory ingestion planning, and monotone estimation-sample state.
- Added specification-first raw-column discovery: factor variables, interactions, absorbed FE/slopes, IV roles, clusters, weights and related metadata can be reduced to the exact raw source columns needed before parsing a wide file.
- Added semantic role tracking on those raw-column requirements. Identifier-only FE/cluster/group columns can be pre-encoded to stable `int32`, while columns also used as explicit factors/regressors retain their original level semantics.
- Standard OLS/linear-IV singleton pruning now records a monotone sample-state audit trail without changing the established HDFE sample rule.
- OLS-HDFE and linear-IV public signatures are unchanged but may now consume a supported file/DataSource input; only required columns are materialized before entering the established estimator pipeline. Existing in-memory DataFrame behavior is unchanged.
- Ingestion profiling is privacy-minimized by default: counts, source type, batch/workspace sizes and timings are recorded, but raw column names and file paths are not placed in estimator profiles.
- Missing-value/drop-sample semantics are intentionally unchanged in phase 1. `EstimationSampleState` is infrastructure for a later unified sample contract, not a new implicit cleaning policy.
- Local validation at checkpoint: 438 tests passing, zero public-contract changes, innovation registry unchanged. Promoted as part of the 0.5.0 architecture release.

## Heterogeneous-specification execution

- Consolidated the pre-materialization heterogeneous-specification optimization layer for common interaction-rich designs such as factor-by-continuous group slopes, event-study/cohort interactions, and the same structures with a small set of shared controls.
- Integrated the structured execution path into DataFrame OLS, linear IV (2SLS/LIML/k-class), PPML-HDFE, and IV-PPML while retaining the existing dense path as the automatic correctness fallback.
- Kept ordinary narrow specifications on the original dense path; the optimizer is syntax- and certificate-gated and does not change estimator equations or public call signatures.
- Added block-aware collinearity, cross-products, robust/multi-way-cluster score aggregation, weighted 2SLS, PPML/IV-PPML standardization, and component FE projection primitives.
- Preserved global PPML/IV-PPML convergence and inference semantics. Shared-coefficient specifications requiring global simplex/ReLU separation fall back to dense execution rather than applying an invalid component-local separation test.
- Kept group+individual FE, GMM2S, HAC/Driscoll-Kraay, unsupported FE-slope topologies, incompatible row partitions, and numerically unsafe partitioned solves on established dense/specialized paths.
- Linear-IV primary estimation is structured, but established weak-IV diagnostics currently materialize role designs once; this deliberately limits the current memory/speed benefit instead of introducing a second diagnostic implementation in this checkpoint.
- Added `docs/technical/heterogeneous-specification-optimization.md` and a four-model benchmark harness. Block-angular QR/TSQR remains classified as established numerical adaptation/engineering optimization; the technical-innovation registry remains unchanged at three items.
- Checkpoint validation: 419 tests passing, zero public-contract changes, technical-innovation gate passing with three registered innovations. A repository test now requires every runtime Python module to be documented by economic/econometric purpose, not only by computational role. Promoted as part of the 0.5.0 architecture release.

# 0.4.10.2

- Added `econhdfe.support_reports` and the `econhdfe-report error|benchmark` console helper for privacy-safe support/report template generation.
- Added strict allowlist-based in-process parameter-only error reporting.
- Packaged byte-identical error/benchmark templates inside the wheel and added release verification for template identity and the console entry point.
- Added support-report unit and isolated-wheel smoke tests. Estimator equations, numerical kernels and the public estimator Python contract were unchanged.

# 0.4.10.1

- Replaced the standard error/mismatch report with a privacy-minimized, **parameter-only** contract. The report no longer requests raw/sampled/synthetic observations, minimal reproductions, exact commands/scripts, raw logs, full tracebacks, file paths, or attachments.
- Error reports now use anonymous model-role aliases (`y`, `x1`, `fe1`, `cluster1`) and collect only environment/runtime parameters, specification counts/structure, DoF/solver settings, structured error code/stage fields, aggregate counters/timings, and parity-difference magnitudes.
- Removed raw-data/script/benchmark consent fields from the error template. Real-data benchmarking remains a completely separate explicit-consent workflow governed by the benchmark template.
- Added release-contract checks requiring repository/Skill template identity and the privacy-minimized parameter-only markers while rejecting legacy reproduction/attachment sections. Runtime estimators, public API/schema/defaults, numerical algorithms, dependencies, and performance behavior are unchanged.

# 0.4.10

- Unified `fv()` with the linear HDFE `absorb=` path: the same symbolic factor-variable expression can now target explicit regressor design or absorbed FE compilation depending on model role.
- Added absorb-context compilation for categorical FEs, categorical interactions, slope-only heterogeneous FEs, intercept-plus-slope FEs, grouped multi-slope forms, and higher-order full factorials.
- Added exact symbolic reduction of redundant lower-order absorbed full-factorial terms before numerical FE encoding/DoF, preventing duplicated heterogeneous-slope components.
- Added structured absorb-context errors for meaningless base selectors, pure continuous absorb terms, and unsupported products containing multiple continuous atoms.
- Repeated-spec sessions now accept categorical-only `fv()` absorb expressions while explicitly rejecting heterogeneous-slope `fv()` terms.
- Added direct/manual parity tests for single FE, joint FE, categorical full factorial, slope-only, intercept+slope, higher-order full factorial, scalar `absorb=fv(...)`, and repeated-session categorical absorption.

# 0.4.9

- Added the Python-safe `fv()` factor-variable DSL for OLS/linear-IV DataFrame specifications. Use `i(x)` / `c(x)` rather than ambiguous dot-style `i.x` / `c.x`.
- Added Stata-like `#` interaction-only and `##` full-factorial expansion, grouped `+` expressions, `i(x,y)` prefix distribution, continuous powers, and up to eight-way interactions.
- Added reference-level controls `base=<value|first|last|freq|none>` plus quoted column/base names.
- `fv()` compiles into the existing `Factor` / `RegressorInteraction` design engine; no second numerical design path was introduced.
- Added exact design-matrix and OLS/linear-IV parity tests against the existing manual factor/interactions API plus structured syntax errors for rejected dot notation.

# 0.4.8.1

- Added a canonical structured error/mismatch report template for crashes, numerical anomalies, reference-package parity failures, convergence problems, regressions, and API/documentation defects.
- Added the byte-identical error-report template to the standalone Agent Skill and routed material failures to it.
- Kept raw-data inspection, replication-script access, and performance benchmarking as separate explicit-consent decisions; attaching a file does not grant any of them.
- Added release checks/tests that require repository/Skill template identity. Runtime estimators, public API/schema/defaults, numerical algorithms, dependencies, and performance behavior are unchanged.

# 0.4.8

- Audited secondary-but-publication-relevant model statistics against current `reghdfe`, `ivreghdfe`/`ivreg2`, and `ppmlhdfe` conventions. The audit distinguishes a true definition bug from missing stored results and from estimator-specific convention differences.
- Fixed PPML deviance reporting under internal outcome standardization: the final deviance is now restored to the original outcome scale, matching `ppmlhdfe` rather than remaining in standardized-y units.
- Added standard OLS fit statistics to `RegressionResult`: RSS, centered/uncentered TSS as appropriate, within TSS, MSS, RMSE, Gaussian log likelihood/null log likelihood, model F/Prob>F, model DoF, fit residual DoF, and covariance-matrix rank.
- Added the unambiguous partialled-out linear-IV ancillary statistics: RSS, within TSS, RMSE, model F/Prob>F, model DoF, fit residual DoF, and covariance rank. Existing econhdfe overall IV R-squared remains an explicitly documented extension rather than being relabelled as `ivreghdfe` parity.
- Added PPML null log likelihood, pseudo-R2, model Wald chi-square/p-value, explicit model DoF, full pre-separation N, and covariance rank.
- Kept `rank` backward compatible as the active design/model rank and added `vcov_rank` for the covariance rank corresponding to `reghdfe`/`ppmlhdfe` stored-result semantics.
- Expanded the testing/beta Agent Skill policy: every consequential run/table should randomly cross-check at least one representative specification against a mature reference implementation when available.
- Added an explicit-consent gate for real-user-data benchmarks and a canonical, self-contained `BENCHMARK_REPORT_TEMPLATE.md` that users can return directly to developers without sharing raw data.
- Added fweight physical-replication tests for the new ancillary linear fit statistics, PPML standardization/deviance parity tests, repeated-session parity, and benchmark/Skill contract tests.

# 0.4.7

- Corrected OLS/linear-IV adjusted R-squared DoF semantics to match `reghdfe`: clustered `df_resid` remains the t/F reference DoF, while adjusted fit statistics use a separate HDFE fit residual DoF.
- FE dimensions nested in cluster variables are charged in the adjusted-R-squared denominator even though they remain excluded from CRV small-sample DoF penalties.
- Added `RegressionResult.r2_adjusted_within` and report all four `reghdfe`-style R-squared measures.
- Unified direct OLS, linear IV and repeated-spec sessions on one `reghdfe_r2_statistics()` implementation.
- Added parity tests for clustered vs unclustered fit statistics, nested FE, linear IV, repeated sessions, slope/no-intercept helper semantics, and fweight physical-replication equivalence.

# 0.4.6.1 — package-wide technical-innovation audit and manuscript governance

- Completed a package-wide novelty audit that distinguishes genuine mathematical/algorithmic contributions from established econometric methods, compatibility implementations, and engineering optimizations.
- Registered exactly three current technical innovations: exact arbitrary-G categorical FE structural rank/DoF; exact arbitrary-G numerical residual-core projection reduction; and exact partition-refinement HDFE canonicalization/structural design reduction.
- Added formal LaTeX/PDF manuscripts for the numerical residual-core theorem and the structural-design reduction framework; retained the existing exact multiway-DoF manuscript as the first registered paper.
- Added `docs/technical/innovation-registry.json` plus `innovation-audit.md`; each registered innovation maps to its manuscript, implementation, tests and benchmark/validation evidence.
- Added `scripts/technical_innovation.py` and wired it into the strict version/release gates. Registered technical innovations must ship `.tex + .pdf`; source/bundle copies are hash-identical and manuscripts remain excluded from the wheel.
- Explicitly classified PPML/IV-PPML/SPJ, cluster WCR/WCU/CRV1, group+individual FE, standard MAP/CG/LSMR/LSQR methods, two-way Schur/PCG, caching, parallelism, fused kernels, runtime planning, Agent Skill and release tooling as established methods or engineering rather than package originality claims.
- No estimator equation, runtime numerical path, public Python contract, config/result schema, error code or dependency changes.

# 0.4.6 — advanced one-way cluster inference and weighted bootstrap repair

- Fixed the legacy OLS `wild_bootstrap()` weighted-refit bug: bootstrap coefficient draws now use the same WLS metric as the fitted model for generic/aweight/pweight/fweight execution rather than refitting pseudo-outcomes with unweighted least squares.
- Added canonical cluster normalization under `compute/clusters.py`; cluster labels are normalized once into dense codes for inference consumers without duplicating estimator-specific factorization logic.
- Added `cluster_diagnostics()` with per-dimension cluster counts/sizes, effective cluster count by size, score-concentration diagnostics for OLS, heuristic review flags, and pairwise intersection counts for multi-way cluster designs.
- Added one-way OLS-HDFE `wild_cluster_test_ols()` implementing studentized WCR11/WCU11 tests with scalar or joint restrictions, Rademacher/Mammen/Webb/normal multipliers, finite-Monte-Carlo p-value correction, and exact full Rademacher enumeration when the cluster count is sufficiently small.
- Added an explicit `econhdfe.inference.cluster` layer. Standard CRV1/multi-way covariance remains in `compute/vcov.py`; generic draw mechanics remain in `resampling`; advanced cluster inference no longer belongs in the OLS model module.
- CRV3/cluster-jackknife is deliberately not promoted into the public API in 0.4.6. The Library prototype requires full delete-cluster HDFE refits and raw-state duplication, which is not yet aligned with econhdfe's large-data memory/execution design.
- Added independent WCR11 exact-enumeration parity against explicit OLS, weighted aweight/pweight/fweight observed-stat parity, weighted coefficient-bootstrap WLS-oracle checks, multi-way diagnostics coverage, and deterministic serial/parallel WCB tests.
- Public-contract expansion is intentional and compatibility-approved; existing estimator signatures remain backward compatible aside from additive optional bootstrap controls.

# 0.4.5 — PPML execution-policy and separation-path repair

- Fixed a PPML/IV-PPML execution-policy bug: direct and reusable nonlinear APIs now propagate `ExecutionConfig.threads` and `memory_budget_mb` into the weighted HDFE projector instead of merely validating the config.
- Added resource-aware PPML projector construction and bounded reusable projector caching through `ExecutionContext`; y/sample-specific separation projectors remain short-lived to avoid stale topology and retained-memory growth.
- Removed the simplex separation hard-coded `engine="replica"` path. Optimized PPML now uses the optimized FE plan for simplex residualization, while zero simplex weights safely fall back from the specialized two-way path to the generic HDFE projector.
- Optimized ReLU separation now reuses a prepared optimized MAP/HDFE absorber instead of unconditionally constructing the replica-style LSMR path. Replica mode retains the LSMR behavior.
- Added per-method separation timings, iteration counts, selected solver labels, and actual weighted-projector resource diagnostics to PPML and IV-PPML diagnostic dictionaries.
- Archived the user-supplied 2026-09-12 real-machine OLS/IV/PPML/complex-FE benchmark verbatim under `benchmarks/real_world/2026-09-12/`, with SHA-256 provenance. PPML figures are intentionally labelled pre-0.4.5 because the source diagnosis proved the requested thread/memory policy was not actually controlling the projector.
- Added regression coverage for direct/reusable resource propagation, optimized separation routing, zero-weight two-way fallback, IV-PPML resource propagation, replica/optimized separation parity, and immutable real-world benchmark provenance.
- No estimator public signature, result dataclass schema, config dataclass schema, structured-error code, or default separation method set changes in 0.4.5.

# 0.4.4.4

- Refined architecture-map edge routing without changing the runtime/API contract.
- Replaced center-style dependency curves with deterministic per-side port allocation and rounded orthogonal corridors.
- Added terminal gaps so open-chevron arrowheads remain visible outside node cards; long reverse edges now approach the target along the target-side normal.
- Added edge halos for crossing separation and distinct visual density for Overview, full AST dependencies, and selected execution flows.
- Added regression coverage for routing primitives, terminal gaps, open-chevron markers, and generated-map freshness.

# 0.4.4.3

- Refined the generated architecture visualization without changing the runtime/API contract.
- Architecture nodes now declare safe card dimensions and balanced one/two-line labels, preventing SVG title overflow.
- Dependency curves attach to node boundaries instead of card centers, reducing line-through-text artifacts.
- Added visual layer bands and a sparse semantic `Overview` as the default interactive mode; the complete AST graph remains available on demand.
- Hardened the detail pane for long paths and added regression tests for node bounds, title wrapping and visual-generation semantics.

# Changelog

## 0.4.4.2 — architecture visualization tooling

- Added an evidence-backed architecture-map workflow derived from Python AST imports rather than hand-maintained dependency drawings.
- Added deterministic `architecture.json`, GitHub/Mermaid `architecture.md`, and a self-contained interactive `architecture.html` under `docs/development/architecture-map/`.
- Added `skills/econhdfe/references/architecture-visualization.md` plus a portable `architecture_map.py` helper and repository wrapper with `--check` stale-map detection.
- Kept import topology distinct from reviewed execution-flow overlays; package `__init__.py` re-export imports do not inflate collapsed dependency counts.
- No estimator, numerical, result, config, error, or public Python API behavior changed.

## 0.4.4.1 — technical-documentation and release-layout normalization

- Introduced the PEP 440-compatible `MAJOR.MINOR.PATCH[.REVISION]` policy. Maintenance-only changes and extremely localized internal fixes can use sequential fourth-component revisions; REVISION releases are blocked from carrying public-contract diffs.
- Added `scripts/version.py next revision|patch|minor|major` and transition validation; the previously prepared documentation-layout artifact was reclassified from 0.4.5 to 0.4.4.1 before external publication.
- Promoted the exact 3+ FE structural-DoF manuscript into the repository under `docs/technical/hdfe/exact-multiway-dof/`, with both canonical LaTeX source and compiled PDF.
- Added a dedicated HDFE technical index and numerical-solver design notes that explicitly separate requested/inference topology, canonical numerical plans, and transient residual cores.
- Moved architecture, benchmark, validation, migration, versioning, and release-governance documents out of the repository root into responsibility-based `docs/development/` and `docs/release/` directories.
- Moved HDFE exact-rank and solver-opt2 machine-readable benchmark evidence into `benchmarks/hdfe/`. New benchmark evidence should use domain subdirectories rather than adding root-level artifacts.
- Release assembly now preserves the nested documentation/benchmark layout and hashes every bundled file recursively. Bundle verification requires the exact-DoF `.tex/.pdf` in source and release artifacts, verifies source/bundle identity, and rejects technical manuscript files from wheels.
- Added repository-layout regression coverage so legacy root documentation files and HDFE technical evidence cannot silently drift back to ad-hoc locations.
- No estimator equation, numerical solver, public API, result/config schema, structured-error catalog, or runtime dependency changes in 0.4.4.1.

## 0.4.4 — multiway HDFE solver-opt2 integration

- Integrated the latest `econhdfe-hdfe-solver-opt2` overlay into the shared HDFE layer. Eligible NumPy 3+ pure-intercept FE systems can use exact degree-one numerical core reduction, solving only the residual hypergraph core and returning exact zero residuals on peeled rows.
- Added `acceleration="auto"` to `HDFEAbsorber`: two real symmetric-MAP sweeps both warm-start the solve and estimate contraction; easy systems continue plain MAP while hard spectra continue with CG. The direct absorber default remains `acceleration="cg"` for compatibility.
- `FEPlan` optimized generic multiway absorption now uses the adaptive planner instead of unconditional CG; the specialized two-way path is unchanged.
- Added a fused weighted multi-RHS group projection with a bounded RHS memory budget; indexed projection remains available and is retained where it is preferable.
- Positive weight updates reuse numerical-core topology. Zero weights disable core reduction because they change the effective numerical topology.
- Adapted the core path to reuse the caller/workspace output buffer rather than allocating another full `N x RHS` result matrix; only the reduced core input is materialized.
- Added the solver-opt2 complex-FE corpus covering stable and mobility panels with individual/year/city/industry and interaction FE combinations. Public requested-vs-effective inference topology and exact requested DoF remain separate from numerical canonicalization.
- `HDFEAbsorber` gains backward-compatible advanced controls: `core_reduction`, `core_min_peel_fraction`, `auto_plain_limit`, `auto_polish_limit`, and `fused_rhs_memory_mb`. The public-contract change is explicitly approved in the compatibility gate.
- Local integration validation passes 248 tests. Performance evidence is recorded in `benchmarks/hdfe/solver_v044_integration.json`; direct solver gains are workload-dependent and complex public-estimator timings remain near parity on irreducible mobility designs.

## 0.4.3 — portable Agent Skill package and public-contract compatibility gate

- Replaced the duplicated monolithic root/`skill/SKILL.md` files with one canonical `skills/econhdfe/` Agent Skill using progressive disclosure: a 54-line router plus focused installation, configuration, empirical-research, advanced-validation, and third-party-development references.
- Added `skills/econhdfe/agents/openai.yaml` and deterministic skill helpers for environment inspection and installed-package OLS/IV/PPML smoke testing.
- Release assembly now emits a standalone `econhdfe-skill-v0.4.3.zip`; bundle verification checks byte-level tree identity against the canonical source skill.
- Added automated skill validation for Agent Skills naming/frontmatter, direct reference routing, OpenAI UI metadata, local-link integrity, script syntax, and prohibition of the legacy duplicate SKILL entrypoints.
- Added `scripts/compatibility.py` and versioned public-contract baselines. The release gate now compares public exports/signatures, result/config dataclass schemas, structured-error codes/stages, and exported dataclass schemas against the previous stable release.
- Public-contract changes require an explicit change-id approval with a non-empty reason; unapproved changes and stale approvals block release. The release build freezes the current contract snapshot for the next version.
- No estimator equations, public estimator signatures, result schemas, config defaults, or error codes changed in 0.4.3.

## 0.4.2 — release lifecycle and structured boundary maintenance

- Added `docs/release/maintenance.json` and `scripts/release_maintenance.py` as a machine-readable closeout contract covering frontend/API, backend, errors, skill, linked modules, results/config/cache compatibility, tests, performance, documentation, packaging, external validation and final artifacts.
- `scripts/version.py bump` now requires a monotone version increase and resets every maintenance area to `pending`; `scripts/version.py gate` requires explicit evidence for every area.
- `scripts/build_release.sh` now invokes the strict gate, and bundle verification requires the completed maintenance manifest to match the source archive.
- Closed a v0.4.1 public-boundary gap in repeated-spec sessions: malformed session specs now raise structured `SpecificationError` with code `specification.session_spec`, and session execution paths are protected by the public error boundary.
- Invalid `preflight_dataframe(..., level=...)` values now raise structured code `specification.preflight_level` at the frontend stage.
- No estimator equations, numerical HDFE algorithms, default DoF methods, or publication-output formulas changed.

## 0.4.1 — repeated-spec execution for linear HDFE workflows

- Added `OLSHDFESession` / `IVHDFESession` and `OLSSpec` / `IVSpec` for repeated empirical specifications.
- Batch paths partial out the union of newly requested y/X/IV columns once per FE specification and cache within-transformed columns for subsequent specifications.
- Treat FE additions/removals as first-class cache keys: each FE combination has an independent compiled absorber/sample/DoF/within bank; returning to a previous FE combination avoids another HDFE transform.
- Reuse dense factorization of FE component columns across FE combinations while preserving FE-specific singleton pruning and canonicalization.
- Added signature-safe cache invalidation for FE, cluster, weight, and numeric source-column mutation; advanced immutable-data workloads may explicitly use `ExecutionConfig(cache_validation="none")`.
- Added machine-executable release version control (`docs/release/versioning.md`, `scripts/version.py check/bump`) and an early version-consistency gate in the release builder.
- Deliberately rejected a cross-FE residual seed shortcut after alternating A/B benchmarks showed only ~3% median benefit with unstable single-run direction; v0.4.1 keeps the simpler combination-level cache.
- 300k OLS/cluster benchmark with default signature validation: 1.38x for multiple outcomes, 1.93x for adding controls, 1.14x for one-time new FE combinations, and 1.88x when three FE combinations are toggled across eight table columns.
- Local regression suite: 218 / 218 passing.

## 0.4.0 — exact multiway HDFE rank and inference-topology separation

- Added the exactdof2 categorical-rank engine for intercept-only 3+ HDFE designs.
- Added duplicate-edge reduction, exact degree-one peeling, connected-core decomposition, proper-connectivity certification, and bounded GF(2) bitset certification.
- Added optional exact backends (`sympy`, `flint`) plus dependency-free modular/rational fallbacks; no floating-point rank tolerance is used.
- Added opt-in `dof_method="exact"` to HDFE/OLS/IV and PPML/IV-PPML configuration while preserving `pairwise` as the compatibility default.
- Separated requested/inference FE topology from canonicalized solver topology so FE canonicalization cannot change absorbed DoF or cluster nesting.
- Added public `categorical_rank`, `categorical_prefix_ranks`, `available_rank_backends`, and rank diagnostics.
- Added exact-rank regression coverage for all 255 nonempty 2x2x2 three-way hypergraphs, randomized 3/4-way designs, solver-canonicalization invariance, PPML, and IV-PPML.
- Local regression suite: 211 / 211 passing.

## 0.3.5 — publication output, advanced configuration and cache lifecycle

- Added a common publication-ready result surface for OLS, linear IV, PPML and IV-PPML: coefficient/SE/test/p-value/CI/stars tables plus model-level N, DoF, VCE, cluster and FE metadata.
- Added overall/within R-squared for linear models and retained log-likelihood/deviance for PPML families; no ad-hoc PPML pseudo-R2 convention is introduced.
- Linear-IV publication output includes reportable first-stage information and core identification/over-identification tests while keeping full diagnostics opt-in.
- Added strategy-level `HDFEConfig`, `InferenceConfig`, and `ExecutionConfig`; implementation-detail solver constants remain private.
- Added optional execution profiling and an internal `ExecutionContext` for reusable cache/resource lifecycle.
- Added content fingerprints to `FEPlan`; reusable DataFrame PPML/IV-PPML models invalidate stale FE topology when source FE columns change.
- Added strict-but-advisory preflight warnings for few clusters, near-constant variables, extreme weights and extreme Poisson zero shares.
- Preserved default estimator equations and external IV-PPML certification gates. Local regression suite: 202 / 202 passing.


## 0.3.0 — frontend, structured errors and resampling infrastructure

- Added a stable structured error hierarchy with machine-readable `code`, `stage`, `details`, and optional `suggestion`; public estimator boundaries translate expected low-level failures without hiding programming errors.
- Preserved backward compatibility by keeping input/specification errors as `ValueError` subclasses, convergence/inference failures as `RuntimeError` subclasses, and structured linear-algebra failures as `numpy.linalg.LinAlgError` subclasses.
- Added `frontend` variable-role validation and public `preflight_dataframe()` for cheap deterministic schema checks: nonnumeric continuous roles, missing/nonfinite values, Poisson outcome/exposure constraints, and missing FE/cluster identifiers.
- Promoted generic bootstrap execution to top-level `resampling`: deterministic seed handling, reusable cluster row groups, parallel execution, and failure-code aggregation. Model-specific bootstrap statistics remain in their model modules.
- Refactored OLS wild bootstrap into `models.ols_bootstrap` and IV-PPML SPJ bootstrap to consume shared resampling infrastructure; bootstrap failures are categorized and programming errors are no longer silently skipped.
- Kept all estimator numerical equations unchanged. Local regression suite: 195 / 195 passing.

## 0.2.0 — unified HDFE econometrics release

- Promoted the consolidated `econhdfe` architecture to 0.2.0 with OLS-HDFE, linear IV-HDFE, PPML-HDFE and IV-PPML-HDFE under one shared HDFE/compute stack.
- Includes the completed local IV-PPML parity work from the 0.1.0a4 development line: Stata/Mata-style weighted standardization, `fweight`/`pweight` semantics, in-loop `mu` separation, defensive convergence guards, Class A/B/C SPJ bias correction, and cluster bootstrap inference.
- Keeps licensed-Stata feature goldens and exact upstream Class A/B/C binary-corpus comparisons as explicit external certification gates; 0.2.0 does not claim those unavailable gates passed.
- Deliberately leaves new 3+ FE solver/DoF research and IV-PPML-specific weak-identification statistics out of scope.
- Local regression suite before release packaging: 187 / 187 passing.

## 0.1.0a4 — IV-PPML parity, bias correction and bootstrap

- Removed the previously experimental cross-IRLS two-way PCG initial-state/fallback heuristic after profiling showed no material benefit; the shared solver is stateless again.
- Added `ivppmlhdfe` as a first-class model that combines Poisson IRLS, generic weighted 2SLS and the shared weighted-HDFE core without coupling generic `iv` to an outcome model.
- Matched upstream IV-PPML semantics for raw `pweight`, frequency-weight effective N/meat, Stata/Mata-style weighted `quadvariance` scaling, optional in-loop `mu` separation and loud nonconvergence/runaway guards.
- Added deterministic licensed-Stata feature-golden and official Class A/B/C validation harnesses under `validation/ivppml/`; actual Stata/corpus certification remains an external gate.
- Added Class A/B/C split-panel jackknife bias correction and model-specific cluster bootstrap inference; bootstrap row groups are compiled once and failed replicates are explicitly counted/skipped.
- Bootstrap reports draw SD, percentile CI and CI-implied SE; SPJ/bootstrap logic remains outside generic `iv`.
- Deliberately did not reuse linear KP/SW/Stock-Yogo diagnostics for IV-PPML because current upstream theory/software does not define a justified nonlinear analogue.
- Local suite: 187 tests passing.
- Current-container IV-PPML baselines inherited from this development line: 100k two-way replica/optimized 1.45s/0.25s (~5.76x), 1M 20.26s/4.50s (~4.51x), true Class-C-style 3FE 2.82s/2.74s (~1.03x).

## 0.1.0a3 — PPML large-scale engineering pass

- Kept the 3+ FE estimator theory/solver unchanged; this release is an execution and memory optimization pass only.
- Split parallel elementwise `exp` from deterministic Poisson-deviance reduction after 10M repeated-IRLS profiling exposed unstable long-tail runtimes in a fused scalar reduction.
- Reused persistent IRLS vectors and the residualized `N x (K+1)` block across outer iterations; fast-partial updates now inject only the working-response delta.
- Reused one compiled weighted FE projector through post-FE rank checking, IRLS, and the final exact solve whenever the estimation sample and effective FE plan are unchanged.
- Compiled two-way topology with one unique-edge pass instead of COO->CSR plus a second unique sort; dynamic weights now aggregate observation weights once to edges and then update node denominators from edge weights.
- Added guarded previous-IRLS Schur/PCG warm starts; stale starts automatically fall back to zero if they worsen the initial residual materially.
- Made intermediate fast WLS normal equations bounded-memory and added reusable residual workspaces.
- Streamed PPML bread/robust/cluster VCE paths without materializing a full `N x K` score matrix.
- Avoided duplicate post-absorption rank checks when separation leaves the sample unchanged, and avoided full-sample separation copies until a method actually removes observations.
- Added cross-specification PPML warm starts through `PPMLHDFE.fit(..., warm_start=previous_result)`.
- Fixed runtime discovery when Numba was imported before `econhdfe` and `NUMBA_NUM_THREADS` was not explicitly set.
- Local suite: 163 tests passing.
- Current-container benchmarks: 1M two-way replica/optimized 15.09s/4.52s (3.34x); 5M optimized 19.23s at ~1.36 GiB RSS; 10M optimized 45.75s at ~2.50 GiB RSS; 144k true-3FE gravity 1.27x; 240k hierarchy canonicalized to 2FE 2.32x.


## 0.1.0a2 — outcome-agnostic IV architecture

- Split generic IV infrastructure from the linear-IV outcome model so future IV-PPML does not depend on 2SLS/LIML model code.
- Moved linear `ivhdfe` orchestration, k-class/GMM estimators, linear weak-ID diagnostics and Stock–Yogo tables to `models.linear_iv`.
- Added generic `iv.IVDesign`, additive IV score/moment primitives, and a reusable weighted-2SLS inner solver.
- Routed the current linear 2SLS estimator and the first step of two-step GMM through the same weighted-2SLS core intended for future iteratively reweighted IV-PPML.
- Strengthened architecture tests: generic IV cannot import HDFE or outcome models; PPML and linear-IV models cannot depend on each other.
- Local suite: 159 tests passing.

## 0.1.0a1 — architecture alpha

- Renamed the distribution and implementation namespace from `pyreghdfe` to `econhdfe` to reflect the broader OLS/PPML/IV scope.
- Added primary estimator names `olshdfe`, `ppmlhdfe`, and `ivhdfe`.
- Retained `reghdfe`, `ivreghdfe`, and the `pyreghdfe` namespace as backward-compatible aliases/shims.
- Reorganized implementation into `models`, `hdfe`, `iv`, and `compute` layers with one-way dependency boundaries.
- Split OLS WLS primitives from IV estimators; IV diagnostics and Stock–Yogo data now live under the IV layer.
- Moved FE compiler/absorbers/DoF/weighted projection into a dedicated HDFE layer.
- Moved numerical kernels, runtime planning, weights, covariance and bootstrap execution into `compute`.
- Added architecture-boundary tests; combined local suite now passes 155 tests.

# 0.8.0

- Hardened the release artifact itself: added BSD-3-Clause licensing, README/package metadata, isolated wheel smoke validation, reproducible source/bundle assembly, SHA256 generation, and source-vs-wheel consistency verification.
- Expanded the external Stata golden harness from 18 to 23 models for hierarchy canonicalization, auto two-way routing, explicit omission, structural collinearity, and event-study references; the comparator now checks covariance, N, absorbed DoF and available weak-ID diagnostics as well as coefficients/SEs.
- Fixed `run_local_validation.sh` by implementing a real `bench_cpu_projection.py --quick` mode and making the benchmark runnable from a clean source checkout without relying on an installed package.
- Added a modular runtime/execution layer: cgroup/affinity-aware CPU discovery, pre-Numba worker-pool sizing, memory-headroom-aware reusable RHS workspace planning, and execution telemetry.
- Preserved the econometric solver layer: resource planning changes execution width/threads only and does not rewrite the model specification.
- Added dense integer-code reuse, NumPy-design zero-copy when no columns are omitted, and exact high-cardinality cluster-meat correction to reduce large-data memory pressure.
- `absorb_info["execution_plan"]` reports the resolved workspace mode, pool width, thread count and memory context; the public `pool_size` metadata now reports the actual execution-plan width.
- Final local suite: 115 tests passed.
- Final 10M complex interaction/event-study benchmark: ~19.60 s, ~2.82 GiB peak RSS, 12 two-way iterations, maximum identified-parameter error ~2.96e-4.
- Generic 10M four-FE sustained-memory performance is left as an explicit target-hardware acceptance item because the 4 GiB cgroup showed unstable sustained memory-bound runtime after repeated 10M stress phases. No single-environment kernel patch was added.

# 0.7.0.dev5

- Added explicit user-controlled omission selectors: `omit_column()`, `omit_term()`, and `omit_level()`; IV exposes role-specific `omit_exog=`, `omit_endog=`, and `omit_instruments=`.
- Added `RegressionResult.user_omitted_variables` and `automatic_omitted_variables` so deliberate references are separated from automatic FE/rank omissions. Unmatched omission selectors raise instead of silently doing nothing.
- Added event-study reference support and regression tests showing an explicit `event_time=-1` omission is parameterization-equivalent to using the corresponding factor base.
- Added a shared component-code cache so repeated categorical components such as `year` are factorized once and reused across many interaction FEs.
- Added multi-RHS two-way PCG so multiple outcome/regressor/instrument columns share Schur-operator traversals instead of invoking independent SciPy CG solves.
- Reduced large-sample memory use by avoiding an unnecessary active-X copy when rank resolution keeps every column and by streaming cluster-score aggregation instead of materializing an N×K score matrix.
- Added `benchmarks/bench_10m_complex_eventstudy.py`: 10M rows, ~556k firms, nested geographic×year FEs, structural collinearity, explicit event-study reference, deliberately composite collinearity and two-way clustering completed in ~21.1 s with ~2.90 GiB peak RSS on the constrained release container. The largest identified-parameter error in the simulation was ~2.96e-4.
- Full local regression suite: 110 passed.

# Legacy pyreghdfe 0.7.0.dev4

- Added a component-level structural dependency DAG with incremental transitive closure. Exact categorical dependencies are certified once and reused across higher-order factor/interaction terms.
- Hierarchical relations such as `city -> province -> region` now infer `city -> region` without an additional full-data mapping scan; direct and closure-derived proof edges are exposed separately.
- Changed the public collinearity default from silent `drop` to `warn`. Omitted regressors/instruments now emit `OmittedVariableWarning` by default while estimation continues on the identified basis.
- Kept `collinearity="raise"` for strict specification validation and `collinearity="drop"` as an explicit quiet opt-in.
- Added `RegressionResult.omitted_variables` and `has_omitted_variables` for direct auditing.
- Expanded the agent skill with a semantic specification preflight for FE nesting, factor/interaction redundancy, IV role conflicts and solver planning.
- Added dependency-DAG and omission-reporting regression tests; full suite: 106 passed.
- Added `benchmarks/bench_dependency_dag.py`; on the release container a 2M-row four-level hierarchy was planned in about 0.20 s using 3 direct full-data checks plus 3 transitive inferences.

# Legacy pyreghdfe 0.7.0.dev3

- Added exact structural collinearity planning before dense design materialization.
- Added `design_structure.py` partition-dependency engine for factor and interaction blocks.
- Pure categorical regressor blocks spanned by absorbed FEs are now omitted before HDFE residualization.
- Full factor interactions are basis-reduced against absorbed coarse FEs (for example full `qob#year` after absorbing `year`).
- Added exact order-preserving structural reductions among regressor blocks sharing the same continuous monomial.
- IV endogenous/excluded-instrument designs can use included exogenous terms as a protected structural basis.
- Added `structural_collinearity=True|False` for parity/diagnostic comparisons.
- Design compilation now occurs on the final singleton-pruned estimation sample in standard HDFE mode.
- Existing public omission categories are preserved while richer `structural_reason` and proof metadata are exposed.
- Test suite: 100 passed.

# Legacy pyreghdfe 0.7.0.dev2

- Make `method="auto"` the public default; solver selection occurs after FE canonicalization.
- Auto-select the specialized two-way Schur/PCG solver for exactly two effective pure categorical intercept FEs; otherwise use symmetric MAP + CG.
- Add explicit regressor frontend terms `factor()` and `reg_interaction()` with expanded-column provenance.
- Add order-preserving post-HDFE collinearity resolution using small weighted Gram matrices and stable Gram-coordinate orthogonalization.
- Distinguish FE-absorbed/zero columns, duplicate/scaled columns and general linear-combination redundancy.
- Add role-aware IV collinearity handling: included exog are resolved first; endogenous regressors are resolved conditional on included exog; excluded instruments are resolved conditional on included exog before first-stage diagnostics.
- Add `collinearity_info` to results and `collinearity="drop|raise"`, `collinear_tol=` controls.
- Add regression tests for nested factor terms, continuous/factor interactions, FE-induced rank loss, duplicate instruments and full factor-interaction IV rank deficiency.


## Legacy pyreghdfe 0.7.0.dev1

- Replaced pairwise ad-hoc nested-FE deletion with exact FE partition canonicalization.
- Added refinement certificates and an explicit refinement DAG in `result.absorb_info`: component-containment proofs, exact functional-dependency proofs, observed-partition equivalences, and transitive reductions.
- Generalized safe redundancy detection to complex empirical interaction chains such as `year <= province#year <= city#year`, including relationships that are only implied by the observed data (for example `city -> province`).
- Added a four-stage candidate pipeline: specification proof, cardinality filter, deterministic sample rejection, then full-data exact certification. No FE is removed on a sampling heuristic alone.
- Added estimation-sample fixed-point handling: canonicalize before recursive singleton pruning, then canonicalize the original FE set again when pruning changes the sample.
- Kept heterogeneous-slope terms outside aggressive whole-term elimination and kept cluster nesting conceptually separate from absorption redundancy.
- Added canonicalization diagnostics (`candidate_pairs`, `specification_proofs`, `sample_rejections`, `full_data_checks`) for performance/traceability.
- Added complex hierarchy, equivalence, false-positive, post-singleton nesting, and three-way interaction tests; full suite now has 78 passing tests.
- In a 2M-row / 9-FE hierarchy microbenchmark, FE encoding took ~0.65 s and canonicalization ~0.07 s, reducing the effective system to `firm + city#ind4#year`.

## Legacy pyreghdfe 0.7.0.dev0

- Changed ordinary HDFE default acceleration to symmetric MAP + conjugate gradient, matching current `reghdfe` solver defaults.
- Replaced the previous CG residual-ratio stopping rule with the Hestenes/Stiefel improvement-potential criterion used by current `reghdfe`.
- Added `HDFEConvergenceError`; public OLS/IV APIs now reject non-converged absorption by default. Use `allow_nonconverged=True` only for diagnostics.
- Added categorical `interaction(...)` FE specifications and concise nested-tuple syntax such as `absorb=["firm", ("city", "year")]`.
- Added mixed-radix interaction encoding to avoid materializing large string/MultiIndex interaction columns for common FE interactions.
- Added `fe_structure.py` and exact FE canonicalization for safely redundant pure-intercept terms. Example: `year` is removed from the numerical absorption system when `city#year` is also absorbed.
- Deliberately retain heterogeneous slope-and-intercept terms even when their intercept is redundant, following `reghdfe` conditioning guidance.
- Added `result.absorb_info` with requested/effective FE systems, level counts, dropped nested terms, solver, convergence criterion, iterations, threads, backend and pool size.
- Added an interaction-FE benchmark modeled on empirical `firm + year + city#year` specifications.
- Added explicit `method="twoway"`: an exact two-intercept-FE Schur-complement solver with diagonal-preconditioned CG on the smaller FE side, designed for structures such as hundreds of thousands of firms crossed with a few thousand city-year cells.
- Expanded the regression suite to 72 passing tests.

## Legacy pyreghdfe 0.6.0

- Added `projection_backend="auto" | "indexed" | "fused"` for ordinary MAP absorption.
- Added compact O(N+G) FE group indexes built by direct counting sort, avoiding `argsort` and thread-private `G x K` accumulators.
- Added Numba `prange` group-parallel unweighted and weighted intercept-FE projection kernels.
- Added `absorb_threads=` with local thread scoping; explicit overrides do not mutate the reusable absorber.
- Added a low-memory parallel MAP convergence kernel with no N x K temporary ratio arrays.
- Updated MAP pool-size accounting to count only additional scratch arrays, allowing wider multi-RHS pools under the same stated scratch budget.
- Coordinated Joblib bootstrap workers with Numba absorption threads to prevent nested oversubscription.
- Kept heterogeneous-slope projectors modular: eligible intercept-only terms use indexed projection while slope terms retain the existing group-cross-product path.
- On the release container, 10M rows / 12 controls / 4 FEs (300k–450k levels) / two-way cluster fell from the prior ~177 s low-memory baseline to 79.8 s with a one-block RHS pool; conservative auto pooling measured 100.9 s.
- Expanded the numerical suite to 59 passing tests.

## Legacy pyreghdfe 0.5.0

- Added `group=` / `individual=` group-level-outcome estimation with the individual ID required in `absorb`, matching current reghdfe data semantics.
- Added `aggregation="mean" | "sum"` for individual FE contributions.
- Added individual-FE heterogeneous slopes in group-level-outcome models.
- Added recursive bipartite 1-core singleton pruning jointly with ordinary FE singleton pruning.
- Added `GroupIndividualAbsorber` with LSMR and LSQR solvers and no dense dummy expansion.
- Added `incidence_backend="auto" | "csr" | "matrix_free"`; auto selects an O(M) sparse CSR fast path under the memory budget and falls back to fully matrix-free reductions for larger graphs.
- Added group-level OLS and 2SLS/LIML/GMM2S paths with shared compression/residualization, cluster/HAC/DK plumbing where group-level inputs are valid, and FE recovery.
- Added group-mode `aweight`/`pweight`; deliberately reject `fweight` for group+individual membership data.
- Added `result.group_info` with group/membership/individual counts, solver and incidence backend.
- Kept individual-FE DoF conservative because current upstream `dof_update_individual_fe()` remains a TODO.
- Extended Stata golden fixture preparation from 15 to 18 models with group/individual mean/sum OLS and mean-aggregation IV.
- Expanded the numerical suite to 52 passing tests.

## Legacy pyreghdfe 0.4.0

- Added explicit `weight_type="fweight" | "aweight" | "pweight" | "generic"`.
- Split weight handling into regression weights, robust/cluster score
  multiplicity and effective-N bookkeeping.
- Made frequency-weight OLS IID/robust/cluster numerically equivalent to
  physically expanded rows in regression tests; did the same for robust 2SLS
  and weak-IV diagnostics.
- Added frequency-mass singleton handling and frequency-weight residual DoF.
- Added analytic/probability-weight normalization and pweight => robust behavior.
- Rejected IV frequency weights with HAC/kernel/Driscoll–Kraay covariance to
  preserve ivreg2/ivreghdfe compatibility boundaries.
- Added `save_fe=True` for OLS and IV, using matrix-free minimum-norm LSMR FE
  recovery with heterogeneous-slope basis conversion and reconstruction error.
- Changed `result.fitted` to full original-scale fitted values (`y - residual`),
  with combined FE fitted contribution available from `result.fixed_effects`.
- Extended Stata golden-test fixture/harness with f/a/p weighted OLS and IV rows.
- Expanded the numerical suite to 40 passing tests.

## Legacy pyreghdfe 0.3.0

- Generalized clustered covariance from two-way to 1–10 way inclusion-exclusion,
  including final PSD repair and the reghdfe-style cluster dimension cap.
- Added HAC/Newey–West and Driscoll–Kraay covariance with panel/time-aware lag
  construction and ivreg2-style spectral kernels.
- Added Kleibergen–Paap rk LM/Wald, Sanderson–Windmeijer diagnostics and
  Stock–Yogo critical-value lookup.
- Added a deterministic Stata golden-test harness and shared fixture generator.

## Legacy pyreghdfe 0.2.0

- Added `FixedEffect` / `fe()` heterogeneous-slope specifications.
- Added intercept+slope, pure-slope and multi-slope MAP/LSMR absorption without
  dummy expansion.
- Added pairwise mobility-group, nested-cluster and continuous-slope absorbed
  DoF accounting.
- Added 2SLS/LIML/k-class/Fuller/GMM2S estimator core.
- Added first-stage partial-R2/F, robust/cluster Wald F, Cragg–Donald and
  overidentification diagnostics.
- Reduced host peak memory via pooled in-place NumPy/MAP residualization and
  automatic memory-budget-based `pool_size`.

## Legacy pyreghdfe 0.1.0

- Initial clean-room release with intercept-only HDFE OLS/2SLS, MAP/CG/LSMR,
  singleton pruning, IID/robust/1–2 way cluster VCE, optional CuPy MAP, and
  parallel bootstrap.

