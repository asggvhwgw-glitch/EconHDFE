# Economic problem map for econhdfe modules

This document is the economics-facing companion to the import/dependency architecture map. The dependency map answers **where code depends on code**; this map answers **why each runtime module exists in an empirical research workflow**.

Each module is described by (i) the econometric/economic problem it serves and (ii) its computational responsibility. A low-level numerical module is therefore not documented merely as “QR” or “projection”: the table states which empirical problem requires that operation.

## Reading rule

- **Economic / econometric problem** describes the research problem, identification issue, nuisance heterogeneity, or inference requirement faced by the user.
- **Computational responsibility** describes how the module implements that requirement.
- Performance layers such as heterogeneous-specification optimization never create new identifying variation or new heterogeneity; they only execute an already requested specification more efficiently.

## Public specification, contracts and workflows

| Module | Economic / econometric problem | Computational responsibility |
|---|---|---|
| `econhdfe.__init__` | **Public econometrics namespace.** Gives researchers one stable import surface for OLS-HDFE, IV-HDFE, PPML-HDFE, IV-PPML-HDFE and shared inference tools. | Re-exports the canonical public API; contains no estimator logic. |
| `econhdfe.api` | **Estimator choice and public dispatch.** Routes a stated empirical model to the correct estimator family without changing its econometric meaning. | Public wrappers, aliases and model-object dispatch. |
| `econhdfe.bootstrap` | **Backward-compatible bootstrap access.** Keeps existing empirical scripts using historical bootstrap entry points working. | Compatibility re-exports into the current resampling/inference implementation. |
| `econhdfe.collinearity` | **Identification after controls and fixed effects.** Detects explicit regressors that are not separately identified once other regressors and absorbed FE are included. | Structural/numerical column-rank filtering, including compact heterogeneous designs. |
| `econhdfe.config` | **Econometric and execution choices.** Separates economically meaningful choices—FE solver policy, inference type, execution resources—from hidden implementation constants. | Stable configuration dataclasses and validation. |
| `econhdfe.design` | **Turn an empirical formula into estimable regressors.** Expands factors, interactions, event-study terms, group-specific slopes and user omissions while preserving the requested parameterization. | Shared symbolic compiler plus dense or heterogeneous-specification materialization. |
| `econhdfe.design_structure` | **Remove structurally redundant controls before estimation.** Recognizes exact nesting/functional-dependency relationships that would otherwise create unidentified or wasteful explicit columns. | Partition-refinement and dependency-DAG structural reduction. |
| `econhdfe.errors` | **Distinguish data, identification, convergence and inference failures.** Lets researchers know whether a failed regression is a specification problem, weak/under identification, numerical failure or inference issue rather than a generic crash. | Structured exception hierarchy with stable codes/stages. |
| `econhdfe.factorvars` | **Express heterogeneous empirical specifications safely in Python.** Supports factor interactions, group-specific slopes, event-study-style terms and reference levels without Stata-style syntax ambiguity. | Parser/AST and specification objects for the fv() DSL. |
| `econhdfe.frontend.__init__` | **Frontend namespace.** Exposes input validation and preflight tools. | Namespace/re-export module. |
| `econhdfe.frontend.report` | **Preflight communication.** Reports input problems in a form usable by researchers or agents before an estimator is run. | Preflight issue/report objects and rendering. |
| `econhdfe.frontend.roles` | **Economic role assignment.** Distinguishes outcomes, regressors, endogenous variables, instruments, FE, clusters, weights and offsets so inappropriate transformations are not applied. | VariableRole definitions and role metadata. |
| `econhdfe.frontend.validate` | **Cheap pre-estimation data validity checks.** Catches nonnumeric outcomes/regressors, invalid weights and shape mismatches before expensive estimation. | Role-aware schema/data checks. |
| `econhdfe.pipeline` | **Create a common estimation sample and nuisance-control state.** Ensures outcome, regressors, FE, weights and clusters refer to the same observations before linear estimation. | Shared DataFrame preparation for linear OLS/IV workflows. |
| `econhdfe.reporting` | **Translate estimator output into paper-facing statistics.** Provides coefficient tables, model statistics and cluster counts that can be reported consistently across model families. | PublicationResult protocol and reporting helpers. |
| `econhdfe.results` | **Stable linear-model results.** Stores coefficients, uncertainty, FE metadata, R-squared/fit quantities and optional reusable state. | Result/state dataclasses for linear estimators. |
| `econhdfe.postestimation` | **Single-model coefficient post-estimation.** Tests economically meaningful linear combinations and joint restrictions without refitting or assuming independence across separate models. | Linear-combination and rank-aware Wald inference over the fitted coefficient vector and covariance matrix. |
| `econhdfe.prediction` | **Preserve exactly what a fitted design means after estimation.** Keeps future prediction from guessing factor bases, dropped columns, realized samples or absorbed-FE semantics. | Frozen compact prediction/sample/design/FE contracts; stores only reconstruction metadata, not the original dataset or estimator object. |
| `econhdfe.support_reports` | **Privacy-safe developer support without exposing empirical data.** Converts explicit allowlisted scalar/counter metadata into parameter-only error reports and writes canonical blank support/benchmark templates. | Strict allowlist extraction, packaged template loading, Markdown rendering and the `econhdfe-report` CLI. |
| `econhdfe.templates.__init__` | **Ship canonical support formats with installed builds.** Makes the privacy-safe error and benchmark report templates available without requiring a source checkout. | Package-resource namespace for wheel-bundled Markdown templates. |
| `econhdfe.sessions` | **Repeated regression-table and robustness specifications.** Speeds workflows that repeatedly change y, controls or FE combinations across many OLS/IV columns without altering any specification; wide file sources can be projected once for the whole table. | Coordinates DataSource/encoded-dataset reuse, exact FE-specific samples/absorbers, multi-RHS transforms and within-column caches with strict invalidation. |


## Econometric data access and estimation-sample preparation

| Module | Economic / econometric problem | Computational responsibility |
|---|---|---|
| `econhdfe.data.__init__` | **Econometric data-layer namespace.** Gives large-data workflows a source/sample/encoding layer that is independent of any one estimator. | Re-exports data-source, batching, encoding and sample-state infrastructure. |
| `econhdfe.data.source` | **Read only the raw variables an empirical specification actually uses.** Prevents wide firm/trade/panel files from being fully parsed when the regression needs only a small subset of columns. | DataFrame/CSV/Stata/Parquet source abstraction with projected and batched scans. |
| `econhdfe.data.planner` | **Fit data preparation inside the same memory constraints as estimation.** Chooses a batch size from the projected econometric columns rather than from the width of the raw source. | Preview-based bytes-per-row estimation and bounded-memory ingestion plans. |
| `econhdfe.data.persistent` | **Resume exploratory empirical work across Python runs without silently reusing stale estimates.** Lets researchers change outcomes/controls or continue an interrupted regression table while treating the source data and specification as authoritative. | Versioned disposable on-disk encoded-column, linear within-column and completed-result caches with strict dependency fingerprints, atomic writes and source-change invalidation. |
| `econhdfe.data.dataset` | **Reuse an estimation-ready econometric snapshot across a regression table or robustness workflow.** Keeps only columns used by the specifications, safely encodes identifier-only FE/cluster variables, and avoids repeating validation/conversion work. | Implements immutable encoded column snapshots, union-spec preparation, validity/signature caches, and source-to-session bridging. |
| `econhdfe.data.encoding` | **Keep firm/year/industry/cluster identities consistent across streamed chunks.** Stable codes are required so one economic group cannot receive different IDs in different batches. | First-seen global categorical dictionary encoding with int32 runtime codes. |
| `econhdfe.data.sample` | **Maintain one auditable estimation sample across data cleaning, FE pruning and estimator-specific exclusions.** Prevents different layers from silently using different observations. | Monotone sample mask and stage/reason exclusion history. |

## Execution planning

| Module | Economic / econometric problem | Computational responsibility |
|---|---|---|
| `econhdfe.planner.__init__` | **Keep performance choices separate from econometric meaning.** Provides the shared planning namespace used by data, HDFE and compute layers after the specification has already fixed the estimand. | Re-exports exactness certificates, resource contracts and execution-plan primitives. |
| `econhdfe.planner.contracts` | **Prevent a faster path from silently changing the requested model.** Represents exact eligibility separately from cost, memory and parallel policy so uncertified optimizations cannot win on performance grounds. | Immutable certificate, memory, representation, parallel and composite execution-plan dataclasses. |
| `econhdfe.planner.resources` | **Respect the actual machine/container available to an empirical job.** Gives all execution layers one view of CPU quota/affinity and memory limits rather than letting each estimator guess independently. | Cgroup/affinity CPU discovery, memory-limit discovery and bounded Numba runtime sizing. |
| `econhdfe.planner.core` | **Choose how to execute a fixed econometric problem under finite resources.** Compares only exact candidate representations and allocates memory/thread budgets without altering identification, samples or estimating equations. | Shared memory envelope, non-oversubscribed parallel layout and byte-traffic representation cost model. |
| `econhdfe.planner.calibration` | **Make automatic threading reflect the researcher’s actual runtime rather than a generic CPU-count assumption.** Measures where memory-bound HDFE-style work saturates on the current machine without touching user data or changing the econometric problem. | Synthetic Numba thread-scaling probe, anonymous runtime fingerprint, persistent calibration cache and saturation-point selection. |
| `econhdfe.planner.report` | **Let researchers report machine-specific planner/performance anomalies without exposing research data.** Produces developer feedback from aggregate resource, calibration and execution metadata only. | Privacy-minimized Markdown/JSON planner report generation and anonymous result/execution summaries. |
| `econhdfe.frontend.columns` | **Translate an economic specification into raw-data requirements before I/O.** Finds the source columns implied by outcomes, controls, IV roles, FE, factor interactions, clusters and weights without expanding the design matrix. | Recursive specification-to-column projection compiler. |

## Estimator-independent compute layer

| Module | Economic / econometric problem | Computational responsibility |
|---|---|---|
| `econhdfe.compute.__init__` | **Estimator-independent compute namespace.** Keeps common numerical work reusable across OLS, IV, PPML and IV-PPML. | Namespace/re-export module. |
| `econhdfe.compute.backend` | **Run the same econometric projection on available hardware.** Chooses NumPy/GPU-compatible numerical backends without changing the model. | Backend discovery and array-module helpers. |
| `econhdfe.compute.block_design` | **Avoid materializing structural zeros in heterogeneous specifications.** Stores event-study/group-specific-slope designs as row-local dense blocks plus shared coefficients when that is exactly equivalent to the requested model. | Validated BlockDesign representation and linear operators. |
| `econhdfe.compute.clusters` | **Represent dependence groups consistently.** Ensures clustered inference uses the intended economic grouping variables and valid group counts. | Cluster normalization/factorization utilities. |
| `econhdfe.compute.context` | **Reuse expensive compiled state safely.** Allows repeated estimations to reuse exact cached objects only when the underlying empirical design is unchanged. | ExecutionContext caches and profiling state. |
| `econhdfe.compute.design_ops` | **Compute sufficient statistics without unnecessary wide matrices.** Supplies column moments, Gram matrices and related operations for dense and heterogeneous designs. | Representation-agnostic design linear algebra. |
| `econhdfe.compute.design_plan` | **Recognize economically requested coefficient heterogeneity before dense expansion.** Finds row components and coefficient support for event studies, factor interactions and group-specific slopes, while accounting for FE connectivity. | Exact structural certificates plus storage/execution planning metadata. |
| `econhdfe.compute.encoding` | **Turn categorical economic groups into compact codes.** Converts firm/year/industry/cluster labels into dense integer representations used by FE and inference algorithms. | Stable factorization/encoding kernels. |
| `econhdfe.compute.execution` | **Choose a feasible execution strategy for the same estimator.** Balances memory, threads and repeated passes so large microdata specifications fit available hardware. | Execution/workspace planning. |
| `econhdfe.compute.kernels` | **Fast grouped sufficient-statistic calculations.** Makes the repeated sums/products behind FE, WLS and covariance feasible on large samples. | Compiled low-level numerical kernels. |
| `econhdfe.compute.linalg` | **Solve small identified parameter problems robustly.** Handles rank checks and linear algebra after nuisance FE or structural zeros have been removed. | Gram certificates, QR fallbacks and shared linear-algebra helpers. |
| `econhdfe.compute.partitioned_lstsq` | **Estimate common and group-specific coefficients without a global dense design.** Solves high-accuracy WLS for event-study/group-specific-slope designs with a small set of shared controls. | Block-angular QR/TSQR with conservative conditioning and memory fallbacks. |
| `econhdfe.compute.runtime` | **Use CPU threads without changing statistical results.** Coordinates thread limits and runtime resources so nested numerical libraries do not oversubscribe. | Runtime/thread context management. |
| `econhdfe.compute.vcov` | **Valid uncertainty under heteroskedasticity and correlated shocks.** Computes IID, robust, multi-way clustered, HAC and Driscoll–Kraay covariance where supported. | Sandwich meats, small-sample scaling and PSD handling. |
| `econhdfe.compute.weights` | **Apply sampling/frequency weights with the intended estimand.** Normalizes and validates weight semantics shared by estimators. | Weight parsing and effective-sample metadata. |
| `econhdfe.compute.wls` | **Core weighted least-squares step.** Provides the linear projection solved directly by OLS and repeatedly inside nonlinear/IV procedures. | Weighted arrays, least-squares solves and residual calculations. |

## High-dimensional fixed effects

| Module | Economic / econometric problem | Computational responsibility |
|---|---|---|
| `econhdfe.hdfe.__init__` | **High-dimensional fixed-effect namespace.** Exposes the shared nuisance-heterogeneity machinery used by model families. | Namespace/re-export module. |
| `econhdfe.hdfe.absorber` | **Partial out 1, 2 or many high-dimensional fixed effects.** Implements Frisch–Waugh–Lovell residualization when nuisance effects such as firm, year, city or industry are too numerous to dummy-expand. | Generic weighted HDFE absorber and iterative solvers. |
| `econhdfe.hdfe.block_projection` | **Absorb FE locally in heterogeneous explicit-coefficient specifications.** Keeps event-study/group-specific-slope fast paths from rebuilding a global wide design when FE topology certifies row components. | Component-local weighted FE projection over BlockDesign. |
| `econhdfe.hdfe.dof` | **Correct residual degrees of freedom after absorbing many FE.** Translates the nuisance FE rank/nesting structure into fit and inference DoF. | Pairwise/exact DoF accounting and cluster-nesting adjustments. |
| `econhdfe.hdfe.encoding` | **Encode fixed-effect groups.** Transforms firm/year/industry and interaction FE identifiers into solver-ready codes. | HDFE-specific factorization helpers. |
| `econhdfe.hdfe.factorvars` | **Translate absorbed factor syntax into FE objects.** Lets researchers state firm#year FE or group-specific trends compactly in absorb=. | Absorb-context fv() compiler. |
| `econhdfe.hdfe.group_individual` | **Control for individual effects in group-level outcomes.** Handles settings such as teams, patents or boards where one observation/group contains multiple individuals and individual FE must be absorbed. | Group–individual incidence absorber and diagnostics. |
| `econhdfe.hdfe.numerical_core` | **Avoid solving nuisance FE equations that are structurally trivial.** Peels degree-one FE observations/levels so only the residual multiway FE core requires iteration. | Exact numerical residual-core reduction for eligible multiway categorical FE. |
| `econhdfe.hdfe.plan` | **Compile the FE problem once.** Creates reusable topology/projector plans for a given set of nuisance fixed effects and sample. | FEPlan construction, fingerprints and projector selection. |
| `econhdfe.hdfe.projection` | **Fast projection on categorical FE groups.** Performs grouped demeaning/projection operations used by the generic absorber. | Fused/indexed projection kernels and thread control. |
| `econhdfe.hdfe.rank` | **Count how many nuisance FE coefficients are actually independent.** Provides exact absorbed-rank information needed for degrees of freedom with 3+ categorical FE partitions. | Exact characteristic-zero categorical design rank and certificates. |
| `econhdfe.hdfe.specs` | **Describe nuisance heterogeneity to absorb.** Represents categorical FE and heterogeneous FE slope terms without explicit dummy expansion. | FE specification dataclasses. |
| `econhdfe.hdfe.structure` | **Remove exactly redundant fixed-effect partitions.** Recognizes nested/redundant FE structures so nuisance controls are not absorbed twice while preserving the requested inference topology. | FE canonicalization and structural certificates. |
| `econhdfe.hdfe.two_way` | **Fast two-way fixed effects.** Specializes the common firm×time / exporter×importer style two-FE problem. | Schur/PCG two-way absorber with reusable topology. |
| `econhdfe.hdfe.weighted_projection` | **Repeatedly absorb FE as weights change.** Supports PPML/IV-PPML IRLS where the economic model is fixed but observation weights change every iteration. | Reusable weighted FE projector lifecycle. |

## Recovered fixed effects

| Module | Economic / econometric problem | Computational responsibility |
|---|---|---|
| `econhdfe.effects.__init__` | **Recovered-effect namespace.** Keeps FE recovery optional and separate from estimators that only need nuisance-effect absorption. | Small namespace for the v0.6 identified categorical-FE recovery API. |
| `econhdfe.effects.topology` | **Determine which reported FE levels can be compared.** Distinguishes ordinary component shifts from additional rank deficiencies before individual FE coefficients are interpreted. | Exact-rank/nullity metadata plus multipartite incidence-component labels. |
| `econhdfe.effects.recover` | **Recover economically interpretable additive FE after estimation.** Decomposes an identified FE contribution such as worker+firm or origin+destination without changing the structural estimator. | Matrix-free categorical FE recovery and component-preserving re-normalization. |
| `econhdfe.effects.adapters` | **Connect existing estimators to FE recovery without coupling the solver layers.** Converts linear fitted values or Poisson linear predictors into the common additive `Dγ` target. | Thin OLS/IV and PPML/IV-PPML post-estimation adapters; indicator FE only. |
| `econhdfe.effects.diagnostics` | **Explain why reported FE cannot be identified on the realized estimation sample.** Separates ordinary normalization from data/specification rank failure and records support lost to singleton pruning or PPML separation. | Stage-aware rank/nullity diagnostics, nested-support detection and structured FE-identification errors. |
| `econhdfe.effects.results` | **Keep recovered FE levels, identification and normalization auditable.** Separates what the data identify from the reporting baseline chosen by the researcher. | Level-sized result, normalization and diagnostic contracts. |

## Instrumental-variable primitives

| Module | Economic / econometric problem | Computational responsibility |
|---|---|---|
| `econhdfe.iv.__init__` | **Instrumental-variable primitive namespace.** Keeps identification mechanics reusable by linear IV and IV-PPML. | Namespace/re-export module. |
| `econhdfe.iv.design` | **Assign exogenous, endogenous and excluded-instrument roles.** Represents the exclusion restrictions that identify causal/structural effects under endogeneity. | IVDesign construction and role bookkeeping. |
| `econhdfe.iv.moments` | **Evaluate instrument orthogonality conditions.** Builds the score/moment objects E[z*u]=0 used by IV estimators and covariance calculations. | Additive IV moment primitives. |
| `econhdfe.iv.solve` | **Recover coefficients from instrumented variation.** Solves the weighted 2SLS inner problem that can be used by different outcome models. | Outcome-agnostic weighted 2SLS sufficient-statistic solve. |

## Outcome-model estimators

| Module | Economic / econometric problem | Computational responsibility |
|---|---|---|
| `econhdfe.models.__init__` | **Estimator-family namespace.** Groups outcome-model implementations while keeping shared infrastructure outside models. | Namespace/re-export module. |
| `econhdfe.models.linear_iv.__init__` | **Linear IV estimator namespace.** Exposes 2SLS/LIML/k-class/GMM and associated diagnostics. | Namespace/re-export module. |
| `econhdfe.models.linear_iv.api` | **Public endogenous linear-regression workflow.** Combines instrument roles, HDFE controls, estimator choice and inference for DataFrame/array users. | Linear IV-HDFE public orchestration and heterogeneous-spec dispatch. |
| `econhdfe.models.linear_iv.diagnostics` | **Assess instrument relevance and identification.** Reports first stages, partial F-type statistics, Cragg–Donald/Kleibergen–Paap/Sanderson–Windmeijer where defined, and over-identification diagnostics. | Linear weak-ID and specification diagnostics. |
| `econhdfe.models.linear_iv.estimators` | **Estimate linear effects under endogeneity.** Implements 2SLS, LIML, Fuller/k-class and two-step GMM estimators for alternative IV robustness/efficiency choices. | Estimator formulas and coefficient/covariance assembly. |
| `econhdfe.models.linear_iv.stock_yogo` | **Interpret conventional weak-IV statistics.** Provides published Stock–Yogo critical values where their assumptions apply. | Critical-value tables/lookups. |
| `econhdfe.models.ols` | **Linear partial effects with high-dimensional controls.** Estimates ordinary/weighted linear regressions after absorbing nuisance FE; supports robust and clustered inference. | OLS-HDFE orchestration, FWL solve, fit statistics and heterogeneous-spec fast path. |
| `econhdfe.models.ols_bootstrap` | **Few-cluster/sampling uncertainty for OLS.** Defines OLS-specific bootstrap statistics while delegating generic resampling mechanics. | Wild/bootstrap wrappers and OLS refit statistics. |
| `econhdfe.models.poisson` | **Poisson multiplicative conditional mean.** Defines the common mean/link/deviance calculations shared by PPML and IV-PPML. | Numerically stable exponential, working-state and deviance primitives. |
| `econhdfe.models.ppml.__init__` | **PPML estimator namespace.** Exposes the PPML-HDFE model, configuration and results. | Namespace/re-export module. |
| `econhdfe.models.ppml.api` | **Public nonnegative-outcome / trade-flow workflow.** Accepts economic specifications for PPML-HDFE, including exposure/offset, FE and clustered inference. | Public DataFrame/array orchestration and structured-spec dispatch. |
| `econhdfe.models.ppml.config` | **PPML numerical and separation choices.** Records tolerances, separation methods and solver strategy without changing the model equation. | PPMLConfig definitions and validation. |
| `econhdfe.models.ppml.estimator` | **Reference pooled PPML-HDFE estimator.** Estimates multiplicative conditional-mean effects with high-dimensional nuisance FE and returns the correctness fallback for unsupported fast paths. | Sample preparation, separation, IRLS, final WLS and result assembly. |
| `econhdfe.models.ppml.execution` | **Resource policy for large PPML.** Chooses projection/workspace resources for repeated weighted estimation. | PPML execution planning and diagnostics. |
| `econhdfe.models.ppml.heterogeneous` | **Fast PPML for event studies and group-specific slopes.** Avoids dense structural zeros in common interaction-rich PPML specifications while preserving global convergence, separation and inference semantics. | Certified heterogeneous-spec design, component projection, block IRLS/final WLS/VCE with dense fallback. |
| `econhdfe.models.ppml.irls` | **Solve the PPML first-order conditions.** Iteratively updates the weighted linear approximation until the Poisson pseudo-likelihood solution converges. | IRLS state, tolerance and predictor-update logic. |
| `econhdfe.models.ppml.results` | **Report PPML estimates and fit.** Stores coefficients, uncertainty, likelihood/deviance/pseudo-R2 and solver diagnostics. | PPMLResult dataclass/protocol. |
| `econhdfe.models.ppml.separation` | **Detect nonexistence of finite PPML coefficients.** Coordinates separation checks so observations implying infinite estimates are handled before/within fitting. | Separation orchestration and result bookkeeping. |
| `econhdfe.models.ppml.separation_relu` | **Detect PPML separation in high-dimensional designs.** Uses the ReLU-based separation procedure from the ppmlhdfe lineage for cases not resolved by cheaper checks. | ReLU separation algorithm. |
| `econhdfe.models.ppml.separation_simplex` | **Detect PPML separation via linear feasibility.** Uses simplex/LP-style feasibility conditions to identify separated observations. | Simplex separation algorithm. |
| `econhdfe.models.ppml.standardize` | **Improve numerical conditioning without changing PPML coefficients.** Centers/scales explicit regressors so very different units do not destabilize IRLS, then maps estimates back. | PPML standardization transforms and inverse mapping. |
| `econhdfe.models.ppml.vce` | **PPML uncertainty.** Constructs model/robust/cluster covariance from PPML score contributions. | PPML-specific covariance wrapper over shared sandwich primitives. |
| `econhdfe.models.ppml_iv.__init__` | **IV-PPML estimator namespace.** Exposes nonlinear IV-PPML, bias correction and bootstrap tools. | Namespace/re-export module. |
| `econhdfe.models.ppml_iv.api` | **Public endogenous PPML workflow.** Lets users combine nonnegative multiplicative outcomes, endogenous regressors, excluded instruments and high-dimensional FE. | Public IV-PPML orchestration and heterogeneous-spec dispatch. |
| `econhdfe.models.ppml_iv.bias` | **Reduce incidental-parameter bias in short panels.** Implements Class A/B/C split-panel jackknife corrections for IV-PPML panel settings. | SPJ panel splitting, recombination and result objects. |
| `econhdfe.models.ppml_iv.bootstrap` | **Sampling uncertainty for bias-corrected IV-PPML.** Bootstraps the model-specific SPJ statistic while retaining cluster structure where requested. | IV-PPML/SPJ bootstrap orchestration. |
| `econhdfe.models.ppml_iv.config` | **IV-PPML numerical choices.** Records convergence, separation, weighting and execution policy for the nonlinear IV estimator. | IVPPMLConfig definitions and validation. |
| `econhdfe.models.ppml_iv.estimator` | **Reference pooled IV-PPML estimator.** Solves additive IV moments for a multiplicative mean with FE and provides the correctness fallback. | Sample/separation setup, IV-IRLS, final inference and result assembly. |
| `econhdfe.models.ppml_iv.heterogeneous` | **Fast IV-PPML with heterogeneous coefficients/instruments.** Avoids dense structural zeros when endogenous slopes and excluded instruments vary by group/event time, while preserving the same global IV moments. | Compact role-specific designs, component FE projection, block sufficient statistics and dense fallback. |
| `econhdfe.models.ppml_iv.irls` | **Iteratively solve nonlinear IV moments.** Updates the multiplicative mean and weighted 2SLS step until the IV-PPML estimating equations converge. | IV-IRLS state and iteration logic. |
| `econhdfe.models.ppml_iv.results` | **Report nonlinear IV estimates.** Stores coefficients, covariance, convergence and IV-PPML diagnostics. | IVPPMLResult dataclasses/protocol. |
| `econhdfe.models.ppml_iv.standardize` | **Condition IV-PPML regressors/instruments.** Applies compatible scaling to exogenous/endogenous/instrument columns and maps parameters back. | IV-PPML standardization transforms. |
| `econhdfe.models.ppml_iv.vce` | **IV-PPML uncertainty.** Builds robust/cluster sandwich covariance from nonlinear IV score moments. | IV-PPML covariance calculations. |

## Post-estimation cluster inference

| Module | Economic / econometric problem | Computational responsibility |
|---|---|---|
| `econhdfe.inference.__init__` | **Post-estimation inference namespace.** Separates advanced inference procedures from estimator definitions. | Namespace/re-export module. |
| `econhdfe.inference.cluster.__init__` | **Cluster-inference namespace.** Exposes cluster diagnostics and certified wild-cluster tests. | Namespace/re-export module. |
| `econhdfe.inference.cluster.diagnostics` | **Judge whether clustered asymptotics look fragile.** Summarizes cluster counts, imbalance and score concentration so researchers can inspect few/dominant-cluster problems. | Cluster-size and score-concentration diagnostics. |
| `econhdfe.inference.cluster.results` | **Report advanced cluster inference.** Stores diagnostics/test outputs separately from estimator results. | Cluster inference result dataclasses. |
| `econhdfe.inference.cluster.wild` | **Inference with few or unbalanced clusters.** Implements certified one-way OLS wild-cluster restricted/unrestricted tests when CRV1 asymptotics may be unreliable. | WCR11/WCU11 wild-cluster test mechanics. |

## Resampling

| Module | Economic / econometric problem | Computational responsibility |
|---|---|---|
| `econhdfe.resampling.__init__` | **Generic resampling namespace.** Exposes estimator-agnostic draw mechanics. | Namespace/re-export module. |
| `econhdfe.resampling.engine` | **Run repeated bootstrap draws reproducibly.** Coordinates seeds, parallel batches, cluster resampling and failure collection. | Generic resampling engine. |
| `econhdfe.resampling.results` | **Represent resampling outcomes.** Standardizes bootstrap draw/failure metadata returned to model-specific callers. | Generic resampling result containers. |
| `econhdfe.resampling.sampling` | **Generate bootstrap sampling weights/draws.** Provides Rademacher and related primitive random draws used by model-specific bootstrap procedures. | Sampling distributions and group-row helpers. |

## Architecture principle

The package is organized around a small number of empirical problems rather than around numerical algorithms:

1. **Observed nuisance heterogeneity:** HDFE absorbs firm, time, geography, industry and related fixed effects without making those nuisance coefficients the object of interest.
2. **Endogeneity:** the IV layers separate instrument/exogeneity logic from the outcome model so linear IV and IV-PPML can share the same exclusion-restriction primitives.
3. **Nonnegative multiplicative outcomes:** PPML supplies the outcome model used heavily for trade flows and other zero-inclusive nonnegative outcomes.
4. **Dependence and finite-cluster inference:** covariance, cluster diagnostics and wild-cluster tests address correlated economic shocks rather than being generic matrix utilities.
5. **Specification-rich empirical work:** factor variables, repeated-spec sessions and heterogeneous-specification optimization make event studies, group-specific slopes and robustness tables computationally feasible without changing the estimand.
6. **Paper-facing comparability:** result/reporting contracts keep statistics interpretable and comparable across estimator families.

The numerical architecture (`compute`, projection kernels, QR/TSQR, runtime planning) is subordinate to these econometric responsibilities.

## Release-audit numerical safeguards (0.6.1)

| Runtime module | Economic/econometric purpose |
| --- | --- |
| `econhdfe.compute.stable_linalg` | Keep estimates, covariance bread and numerical rank consistent under changes of regressor units, using equilibrated orthogonal least-squares reduction. |
| `econhdfe.hdfe.codes` | Enforce the observed-level integer coding assumptions needed for exact categorical absorbed degrees of freedom and fixed-effect identification. |
