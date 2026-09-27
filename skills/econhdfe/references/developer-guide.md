# Third-party development and release integration

## Contents

- Architecture boundaries and adding public features
- Structured errors, results and configuration contracts
- Public-contract snapshot workflow
- Version/release closeout
- Extending the Agent Skill


Use this guide when changing econhdfe itself or building a downstream extension against its public contracts.

## Architecture boundary

Preserve separation between:

- frontend/schema/specification handling;
- shared HDFE encoding, canonicalization, projection and DoF;
- estimator-specific equations (OLS, linear IV, PPML, IV-PPML);
- inference/covariance;
- results/reporting;
- execution/runtime/cache policy;
- validation/release tooling.

Do not fork a second FE implementation inside a new estimator unless the mathematical operator genuinely differs and the duplication is justified/tested.

`econhdfe.factorvars` is a frontend-only parser that produces one symbolic factor-variable expression. Keep parsing separate from compiler targets: explicit regressor/instrument roles compile into the canonical `Factor` / `RegressorInteraction` representation handled by `econhdfe.design`; absorbed roles compile through `econhdfe.hdfe.factorvars` into canonical `FixedEffect` / `Interaction` specifications. Do not create a parallel dummy engine or a second FE solver. Absorb compilation must reduce exact lower-order full-factorial redundancies before DoF/numerical encoding, especially for heterogeneous slopes, so equivalent symbolic expressions do not duplicate absorbed slope components.

### HDFE solver invariant

Keep three objects separate when extending the HDFE backend: requested/inference FE topology, canonicalized numerical FE partitions, and any row-level numerical core reduction. `hdfe/rank.py`/DoF may reason about structural column-space rank; `hdfe/numerical_core.py` reduces only the numerical solve and must not rewrite the requested design used for inference. Weight updates may reuse a numerical core only while all effective row weights remain strictly positive.

## Repository documentation contract

Current technical/developer documentation is canonical under `docs/`: HDFE theory and solver material under `docs/technical/hdfe/`, architecture/validation material under `docs/development/`, and version/release governance under `docs/release/`. HDFE benchmark evidence belongs under `benchmarks/hdfe/`. Do not restore the former root-level document dump.

The exact multiway-DoF `.tex/.pdf` is a source/release audit artifact and must remain excluded from wheels.

The contributor architecture map is generated, not hand-maintained. After changing module boundaries or imports, run `python scripts/generate_architecture_map.py` from the repository root and `python scripts/generate_architecture_map.py --check` before release. The generated `architecture.json` is the evidence layer, `architecture.md` is the static/GitHub view, and `architecture.html` is the interactive view. Keep semantic execution flows separate from AST import claims.

## Adding a public feature

When exposing a new option or estimator behavior:

1. define the economic/statistical contract first;
2. decide the correct public layer (frontend, config, estimator, result, diagnostics);
3. reuse shared FE/runtime/inference components where mathematically valid;
4. add a structured public error for invalid user states rather than leaking internal exceptions;
5. add deterministic tests for success and failure paths;
6. update the appropriate skill reference only if end users/agents must change behavior;
7. run the public-contract diff before release.

Avoid exposing private numerical thresholds as permanent API parameters. Prefer stable strategy-level controls.

## Error-report contract

`docs/development/ERROR_REPORT_TEMPLATE.md` is the repository canonical user/developer bug-report format and must remain byte-identical to `skills/econhdfe/references/error-report-template.md`. It is intentionally **parameter-only**: collect anonymous model-role structure, dimension/count metadata, numerical/execution configuration, structured `EconHDFEError` code/stage fields, aggregate diagnostics, and reference-package difference magnitudes. Do not request raw/sampled/synthetic observations, actual variable/entity identifiers, paths/filenames, exact commands or replication scripts, raw logs, full tracebacks, stack locals, or file attachments. If the first report is insufficient, request narrower additional parameters/counters before considering any other workflow. Benchmarking remains separate and requires explicit user agreement.

## Privacy-safe report helper

The installed `econhdfe-report` console helper is support tooling, not an estimator API. `econhdfe-report error` and `econhdfe-report benchmark` copy the canonical templates shipped inside the wheel. For automatic in-process error reports, use `econhdfe.support_reports.safe_error_payload` / `build_error_report` / `write_error_report`. These helpers are deliberately **allowlist based**: do not broaden them by reflecting over arbitrary result/error/config attributes. In particular, never emit coefficient/fitted/residual arrays, variable or FE names, arbitrary string-valued `EconHDFEError.details`, error messages, file paths, commands, logs, or traceback text. Numeric error details must match the explicit counter/diagnostic key allowlist.

The wheel copies of both report templates must remain byte-identical to their repository/Skill canonicals; release verification checks this. Benchmark consent remains separate from merely generating a blank template.

## Structured errors

Public user-facing failures should derive from `EconHDFEError` and provide stable `code`/`stage`, structured details, and a useful suggestion when possible. Reuse an existing error family if it accurately describes the failure; do not create a new class for every message variation.

Changes to public error codes are contract changes and are caught by the compatibility snapshot.

## Results and reporting

Do not add ad hoc estimator attributes without deciding whether they belong in:

- the estimator-specific result dataclass;
- shared `RegressionResult`/publication schema;
- diagnostics metadata;
- execution profile/reproducibility metadata.

Paper-facing fields should remain stable and serializable enough for downstream table/reporting code. Internal diagnostic expansion should not force ordinary users to consume large telemetry objects.

## Configuration and defaults

Changes to `HDFEConfig`, `InferenceConfig`, `ExecutionConfig`, public signatures or defaults are compatibility-sensitive. Prefer additive optional controls with conservative defaults. For deprecation, preserve the old behavior long enough to emit an actionable migration path unless the package's current stability policy explicitly permits a break.

## Public-contract snapshot workflow

Before a release:

```bash
python scripts/compatibility.py diff
```

The diff is against the previous released baseline. If a public change is intentional, approve the exact change id with a written reason:

```bash
python scripts/compatibility.py approve <change-id> --reason "intentional API change; migration documented in ..."
```

The release gate rejects unapproved or stale approvals. Do not edit a baseline snapshot to hide a change; snapshots represent released artifacts.

## Version/release closeout

Do not manually edit version strings. When the update is complete:

```bash
python scripts/version.py bump <new-version>
```

The controlled bump resets the release-maintenance manifest and compatibility approvals against the previous release. Review every release responsibility and record evidence with `scripts/release_maintenance.py set`.

At minimum review:

- frontend/public API;
- backend compute/solver/cache behavior;
- structured errors;
- public-contract compatibility;
- skill routing/references/scripts;
- linked/shared modules;
- results/reporting and configs/defaults;
- state/cache invalidation;
- tests and performance relevance;
- docs/migration;
- packaging/dependencies;
- external validation claims;
- final artifacts/hashes/clean-install behavior.

Then run:

```bash
python scripts/version.py gate
bash scripts/build_release.sh
```

Already-published release artifacts are immutable. If bytes or public behavior change, publish a new version rather than replacing a wheel/source/bundle under an old version.

## Extending the skill

Keep `skills/econhdfe/SKILL.md` as a short router. Put role-specific detail one level deep under `references/`; put deterministic reusable checks under `scripts/`; keep OpenAI UI metadata in `agents/openai.yaml`. Do not duplicate large blocks between the router and references.

When adding a reference, link it directly from `SKILL.md` and state exactly when the agent should read it. Do not create nested sub-skills inside this skill.

## Technical-innovation claims

Treat originality as a release contract, not as marketing language. Before describing a new algorithm, theorem or inferential construction as an econhdfe technical innovation, read `docs/technical/innovation-audit.md` and update `docs/technical/innovation-registry.json`.

A registered `technical_innovation` must ship in the same release with a formal LaTeX manuscript and compiled PDF, an explicit prior-art boundary, implementation correspondence, correctness tests and appropriate benchmark/validation evidence. The release gate validates this chain.

Do **not** register caching, parallelism, backend selection, memory planning, kernel fusion, API design, compatibility reproduction, documentation or release tooling as technical innovations by themselves. Established econometric/numerical methods implemented by econhdfe should retain their upstream attribution even when the implementation is substantially faster.

## 0.6.4 native exact-rank resource boundary

Native modular and rational elimination share a per-residual-component budget:
50,000,000 charged coefficient-visit/update work units, 256 MiB conservative
sparse-row storage accounting, and 16,384 bits for integer intermediates.
These private implementation guards are not public tuning parameters. They do
not meter input/topology allocation, external SymPy/FLINT calls, hard process
RSS or wall time. A resource-limit RuntimeError means no exact rank was obtained;
do not catch it and return the last modular lower bound, silently switch DoF
methods, or increase private limits on behalf of a user. Report only the stage,
resource/counter/limit and environment via the parameter-only support workflow.
No data labels or observations are needed. See the development maintenance note
in the source tree for tests and measurement limitations.
