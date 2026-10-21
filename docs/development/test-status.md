# Validation status

## Current source: 0.8.3

The 0.8.3 candidate upgrades the package version and carries the compact suite
and documentation in [PR #21](https://github.com/asggvhwgw-glitch/EconHDFE/pull/21).
Its final head requires fresh full CI; no earlier green run is transferred to it.

The integrated 0.8.1 baseline is `main@b8890a7`, merged from PR #20. Its tested PR head
is `d41432f`; [CI 39573323752](https://github.com/asggvhwgw-glitch/EconHDFE/actions/runs/39573323752)
completed all 16 OS/Python/dependency jobs and the bundle job successfully.
These records describe that frozen tree, not later maintenance edits.

The current peripheral cleanup adopts a compact **205-case** regression suite.
It retains independent estimator references and selected statistical/failure
contracts; it intentionally narrows the broader 916-case checkpoint. The source
implementation and public API are unchanged by this test reduction. Run `python scripts/run_tests.py --suite full -- -q`
for all four layers; plain pytest runs only contracts + behavior. A local pass
for the cleanup does not extend the baseline CI result to a new commit.

Source version 0.8.3 alone is not evidence of a tag, GitHub Release, PyPI upload
or final artifact acceptance. Check [GitHub Releases](https://github.com/asggvhwgw-glitch/EconHDFE/releases)
and [PyPI](https://pypi.org/project/econhdfe/) for publication, and use
[testing.md](testing.md) for the current execution workflow.

## Historical 0.7 evidence

The following checkpoint is retained for provenance and does not describe the
current suite size or validate 0.8.1.

## Frozen 0.7.0 source, checked 2026-09-30

This record describes commit `e738333ad8a4c07fa15e3587238e4997ba21c19f`, not every later commit carrying version 0.7.0.

| Evidence | Actual result | Scope |
| --- | --- | --- |
| [CI 36613096958](https://github.com/asggvhwgw-glitch/EconHDFE/actions/runs/36613096958) | 16 environments and bundle succeeded | Linux/macOS/Windows × Python 3.10–3.13; four feasible minimum-dependency jobs |
| Source suite in that CI | 1067 passed, 1 skipped per environment | Full contracts/behavior/numerics/tooling suite; optional exact-backend skip is not a pass |
| Frozen bundle installed-wheel numerical check | 338 passed, 1 skipped | Tests run outside the source import path |
| Isolated build and fresh dependency installs | Passed | Executed by matrix and bundle jobs, separate from host-dependency checks |
| Local full suite with optional exact backend | 1068 passed | Python 3.12/macOS task environment; not a clean dependency-install claim |

The candidate artifact is `11054856349`, archive SHA256
`ff92a24671b51b611fd361a7eab5f3b186bca319565898d3c022d438d1a34e19`.
A detached local acceptance record was completed for those frozen bytes. It is
not embedded in the bundle and does not certify subsequent PR changes.

At this checkpoint the latest public release was **v0.6.3**; preparing and
validating 0.7.0 had not yet created its tag, GitHub Release or PyPI upload.
Check [GitHub Releases](https://github.com/asggvhwgw-glitch/EconHDFE/releases) and
[PyPI](https://pypi.org/project/econhdfe/) for the current publication state.

## Current checkout or PR

Use [testing.md](testing.md) for commands and the Actions run attached to the
exact commit for execution results. A version string, historical green run,
maintenance review, or generated map is not a substitute for testing that commit.
A changed runtime/validation fingerprint requires fresh evidence;
[formal publication](../release/acceptance.md) additionally binds final artifacts.

Internal numerical oracles do not establish complete licensed-Stata/upstream
certification. Optional GPU/IO/backends and target-scale benchmarks require their
own executed evidence. Lean verifies stated mathematics, not Python/Numba software.

## History

The prior [0.6.3 local candidate record](history/test-status-0.6.3.md) is retained
verbatim. Its older counts, environment and source-relative references describe
that checkpoint only. [0.6.5 test migration](history/test-migration-0.6.5.md)
explains the reorganized suite; historical counts are not current suite sizes.
