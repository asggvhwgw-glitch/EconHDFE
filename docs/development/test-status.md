# Validation status — 0.7

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
