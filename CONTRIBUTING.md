# Contributing

Use Python 3.10–3.13, create a virtual environment, and install `.[test]` plus
`build`. For daily public behavior, run `python scripts/run_tests.py --suite core -- -q`.
Plain `python -m pytest -q` now selects this core subset, NOT the full release suite.
Before submitting a PR run `python scripts/run_tests.py --suite full -- -q`,
`python scripts/version.py gate`, and `python scripts/generate_architecture_map.py --check`.
See `docs/development/testing.md` for contracts/behavior/numerics/tooling responsibilities
and `docs/development/history/test-migration-0.6.5.md` for every retired test's replacement.
For predictable CPU use, set OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1 and
NUMBA_NUM_THREADS=4 (tests exercise explicit two- and three-thread settings).

Changes to numerical code must include an independent reference calculation,
unit/column permutation checks, and failure-path tests. Do not weaken a regression
assertion to make a change pass. New mathematical claims require explicit
assumptions, a proof or a clearly marked conjecture, implementation mapping, and
adversarial examples. A passed test is not a proof or a priority certification.
See `docs/technical/history/mathematical-review-0.6.1.md`.

Register new runtime files in the economics-facing module map. Review both the
root API and `econhdfe.effects`, record intentional compatibility changes, then
freeze the versioned snapshot. Never rewrite an earlier version's baseline.

`bash scripts/build_release.sh` performs isolated PEP 517 builds, a new-venv
installation (including dependencies), smoke tests, source-archive retesting,
and release-bundle verification. It requires package-index access, or a complete
configured local wheelhouse. For offline diagnostics only, set
`ECONHDFE_OFFLINE_BUILD=1`; this explicit mode uses the host build backend and
host-dependency target smoke, and MUST NOT be described as an isolated build or
a clean dependency-resolution validation.

Use the parameter-only support templates. Do not post confidential observations,
identifiers, credentials, local file paths, or unredacted tracebacks in an issue.
No remote repository URL is hard-coded in this source distribution; maintainers
should add the canonical project/issue links when the repository is created.
