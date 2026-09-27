# Review, execution acceptance and formal authorization

`maintenance.json` is a REVIEW checklist. `reviewed` can legitimately mean that
a missing external result was considered and documented. It cannot mean that
the missing validation ran successfully. The uniform roadmap is `../../TODO.md`.

`execution.json` is a separate execution record. Every fixed required area uses
`not_run`, `blocked`, `failed` or `passed`. Required areas cannot be disabled with
a `required: false` field. A successful row requires a timezone-aware timestamp,
an actual environment, the current runtime/validation fingerprint, and SHA256
references to existing evidence under the repository. Failed, blocked and
not_run checks need explicit reasons; they never count as passed.

Required areas: local_regression, installed_numerics, isolated_build,
clean_install, support_matrix, dependency_floor, reference_claims, final_artifacts.
`reference_claims` may be satisfied by obtaining the claimed parity evidence OR
by an explicit, checked narrowing of the public support/compatibility claims.
A fake run or a mock test is not an acceptable substitute.

## Code identity versus artifact identity

`runtime-validation-v1` fingerprints runtime packages, scripts, tests, Skill,
CI, compatibility records, pyproject.toml and MANIFEST.in. It excludes JIT/cache
files and docs/evidence so execution records do not hash themselves. This is
not a Git commit or the hash of the full source archive. A change to included
files makes evidence stale. Never simply relabel an old passed row: rerun its
affected validation. A bump resets all rows to not_run.

The final manifest is DETACHED: keep it outside the immutable published bundle.
It records the actual source_commit and SHA256 records for the final wheel,
sdist, source ZIP and release bundle. Gate hashes check consistency, not author
honesty, CI attestation or cryptographic signatures. Reviewers still need to
inspect the real logs and provenance. The source_commit field is format-checked;
repository/CI provenance is part of the required final_artifacts evidence.

## Commands

```bash
# Schema + code/evidence binding; unpassed states are permitted, never promoted.
python scripts/release_acceptance.py check --mode candidate

# Normal candidate build still requires isolated construction and clean install.
bash scripts/build_release.sh

# Explicit HOST-backend/dependency diagnostic candidate only.
ECONHDFE_OFFLINE_BUILD=1 bash scripts/build_release.sh

# Formal authorization checks existing immutable artifacts; it never rebuilds.
python scripts/release_acceptance.py check --mode release \
  --manifest /path/to/detached-execution.json --artifact-dir /path/to/final-dist
ECONHDFE_RELEASE_MODE=release \
ECONHDFE_EXECUTION_MANIFEST=/path/to/detached-execution.json \
  bash scripts/build_release.sh /path/to/final-dist
```

Formal mode refuses offline-host validation and checks the review/compatibility
gate, execution records and release structure before authorizing anything. It
fails before deleting outputs. It does not publish a tag, GitHub Release or PyPI
package. A valid candidate can and currently does report `release_ready=false`.

The GitHub workflow still produces candidate artifacts. Configuring a matrix
is not passing that matrix. Maintainers must collect its actual results, the
feasible dependency-floor tests and clean build/install evidence before formal
mode can succeed. The package does not invent commit IDs or remote test results.

## Publication handoff

After detached formal authorization succeeds, public tag/Release/PyPI publication is
performed by the manually dispatched **Publish release** workflow. It consumes the
already-built `release-candidate` from the authorized validation run and never
rebuilds it. See [publishing.md](publishing.md) for inputs, identity checks, retry
semantics and the PyPI Trusted Publishing constraint.
