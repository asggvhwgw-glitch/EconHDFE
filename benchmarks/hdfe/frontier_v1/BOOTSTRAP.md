# Reproduce the native backend environment

This benchmark calls projection kernels directly. No native wheel is committed
to this repository. The measured environment uses the previously built GitHub
Actions wheels described below, without changing the installed production
EconHDFE package or its defaults.

## Exact source and build evidence

| Item | Identifier |
|---|---|
| Audited main commit | `44dae3f41f978a3ce78e0d780feab42b49a352f3` |
| Native adapter CI source | `23a0c12f04983514a64e7cc9b10145d1e34c584e` |
| CI run | `40828420235` |
| Artifact ID | `12876864543` |
| Artifact name | `bounded-planner-validation-40828420235` |
| Artifact ZIP SHA-256 | `5e1177bf350255bc868c8017191a09ffa55f42ee6efa0403f77f892bf4426105` |
| within core revision | `fc44282cea6124f5f18bb142908605a656d35a7c` |
| fereg core revision | `0fa59e3b539addbddaee8bd71acd031c497166a0` |

The Git blob IDs of **all 15 files under `native/within-accel/`** are identical
between the audited main commit and the CI source commit. Thus the `native_auto`
wheel contains the same wrapper source as this audited main, although the wheel
was built in the experimental branch's CI job. It is not claimed to have been
built by main's CI. The dual adapter comes from the experimental CI source and
has not been merged into production main.

| Installed wheel | SHA-256 |
|---|---|
| `econhdfe_within-0.1.0a2-cp310-abi3-linux_x86_64.whl` | `9f5744c473986a528eb1856f2c2ddf8a5894694225692519ba9fff321f7e7482` |
| `econhdfe_dual_prototype-0.1.0a1-cp310-abi3-linux_x86_64.whl` | `4f6e8825eed659977d546eaaeebc99cf868b13b5372237a95b70c44b5911bf56` |

`within` in this benchmark means the pinned upstream `Solver::solve_batch`
called through the dual adapter with `backend='within'` and
`allow_fallback=False`. `fereg` uses the same adapter with `backend='fereg'`.
This makes thread-pool control and wrapper certification comparable. They are
not the separately released PyPI `within-py` and `fereg` packages. Installing
those packages is unnecessary for these four benchmark backends.

The validated interpreter is CPython 3.12.14 on Linux x86-64. These two wheels
use CPython's ABI3 for 3.10+, but the full operating-system/runtime compatibility
matrix has not been tested. An import failure is a missing backend, not a pass.

## Download and install

Run from the repository root. Put the environment and artifact directory beside
the checkout so neither generated binaries nor dependency files enter Git.
The example assumes `python3` is the intended CPython 3.12 interpreter.

```bash
python3 -m venv ../fe-frontier-env
../fe-frontier-env/bin/python -m pip install numpy==2.3.5 scipy==1.17.0 pytest==9.2.0
mkdir -p ../fe-frontier-artifacts
gh api repos/asggvhwgw-glitch/EconHDFE/actions/artifacts/12876864543/zip > ../fe-frontier-artifacts/bounded-planner.zip
```

The GitHub Actions artifact endpoint requires a permitted GitHub account even
for a public repository. Use an existing `gh` authenticated session or supply
`GH_TOKEN` through the environment; do not print or commit the token. Without
that access, skip the download and report native backends unavailable. The
SciPy baseline remains runnable. The artifact's recorded expiry is
2027-01-27; after expiration this exact binary must come from a separately
preserved copy with the same digest or an explicitly labelled new build.

Verify and extract before installing:

```bash
../fe-frontier-env/bin/python - <<'PY'
from pathlib import Path
import hashlib
import zipfile

archive = Path('../fe-frontier-artifacts/bounded-planner.zip')
expected = '5e1177bf350255bc868c8017191a09ffa55f42ee6efa0403f77f892bf4426105'
actual = hashlib.sha256(archive.read_bytes()).hexdigest()
if actual != expected:
    raise SystemExit(f'Artifact checksum mismatch: {actual}')
with zipfile.ZipFile(archive) as bundle:
    bundle.extractall(archive.parent / 'bounded-planner')
wheelhouse = archive.parent / 'bounded-planner' / 'wheelhouse'
expected_wheels = {
    'econhdfe_within-0.1.0a2-cp310-abi3-linux_x86_64.whl':
        '9f5744c473986a528eb1856f2c2ddf8a5894694225692519ba9fff321f7e7482',
    'econhdfe_dual_prototype-0.1.0a1-cp310-abi3-linux_x86_64.whl':
        '4f6e8825eed659977d546eaaeebc99cf868b13b5372237a95b70c44b5911bf56',
}
for name, digest in expected_wheels.items():
    if hashlib.sha256((wheelhouse / name).read_bytes()).hexdigest() != digest:
        raise SystemExit(f'Wheel checksum mismatch: {name}')
print('Artifact and native wheel checksums verified.')
PY
../fe-frontier-env/bin/python -m pip install --no-deps ../fe-frontier-artifacts/bounded-planner/wheelhouse/econhdfe_within-0.1.0a2-cp310-abi3-linux_x86_64.whl ../fe-frontier-artifacts/bounded-planner/wheelhouse/econhdfe_dual_prototype-0.1.0a1-cp310-abi3-linux_x86_64.whl
../fe-frontier-env/bin/python -m pytest benchmarks/hdfe/frontier_v1/tests/test_backends.py -q
```

The preparation used a `--system-site-packages` environment because the same
NumPy/SciPy versions were already present; the commands above create an isolated
equivalent dependency set. They do not install the whole EconHDFE package.

## Interpretation of adapter metrics

- `setup_s` includes common input checks, exact label compaction where needed,
  array layout conversion and public constructor time.
- `solve_s` includes the entire projection call, native certification and output
  layout/copy. A backend can defer preconditioner construction to this call;
  it is therefore also included here. `solver_reported_setup_s`, when present,
  retains the backend's own subdivision without subtracting cost from the total.
- `total_s = setup_s + solve_s`. Dependency imports, provenance hashing and the
  runner's independent error/rank/spectrum oracles are excluded. The runner can
  separately record whole-process elapsed time and enforce its own deadline.
- Only `scipy_lsmr` reports instrumented matrix-vector-product counts. Native
  `actual_matvec_counts` is null. Native iteration counts, including corrective
  attempts, are retained as reported; they are not relabelled operator counts.
- For `fereg`, the wrapper repeats one lockstep-batch iteration count for every
  RHS. Thus `[7, 7]` must not be summed as two independent solves or actual
  matvec work. The core preserves prior PCG steps across adaptive restarts and
  counts an exact-factor correction as one step; factor construction and extra
  restart matvecs are not separately counted. The wrapper adds the shared count
  of any corrective projection to each entry. These costs remain timed.
- `within`/`fereg` use no cross-kernel fallback, but their shared adapter can make
  one corrective projection. Its `max_iter` limit is per attempt. `native_auto`
  shares its per-RHS limit across initial, continuation, polish and fallback.
  The metrics retain these different budget scopes.
- A returned native-auto nonconvergence flag is preserved alongside its output.
  If the dual adapter raises on a failed certificate, the runner records the
  exception. Missing packages, nonconvergence, exceptions and timeouts must not
  be removed from a comparison table or counted as passing projections.

The adapters also record package versions, upstream revision identifiers and
SHA-256 hashes of the actual loaded extension files. A rebuilt wheel with the
same package version is therefore distinguishable from this archived build.

The rigidity scaling trajectory crosses fereg's `small_limit=4096` between
ell32 (all remaining blocks small) and ell64 (both remaining blocks large).
The larger runs' seven-step count is consistent with a sixth-step joint-factor
upgrade plus an exact correction, but that route remains inferred because the
archived wrapper omits the relevant core diagnostics. See README.md's native
iteration discussion and pinned source links before interpreting that count.

## License evidence for preserved binary copies

The `econhdfe_within` wheel includes the EconHDFE BSD-3-Clause text and the
upstream within MIT text. The dual-prototype wheel's metadata declares
BSD-3-Clause, but that wheel contains no LICENSE or NOTICE files. Its embedded
within and fereg kernels retain their own upstream MIT notices; the wrapper's
metadata is not a complete inventory of linked third-party licenses.

The accompanying `fe-frontier-release-notices` directory preserves retrieved
license/notice texts, exact source locations and file hashes. It also includes
available direct numerical-dependency notices: approx-chol has a separate
NOTICE, and the inspected faer source distribution carries LAPACK/SuiteSparse
BSD notices and an Eigen MPL-2.0 notice in addition to its top-level MIT
declaration. The collection records the provenance and limits of that evidence;
it must not be described as a complete license audit of both native binaries.

The CI source has no committed `native/dual-absorber/Cargo.lock`. A subsequent
inspection found CycloneDX SBOMs **inside both preserved wheels**, which the
initial review had missed: 159 components in the dual wheel and 62 in the
native-within wheel. Their digests are respectively
`af75962626a5a291d03f126efdc3ce4e1dc99fa4ec9b9ce7f5b8bad02f35adfd` and
`dd1cc6fbd49b9ee7268c06f48ee8c2ad695ce9a6b99aa8265396de715900a56b`.
These provide a build-tool dependency inventory with resolved versions and
source hashes. The SBOMs mark `all_targets=true`; they are not a closed inventory
of code actually linked into the final extension or a completed file-level
license review. Preserve the collected notices with archived copies.

## Instrumented follow-up

The [native cost research build](native_cost/README.md) rebuilds these same
pinned kernels with explicit, recorded diagnostic patches and committed
lockfiles. It exposes actual operator calls and factor phases, including the
route that the original reports could only infer. Use its separate environment
and `+cost1` binary identities; do not relabel the original opaque measurements
as if they had contained these diagnostics.
