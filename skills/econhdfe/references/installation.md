# Installation and environment

Use this guide for package installation, optional extras, skill installation and environment verification.

## Python requirements

The metadata permits Python 3.10+; the tested support matrix is Python 3.10–3.13. Newer interpreters are not implied to be validated. Core dependencies are NumPy, SciPy, pandas, Numba, Joblib and threadpoolctl. Install the exact release artifact supplied with the project when reproducibility matters.

The repository version and package-index version may differ. Check the published
[releases](https://github.com/asggvhwgw-glitch/EconHDFE/releases) or
[PyPI](https://pypi.org/project/econhdfe/) and verify `econhdfe.__version__` after installation.
Post/Prediction requires 0.7.0; installing an older release does not enable it.

### Install a release wheel

From a downloaded release bundle:

```bash
python -m pip install ./econhdfe-<version>-py3-none-any.whl
```

Do not copy the example version literally; use the wheel filename that is actually present.

### Install from source

From the source tree:

```bash
python -m pip install .
```

For editable third-party development:

```bash
python -m pip install -e '.[test,io]'
```

### Optional extras

```bash
python -m pip install '.[io]'
python -m pip install '.[gpu]'
python -m pip install '.[test]'
```

- `io` adds Polars/PyArrow convenience dependencies. The estimator frontend remains pandas/NumPy oriented; convert external tabular objects before estimation unless a public adapter explicitly supports them.
- `gpu` installs the declared CuPy CUDA 12 build. A successful install does not prove that the host has a working compatible CUDA device/runtime.
- `test` installs the test-only dependencies used by the repository suite.

Do not install GPU extras merely because the data are large. CUDA transfer/setup and device memory can make CPU execution preferable.

## Verify the installed runtime

From this skill directory run:

```bash
python scripts/check_environment.py
```

Check at least:

- `econhdfe` import/version;
- Python and platform;
- core dependency versions;
- logical CPU count and Numba thread budget;
- optional IO libraries;
- CuPy import and CUDA device visibility when GPU execution is intended.

Then run:

```bash
python scripts/smoke_test.py
```

This is an installation smoke test, not a substitute for the repository validation matrix.

## Install the skill itself

The release bundle provides an `econhdfe` skill directory/zip. Put the whole `econhdfe/` directory in the skills directory used by the target agent client. Keep the directory intact so `SKILL.md`, `agents/`, `references/` and `scripts/` remain siblings.

Agent Skills clients load the skill progressively: metadata first, then `SKILL.md`, then individual references/scripts when needed. Do not flatten the reference files into `SKILL.md` during installation.

Client-specific skill locations vary. Prefer the client's documented skills directory rather than inventing a path. The skill folder name should remain `econhdfe` so it matches the declared skill name.

## Reproducible environment notes

For replication or paper production, record:

```bash
python --version
python -m pip freeze
python scripts/check_environment.py > econhdfe-environment.json
```

Also preserve the release wheel/source hash from the release bundle. Package version alone is weaker evidence than version + artifact hash + environment details.
