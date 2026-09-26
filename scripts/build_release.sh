#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="$(python -c 'import pathlib,re,sys; print(re.search(r"^version\s*=\s*\"([^\"]+)\"", pathlib.Path(sys.argv[1]).read_text(), re.MULTILINE).group(1))' "$ROOT/pyproject.toml")"
DIST="${1:-$ROOT/dist}"
MODE="${ECONHDFE_RELEASE_MODE:-candidate}"
[[ "$MODE" == candidate || "$MODE" == release ]] || { echo 'invalid ECONHDFE_RELEASE_MODE' >&2; exit 2; }
if [[ "$MODE" == release && "${ECONHDFE_OFFLINE_BUILD:-0}" == 1 ]]; then
  echo 'Formal release refuses ECONHDFE_OFFLINE_BUILD=1' >&2; exit 2
fi
# Review completion is NOT actual execution acceptance. An unpassed formal
# gate fails before any output is deleted or any dependency is downloaded.
if [[ "$MODE" == release ]]; then
  echo 'Formal authorization checks existing immutable artifacts; it does not rebuild them.' >&2
  [[ -n "${ECONHDFE_EXECUTION_MANIFEST:-}" ]] || { echo 'Set ECONHDFE_EXECUTION_MANIFEST to a detached final record' >&2; exit 2; }
  python "$ROOT/scripts/version.py" gate
  python "$ROOT/scripts/release_acceptance.py" check --mode release --manifest "$ECONHDFE_EXECUTION_MANIFEST" --artifact-dir "$DIST"
  python "$ROOT/scripts/verify_release.py" "$DIST/econhdfe-v${VERSION}-release-bundle.zip"
  exit 0
fi
python "$ROOT/scripts/release_acceptance.py" check --mode candidate
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-4}"
python "$ROOT/scripts/run_tests.py" --preflight

rm -rf "$DIST" "$ROOT/build" "$ROOT"/*.egg-info
mkdir -p "$DIST"
cd "$ROOT"

python scripts/version.py gate
# Freeze repository state BEFORE building either source distribution or wheel.
# Otherwise sdist and source ZIP can contain different generated maps/baselines.
python scripts/compatibility.py snapshot --version "$VERSION"
python scripts/compatibility.py check --version "$VERSION"
python scripts/generate_architecture_map.py
python scripts/generate_architecture_map.py --check
python -m compileall -q econhdfe pyreghdfe validation scripts benchmarks skills/econhdfe/scripts
python scripts/run_tests.py -- -q
if [[ "${ECONHDFE_OFFLINE_BUILD:-0}" == "1" ]]; then
  echo 'OFFLINE DIAGNOSTIC BUILD: host backend/dependencies; NOT isolated validation' >&2
  python -m pip wheel . --no-deps --no-build-isolation -w "$DIST"
  python -c 'import setuptools.build_meta as b,sys; b.build_sdist(sys.argv[1])' "$DIST"
else
  # The default frontend builds the wheel FROM the sdist, not alongside it.
  python -m build --outdir "$DIST"
fi
WHEEL="$(find "$DIST" -maxdepth 1 -type f -name "econhdfe-${VERSION}-*.whl" -print -quit)"
[[ -n "$WHEEL" ]] || { echo 'built wheel not found' >&2; exit 1; }
python scripts/verify_wheel_install.py "$WHEEL"
python scripts/verify_installed_numerics.py --wheel "$WHEEL" --extra-test test_weighted_colsum_fusion.py
if [[ "${ECONHDFE_OFFLINE_BUILD:-0}" != "1" ]]; then
  python scripts/verify_clean_install.py --wheel "$WHEEL"
fi

python scripts/assemble_release.py --wheel "$WHEEL" --sdist "$DIST/econhdfe-${VERSION}.tar.gz" --out-dir "$DIST"
TMP_SRC="$(mktemp -d)"
trap 'rm -rf "$TMP_SRC"' EXIT
unzip -q "$DIST/econhdfe-v${VERSION}-source.zip" -d "$TMP_SRC"
(cd "$TMP_SRC/econhdfe-${VERSION}" && PYTHONPATH=. python scripts/run_tests.py -- -q)
rm -rf "$TMP_SRC"
trap - EXIT

python scripts/verify_release.py "$DIST/econhdfe-v${VERSION}-release-bundle.zip"
echo "Candidate artifacts built; use detached execution evidence and --mode release for final authorization. No publication performed."
ls -lh "$DIST"
