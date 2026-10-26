#!/usr/bin/env bash
set -euo pipefail
# Intended for a sufficiently provisioned external host. Environment variables
# let CI choose how expensive the validation should be.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
python -m pytest -q

TMPDIST="$(mktemp -d)"
trap 'rm -rf "$TMPDIST"' EXIT
python -m pip wheel . --no-deps --no-build-isolation -w "$TMPDIST"
WHEEL="$(find "$TMPDIST" -maxdepth 1 -type f -name 'econhdfe-*.whl' -print -quit)"
python scripts/verify_wheel_install.py "$WHEEL"

if [[ "${RUN_CPU_BENCH:-1}" == "1" ]]; then
  python benchmarks/bench_cpu_projection.py
fi
if [[ "${RUN_10M_BENCH:-0}" == "1" ]]; then
  python benchmarks/bench_10m_multife.py
fi
if [[ "${RUN_STATA_GOLDEN:-0}" == "1" ]]; then
  python validation/make_fixture.py
  "${STATA_BIN:-stata-mp}" -b do validation/stata/generate_golden.do
  python validation/compare_golden.py
fi
