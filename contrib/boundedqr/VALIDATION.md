# Contribution validation

Validated 2026-09-25 on Windows, Python 3.12, NumPy 2.5.3, SciPy 1.18.1, with BLAS/OpenMP capped at one thread. Fresh virtual environment with system packages disabled; dependencies resolved from the package index and all three Python distributions installed with isolated PEP 517 builds. No user datasets were used.

EconHDFE core suite: 905 passed, 1 optional test skipped (SymPy unavailable), no failures. Each independent branch passed `python scripts/version.py gate` and `python scripts/generate_architecture_map.py --check`. New companion checks are intentionally separate from parent test discovery.

82 checks passed and 15 optional GPU checks skipped. Checks include frozen synthetic quantreg draws, independent LP/radius checks, replay, scaling and input boundaries. The Python example completed all 57 draw/quantile pairs with finite numerical radii.

The live R example passed using R 4.6.1, quantreg 6.1 and jsonlite: nine fixed-multiplier draws differed from `quantreg::boot.rq` by at most 5.56e-16; all nine radii were finite. This small fixture is not a coverage or large-sample performance experiment. GPU execution has not been revalidated by this contribution.
