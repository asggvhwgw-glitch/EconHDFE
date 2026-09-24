# Contribution validation

Validated 2026-09-25 on Windows, Python 3.12, NumPy 2.5.3, SciPy 1.18.1, with BLAS/OpenMP capped at one thread. Fresh virtual environment with system packages disabled; dependencies resolved from the package index and all three Python distributions installed with isolated PEP 517 builds. No user datasets were used.

EconHDFE core suite: 905 passed, 1 optional test skipped (SymPy unavailable), no failures. Each independent branch passed `python scripts/version.py gate` and `python scripts/generate_architecture_map.py --check`. New companion checks are intentionally separate from parent test discovery.

382 checks passed, including independent HAC/group-score references, statsmodels OLS/WLS covariance, the guarded linearmodels 7.0 overlay, and explicit EconHDFE bandwidth/normalization parity. The macrodata example and EconHDFE score example passed. One expected warning comes from the rank-deficient OLS fixture. No performance benchmark is claimed for this import.
