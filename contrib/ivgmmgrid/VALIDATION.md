# Contribution validation

Validated 2026-09-25 on Windows, Python 3.12, NumPy 2.5.3, SciPy 1.18.1, with BLAS/OpenMP capped at one thread. Fresh virtual environment with system packages disabled; dependencies resolved from the package index and all three Python distributions installed with isolated PEP 517 builds. No user datasets were used.

EconHDFE core suite: 905 passed, 1 optional test skipped (SymPy unavailable), no failures. Each independent branch passed `python scripts/version.py gate` and `python scripts/generate_architecture_map.py --check`. New companion checks are intentionally separate from parent test discovery.

A fresh hidden Stata/MP 19 session ran the shipped `examples/ivgmmgrid_example.do` and `checks/ivgmmgrid_contract.do`. Both passed, with terminal markers `IVGMMGRID_EXAMPLE_OK` and `IVGMMGRID_GENERIC_CONTRACT_OK`. The checks cover native full-grid parity, if/in sample restrictions, different candidate samples, data order/content preservation, collinearity fallback, empty-sample error propagation and cleared e() state.

Native stack: ivreghdfe 1.1.4 (`bfb5577a6dbdfb029ab4ab6a7e93f7257a827b42`), reghdfe 6.14.1, ftools 2.50.0, require 1.4.0, ivreg2 4.1.12, ranktest 2.0.04 and moremata. No Stata executable or upstream dependency source is bundled. Re-run both shipped do-files after installing/changing native dependencies, in a fresh Stata session. GitHub-hosted CI does not execute licensed Stata.
