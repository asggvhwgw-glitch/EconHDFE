# Provenance and validation notice

`econhdfe` (formerly distributed as `pyreghdfe`) is implemented as a Python-native clean-room codebase. Public documentation, papers, command help and source behavior from the `reghdfe` / `ivreghdfe` / `ivreg2` ecosystem are used as econometric compatibility references; external source files are not vendored into this repository.

The project names `reghdfe`, `ivreghdfe` and `ivreg2` refer to independent Stata software maintained by their respective authors. Users requiring exact command parity should run the version-pinned golden harness in `validation/` against their licensed Stata installation.

The bounded Stock–Yogo critical-value subset in `stock_yogo.py` contains published/tabulated numerical research constants and was cross-checked against a public current representation used for Stata-compatibility testing. The function reports its limited coverage and the fact that these cutoffs are intended for homoskedastic Cragg–Donald F statistics.
