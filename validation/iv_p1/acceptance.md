# Local P1 acceptance — 2026-10-28

Final complete suite: **636 passed, 1 skipped, 4 warnings**, 61.21 seconds.
The skip requires the optional native within extension. New numerical cases:
**47 passed**; outside-source runtime-copy tests: **47 passed**, using existing
host dependencies, not an isolated installation.

The first full run is preserved (634 passed, 2 failed, 1 skipped): an architecture
map needed regeneration and a model-layer import violated layer ownership.
The adapter's numerical AST is identical after moving the new client to its
proper lower-layer copy; no Monte Carlo formulas or settings were altered.

Formal Monte Carlo: **14000 datasets**, 1000 repetitions per cell, all completed.
No numerical failures are silently removed. The panels with invalid long-panel
rates and the serial-dependence counterexample remain in the summary.
A 6.4% rejection case is not described as exact nominal 5% calibration.

Large-sample timing: 1M/5M one-way prepared tests and 1M three-FE restricted-null
test, three repeats. These are one-null evaluations, not complete IV estimates.
Raw times, separate preparation cost and process peak RSS are retained.

This checkpoint does not replace existing estimators or covariance defaults,
change the package version, push PR source, integrate current main, or establish
new Windows/macOS/licensed-Stata certification. See the theoretical assumptions
and counterexamples before using any new test.
