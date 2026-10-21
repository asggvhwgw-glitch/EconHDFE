# Functional tests and performance measurements

Use `python scripts/run_tests.py --suite full -- -q --junitxml=results.xml`
for the full suite. Without `--suite full`, the default is the daily core only.
The launcher defaults Numba's maximum thread pool to four, and BLAS/OMP/MKL to
one when those variables are absent. Explicit caller settings are retained.
An explicit Numba cap below three is rejected before importing Numba: the
functional suite exercises real 2–3-thread execution. None is skipped.
A single-thread performance cap is therefore NOT a functional-test environment.

`python scripts/run_tests.py --preflight` inspects those prerequisites without
claiming test execution. CI uses the same launcher. Direct `pytest` remains
possible under the documented environment; it is not silently monkeypatched.

For benchmark-only child processes, set OPENBLAS_NUM_THREADS, MKL_NUM_THREADS,
OMP_NUM_THREADS and NUMBA_NUM_THREADS to 1 before importing the package. Do not
export that last cap globally and reuse it for the complete functional suite.
Always separate warmup/JIT compilation, timed repetitions and profiling.

Numba documents that NUMBA_NUM_THREADS fixes the maximum thread pool before
initialization, while set_num_threads masks threads within that maximum:
https://numba.readthedocs.io/en/stable/user/threading-layer.html

This launcher validates declared configuration, not availability of every
hardware platform or optional thread backend. No missing support is counted
as a successful test.
