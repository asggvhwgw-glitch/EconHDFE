> 本文件是 2026-09-14 的历史执行记录，以下计数不是当前默认或完整套件。
> 0.6.5 的新测试结构及命令见 [testing.md](testing.md)，迁移见 [test-migration-0.6.5.md](test-migration-0.6.5.md)。

# econhdfe 0.6.3 local candidate validation — 2026-09-14

**Executed local result: 906 passed / 0 failed / 0 skipped**, with 10 existing omission warnings. The installed-host subset has **350 passed / 0 failed / 0 skipped**.

This release integrates PERF-01 only plus roadmap/release-test tooling. Actual
results are in `../release/evidence/0.6.3/validation-summary.json` and the accompanying
outer evidence archive. A required threshold is not itself a successful run.

Unchanged 0.6.2 was rerun: 807 passed, 0 failed, 0 skipped. The candidate adds
52 reduction/projection/weight/stopping tests and 47 execution-gate/thread-launcher
checks, so a complete run contains 906 cases. All 63 previous test files are
retained without changing assertions or marking xfail.

The first integrated run had 905 passed and one stale architecture inventory
failure: execution/performance documents were added after the generated map.
The original failure log is retained. Final packaging freezes the inventory
before re-testing. No numerical assertion was weakened to fix that failure.

The installed-wheel suite remains the previous five independent suites (298
cases) plus an explicit `--extra-test test_weighted_colsum_fusion.py` (52 cases).
Those 350 tests run outside the source tree and validate import provenance;
they use HOST dependencies and do not certify clean dependency installation.

Runtime/validation fingerprint and evidence-file SHA256 links are separate from
maintenance REVIEW entries. The shipped execution record stays a candidate;
formal authorization requires missing environment results and detached final
artifact/commit identity. Synthetic tests of a fully passed manifest are not
real external validation evidence.

Environment: Linux x86_64 / Python 3.13.5 / NumPy 2.3.5 / SciPy 1.17.0 /
Numba 0.65.1 / pandas 2.2.3 / statsmodels 0.14.6 / SymPy 1.14.0 / pytest 9.0.2.
Functional tests use a Numba maximum >=3 (this run 4), separate from single-thread
benchmark processes. Existing omission warnings remain. Current coverage is not
remeasured; the 0.6.2 coverage percentages are historical, not new claims.

See the per-release performance document for the two fresh-process rounds per
version/case, all timed repetition medians and scope limits. No 10M/GPU/remote
matrix, full external parity or third-party mathematics/originality certification
is inferred from these local tests. Prior mathematical TEX/PDF files are unchanged.
