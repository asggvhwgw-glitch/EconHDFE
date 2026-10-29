# First-run evidence: 2026-10-29

The four JSON reports in [2026-10-29/](2026-10-29/) are preserved without
changing any recorded request, outcome, timing or provenance field. They
contain all 60 formal observations from the prototype's first experiment:

| Report | Purpose | Records | Passed | Accuracy failed |
|---|---|---:|---:|---:|
| [smoke.json](2026-10-29/smoke.json) | Three grid families, sides 8 and 16, four backends | 24 | 24 | 0 |
| [rigidity_scaling.json](2026-10-29/rigidity_scaling.json) | Rigidity grid sides 32, 64, 128 and 256 | 16 | 14 | 2 |
| [unique_controls.json](2026-10-29/unique_controls.json) | Two all-unique tuple families, three sizes, three native backends | 18 | 18 | 0 |
| [precision_followup.json](2026-10-29/precision_followup.json) | Tighter LSMR tolerance for the two failed settings | 2 | 2 | 0 |
| **Total** | | **60** | **58** | **2** |

The two original LSMR failures remain failures. Passing at a tighter requested
tolerance does not retroactively change those records. All 48 native-backend
records passed the external forward-error gate. There were no timeout or
unavailable-backend observations in these four reports.

## Largest rigidity-grid case

At side 256, `N=1,310,720`, `G=393,216`, `rank=G-2`, and the declared analytic
bound is `kappa_FE <= 786,432 < N`. Two RHS columns are projected together.
The common gate is the maximum per-column `||returned-truth||_W / ||y||_W`
at most `2e-8`.

| Backend | Requested tol | Setup + solve (s) | Max forward error | Status |
|---|---:|---:|---:|---|
| native_auto | 1e-10 | 2.613632 | 5.026528e-12 | passed |
| within | 1e-10 | 2.953359 | 1.530540e-13 | passed |
| fereg | 1e-10 | 3.385536 | 1.682416e-12 | passed |
| scipy_lsmr | 1e-10 | 45.834186 | 9.293716e-8 | accuracy_failed |
| scipy_lsmr, follow-up | 1e-12 | 54.584401 | 9.214811e-10 | passed |

These are one-run implementation measurements, not a language-neutral or
thread-neutral algorithm ranking. Native backends use two threads; the
Python/NumPy categorical LSMR operator uses one effective thread. Actual
stopping errors differ considerably. No confidence intervals or universal
complexity exponents are inferred.

Setup plus solve includes adapter preparation, construction, the full native
call, internal corrective projections, delayed factorization and output
recovery. Imports, DGP, provenance hashing and independent validation have
separate clocks. Peak RSS covers the entire isolated worker, including
fixture and oracle buffers, not just solver memory.

## Coverage and source identity

- All reports use seed `20261029`, `broadband`, two RHS, unit weights,
  `repeat=1`, and one timing trial per setting.
- The two all-unique tuple controls reach `N=U=1,048,576`. Their 18 native
  measurements all pass. This rules out duplicate compression as the sole
  explanation for the measured performance on these structured families.
- Cycle, random-graph, changing-weight, replication and outer-estimator
  performance are not covered by these four reports.
- The exact measured benchmark commit is
  `b9127b174c10b065deee1fc68ad0a64ffb928e85`. Subsequent changes before
  publication only add documentation and these archived records. Every
  executable `.py` source hash in the reports still matches the published
  prototype.
- Native sources, upstream pins, wheel hashes and build provenance are
  documented in [BOOTSTRAP.md](../BOOTSTRAP.md). Native binaries are not
  committed in this PR.
- The machine was an AMD EPYC 9V74 host with an 8 GiB container memory limit
  and 8 CPU quota, CPython 3.12.14, NumPy 2.3.5 and SciPy 1.17.0. Environment
  details are retained in each report.

The parent reports include every child result and request required to inspect
the measurements. Relative `worker_log` names are preserved as original run
metadata; the separate worker directories are not committed here. The complete
worker request, result, phase and log archive is retained in the separately
delivered prototype bundle. Large numerical input arrays are regenerated from
the recorded parameters and seed.

## Reproducing and preserving evidence

Use the commands in [README.md](../README.md) and [BOOTSTRAP.md](../BOOTSTRAP.md).
The runner's normal output filenames are ignored in this directory, so reruns
do not overwrite the dated archive. Put a deliberate new evidence snapshot in
a new dated directory when its measurement context changes.

The original main-series command is expected to exit nonzero when its LSMR
accuracy failures recur. Use a separate output path with `--backends scipy_lsmr
--sizes 128 256 --tol 1e-12` to reproduce the successful follow-up; do not remove
or replace the failures.

The native shared-iteration and strategy-boundary caveats are explained in
[the main README](../README.md#native-iteration-counts-and-strategy-changes).
In particular, fereg's `[7,7]` is a replicated shared batch counter and does not
measure all factorization or matvec work. These observations are evidence for
the tested structured subclasses, not a universal FE complexity improvement.
