# FE complexity frontier benchmark — prototype v0.1

This is an isolated research benchmark, outside the production `econhdfe` API.
It connects three-way categorical FE families with explicit spectral structure
to actual projection backends and an independent projection-error oracle.

The question it can test is: **at a fixed requested tolerance and work budget,
which implementations meet the same forward-error target, and what setup and
projection work do they pay as rank and spectral difficulty grow?** It does
not establish a universal complexity theorem or an optimal time-to-accuracy
comparison. Failed and timed-out attempts stay in every result file.

## What is implemented

- Five three-way families with `G = Theta(N)` when the replication factor is
  fixed: coupled grids, directional rigidity patches, unique-tuple rigidity,
  a more ill-conditioned cycle, and a random connected control.
- Two independent witnesses for the oracle: weighted group cancellation and
  supplied categorical coefficients reconstructing `y - truth`. Small cases
  also use a separately assembled dense least-squares reference.
- Multi-scale targets and a `broadband` target whose Fourier support grows
  with problem size. Pure finite-mode `slow` targets are explicitly diagnostic.
- Four explicit backends: instrumented SciPy LSMR, pinned upstream within via
  a common adapter, the main-source EconHDFE native auto route, and pinned fereg
  via the same common adapter. No complete regression API substitutes for FE
  projection. No native-to-other-kernel fallback is enabled in the dual adapter.
- Separate setup, projection, oracle and whole-process clocks; real operator
  counts where exposed; native opaque counts remain `null`.
- A fresh child process per workload/backend/trial, preserved failure records,
  finite worker deadlines, binary hashes and source hashes.

Read [FAMILY_DESIGN.md](FAMILY_DESIGN.md) for constructions, rank arguments,
spectral promises, weighting and target limitations. Read
[BOOTSTRAP.md](BOOTSTRAP.md) for pinned native-wheel provenance and setup.
The [first-run evidence](results/README.md) preserves all 60 measured records,
including the two original accuracy failures and their separate precision
follow-ups. It identifies the exact measured code and binary sources.

## Quick start

From the repository root, using the isolated environment in BOOTSTRAP.md:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 ../fe-frontier-env/bin/python -m pytest benchmarks/hdfe/frontier_v1/tests -q
../fe-frontier-env/bin/python benchmarks/hdfe/frontier_v1/run.py \
  --families rigidity_grid kernel_grid rigidity_grid_unique \
  --sizes 8 16 --rhs-kind broadband \
  --backends scipy_lsmr within native_auto fereg \
  --output benchmarks/hdfe/frontier_v1/results/smoke.json
```

To run without the optional native binaries, explicitly select
`--backends scipy_lsmr`. Missing requested native binaries are recorded as
unavailable; they never become successful SciPy runs.

### Rank-growing, condition-controlled experiment

```bash
../fe-frontier-env/bin/python benchmarks/hdfe/frontier_v1/run.py \
  --families rigidity_grid --sizes 32 64 128 256 \
  --rhs 2 --rhs-kind broadband --repeat 1 --weight-ratio 1 \
  --backends scipy_lsmr within native_auto fereg --threads 2 \
  --tol 1e-10 --forward-tol 2e-8 --max-iter 2000 --timeout 180 \
  --output benchmarks/hdfe/frontier_v1/results/rigidity_scaling.json
```

Here `size` is the periodic grid side. Actual `N` is 20,480; 81,920;
327,680; and 1,310,720, with `G=6*size**2` and normal rank `G-2`.
The family analysis gives `kappa_FE <= 12*size**2 < N`, with condition number
growing in order as `N`. This is a structured subclass of that promise, not
an arbitrary input from the entire promise class.

`--timing-repeats 3` runs fresh-process repeats and alternates backend order.
The default is one trajectory, which does not supply confidence intervals.

### Orthogonal control axes

- Change `--families` to `kernel_grid` or `rigidity_grid_unique` to remove
  duplicate tuple compression as an explanation for performance.
- For `kernel_cycle`, `size` means module count, not grid side. Its condition
  number grows in order as `size**2`; it is outside the `kappa_FE <= N` promise.
- Change `--rhs-kind` to `mixed` or `random` while keeping graph parameters
  fixed. Do not fit a complexity exponent across different target kinds.
- Change `--repeat` only for a compression experiment. Increasing replication
  adds observations but does not grow rank.
- Change `--weight-ratio` only for a weight-conditioning experiment. The true
  residual is reconstructed for those weights. The unweighted spectral promise
  is replaced by the declared weight-ratio comparison bound.

## Accuracy and timing contracts

The common external target is

```
||returned - truth||_W / ||y||_W <= forward_tol.
```

It is not relative to `||truth||_W`, a coefficient error, a native stopping
flag, or a normal-equation residual. Both fit and truth have half of target
energy, avoiding an artificially easy already-residualized target.

The same requested `tol` does not force the backends to stop at identical
actual accuracy. A failure at the fixed tolerance/budget means that setting
failed; it does not prove that tighter settings could not meet the target.
Full time-to-target tuning is a separate follow-up experiment.

`solver.total_s` includes common input validation/layout, solver construction,
the projection call, internal certifications and output recovery. Delayed
preconditioner construction can occur in `solve_s`; it remains in total cost.
Imports, binary hashing, data generation, independent oracle verification and
small dense spectrum diagnostics are outside this clock and reported separately.
The entire worker wall clock includes them. Whole-worker peak RSS includes the
fixture and verification buffers and must not be called solver-only memory.

`timeout` is a deadline for the complete worker. Timeouts are censored failures,
not solved observations at the deadline. Report their last completed phase.
The native `max_iter` scopes differ: see BOOTSTRAP.md and recorded metrics.
Iteration counts are never relabelled matrix-vector-product counts.

### Native iteration counts and strategy changes

For `fereg`, the pinned core solves the RHS columns in a lockstep batch and
returns one `SolveInfo.iterations` value. The dual wrapper repeats that value
for each RHS: `[7, 7]` is one shared batch count, not two independently measured
seven-step solves. **Do not sum the array as actual matvec work.** If the wrapper
makes a corrective projection, its shared count is added to each entry.
The core keeps its cumulative counter across adaptive restarts: earlier PCG
steps are included, while an exact-factor correction counts as one additional
step. Factor assembly, symbolic/numeric factorization and extra residual
matvecs at restart are not separately represented by this counter. Their costs
remain inside the complete projection call and `solver.total_s`.

The recorded `rigidity_scaling.json` trajectory crosses a specific policy
boundary. Each FE dimension has `2*ell**2` categories. After eliminating the
first dimension, the other two total 4,096 categories at `ell=32`, exactly the
pinned fereg `small_limit=4096`; both are classified as small. At `ell=64` each
remaining dimension alone has 8,192 categories, so both are large. The extra
assembly-work gate does not demote the small blocks in this family. At ell32
the cost model defers the small exact factor, and the observed 139 iterations
with no wrapper correction imply that no small-factor upgrade occurred.

For ell64 and above, the large-block path can choose an exact joint sparse
factor at the sixth or twelfth PCG step. The reported seven steps are consistent
with six PCG steps followed by one exact correction after a joint-factor
upgrade. **That upgrade remains an inference:** the archived wrapper omits the
core's `exact`, `escalated` and `precond_kinds()` diagnostics, so the result file
does not directly identify the factor used. Seven ordinary PCG steps cannot
be excluded from the counter alone. A constant reported count across these
sizes therefore does not establish constant operator work, a single unchanged
solver path, or a universal complexity improvement.

These definitions follow the pinned
[dual wrapper](https://github.com/asggvhwgw-glitch/EconHDFE/blob/23a0c12f04983514a64e7cc9b10145d1e34c584e/native/dual-absorber/src/lib.rs),
[fereg PCG counter](https://github.com/leoliu0/fereg/blob/0fa59e3b539addbddaee8bd71acd031c497166a0/crates/fereg-core/src/cg.rs),
[structure classification](https://github.com/leoliu0/fereg/blob/0fa59e3b539addbddaee8bd71acd031c497166a0/crates/fereg-core/src/structure.rs)
and [factor policy](https://github.com/leoliu0/fereg/blob/0fa59e3b539addbddaee8bd71acd031c497166a0/crates/fereg-core/src/solve.rs).

## Result structure and reproducibility

The runner writes one strict JSON report and a unique directory of child
requests, child results, phase checkpoints and logs. Reports are atomically
updated after every attempt. Reusing an output filename never reuses old child
results. Nonzero process exits and request-mismatched outputs cannot pass.

Each record includes actual row/category counts, topology/target metadata,
the declared analytic condition bound, the small-instance numerical spectrum
when within budget, oracle checks, solver metrics, independent output accuracy,
and a status. Large numerical spectra are explicitly unavailable; analytical
family bounds are not relabelled measured condition numbers.

The measured code and binaries are identified by hashes. Raw arrays are
regenerated from the family, parameters and random-stream seed; they are not
committed as multi-million-row data files. Production solvers and default
routing are not modified by this benchmark.

## Limits of this version

The family templates and coordinates are provided by the generator. Generic
backends receive only categorical labels, weights and targets. This tests how
they handle the structure; it does not solve universal decomposition discovery.

Native internal matvec counts, factor fill, preconditioner applications and
per-stage peak memory are not yet exposed. SciPy is a useful instrumented
reference but uses a single-threaded Python/NumPy categorical operator, so its
absolute time is not a language-neutral algorithm ranking.

No outer OLS/IV/PPML/IV-PPML estimator is timed here. A single fixed-weight FE
projection result does not establish the complexity or inferential correctness
of a complete nonlinear estimator.
