# FE selector 0.5: bounded continuation, not a stronger FE estimator

## Scope

The four original hyperFE mathematical kernels are unchanged. The target remains
the original positive-weight pure-categorical projection. No estimator default,
sample selection, singleton, degrees-of-freedom, covariance or remote repository
is changed. This is an explicit research backend, not a production release.

## Sample evidence

The 0.4 group scout still runs within its configured resource budget. A raw
`reject` is recorded, then mapped to effective `unknown`; it cannot alone veto
full-data candidate evaluation or cancel a marginal warm confirmation. Resource
and full-data numerical gates remain. Sampling therefore still incurs overhead;
this revision does not claim that advisory sampling is free or useful everywhere.

## One live LSMR recurrence

The useful-column pilot uses the SciPy 1.17.0 recurrence with a scalar observer.
Every eight iterations it records the existing backward-error stopping metrics,
elapsed time, and conservative continuation estimates. No restart or repetition
of the initial 24 iterations occurs. Standard SciPy stopping codes have priority;
a cost-controller stop has distinct code 8 and is NEVER accepted as convergence.
All accepted outputs still pass the original-target audit. An interrupted local
fit is reconstructed as a feasible partial residual and finished by native within.

At a checkpoint, the gap to the original stopping rule and its recent median
log-contraction estimate predict remaining work. Continuation considers actual
remaining RHS, native warm cost, and a gain margin. Construction costs are sunk
for this decision but remain included in reported end-to-end timings.

A geometric forecast can be too pessimistic during Krylov transients. The first
implementation reproduced this failure on the old 83131 counterexample. The
archived development_source_A and logs retain it. With sustained progress, the
final rule can buy one further checkpoint while total exploratory time fits 35%
of one measured native RHS. The overall pilot is capped at 90% of that native RHS
and 5 seconds (whichever smaller), plus checkpoint/operation granularity. This is
not a hard process wall-time guarantee: individual sparse/native operations are
not preempted. Normal maxiter and finite-precision gates are unchanged.

These policy constants are explicit heuristics, not a convergence theorem or a
calibrated expected-value model. They can still mispredict future convergence,
miss useful candidates, or select a route that does not amortize setup.

## Provenance

The monitor adapts SciPy 1.17.0 LSMR. Source snapshot, hash, authorship and BSD
license are retained under third_party/. With observer always continuing, tests
compare the solution, stopping status, iteration count and scalar outputs exactly
to the installed upstream version. This is not a new LSMR method or a rewrite of
the native within Rust kernel.

## Validation design

Known old seeds 83131/83132 and 73021 are regression/development cases, not holdout.
Formal policy is frozen before seeds 93517/93518. These remain existing synthetic
families, not independent evidence of real-data generalization. All performance
inputs have one million rows; cold setup, failed attempts and runtime audits are
included. Native controls directly call within-py 0.4.0 on identical arrays.

The original 0.4 sample-hard-veto test is intentionally replaced by a test that
negative evidence cannot bypass full-data checks. The cache test now establishes
native rejection from an explicit full-data rejection, not from sampling. No
numerical accuracy threshold was loosened. The complete former tests are retained
in frozen_v04/tests for audit and do not enter current test collection.
