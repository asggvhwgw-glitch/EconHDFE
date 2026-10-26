<!-- econhdfe architecture note -->
> **Lineage note:** this document records validation/benchmark work from the pre-rename `pyreghdfe` lineage. The `econhdfe 0.2.0` package preserves the numerical implementations and regression tests but did not rerun every historical external/performance matrix. Treat the figures below as inherited evidence until the corresponding `econhdfe` benchmark is rerun.

# Upstream compatibility references (checked 2026-09-10)

`econhdfe` (formerly `pyreghdfe`) is a clean-room implementation. The files below were consulted as
behavioral/econometric references; their source code is not vendored here.

## reghdfe

Current help identifies **reghdfe 6.14.1 (08Jul2026)**; the package manifest has
`Distribution-Date: 20260708` and lists the current modular Mata files.

- https://github.com/sergiocorreia/reghdfe
- https://raw.githubusercontent.com/sergiocorreia/reghdfe/master/current-code/reghdfe.pkg
- https://raw.githubusercontent.com/sergiocorreia/reghdfe/master/current-code/reghdfe.sthlp
- https://raw.githubusercontent.com/sergiocorreia/reghdfe/master/current-code/FE.mata
- https://raw.githubusercontent.com/sergiocorreia/reghdfe/master/current-code/Factor_FE.mata
- https://raw.githubusercontent.com/sergiocorreia/reghdfe/master/current-code/Factor_Indiv.mata
- https://raw.githubusercontent.com/sergiocorreia/reghdfe/master/current-code/DoF.mata
- https://raw.githubusercontent.com/sergiocorreia/reghdfe/master/current-code/LSMR.mata
- https://raw.githubusercontent.com/sergiocorreia/reghdfe/master/current-code/LSQR.mata
- https://raw.githubusercontent.com/sergiocorreia/reghdfe/master/current-code/MAP.mata
- https://raw.githubusercontent.com/sergiocorreia/reghdfe/master/current-code/MAP_Accelerations.mata
- https://raw.githubusercontent.com/sergiocorreia/reghdfe/master/current-code/MAP_Transformations.mata
- https://raw.githubusercontent.com/sergiocorreia/reghdfe/master/current-code/Regression.mata

Group/individual behavior used by v0.5 comes primarily from `reghdfe.sthlp` and
`Factor_Indiv.mata`: group-level variables must be constant within group;
`individual()` requires `group()` and the individual FE must be listed in
`absorb()`; aggregation is `mean` or `sum`; the incidence factor implements
`mult()`/`mult_transpose()` and recursive bipartite 1-core singleton removal.
The current `DoF.mata` leaves `dof_update_individual_fe()` as a TODO, so v0.5
keeps the individual-FE DoF adjustment conservative.

## ivreghdfe / ivreg2

Current upstream source identifies **ivreghdfe 1.1.4 (29Nov2025)** and is based
on `ivreg2 4.1.11` with a `reghdfe` absorption hook.

- https://github.com/sergiocorreia/ivreghdfe
- https://raw.githubusercontent.com/sergiocorreia/ivreghdfe/master/src/ivreghdfe.ado

The current parser accepts a wildcard option set and appends those options to
the internal `reghdfe` construction when `absorb()` is used. The v0.5 golden
harness therefore includes a group/individual IV row, but external Stata
execution remains required before claiming command-level parity.

## Algorithm papers / documentation

- Sergio Correia, *Linear Models with High-Dimensional Fixed Effects: An Efficient and Feasible Estimator* (working-paper/software algorithm reference): http://scorreia.com/research/hdfe.pdf
- Current `reghdfe` help references Constantine and Correia for individual fixed effects with group-level outcomes.

The Python implementation intentionally substitutes Python/SciPy/NumPy/Numba/
CuPy engineering choices where they preserve the targeted column space or
estimator semantics.


## IV-PPML / SPJ references checked for 0.2.0

The IV-PPML implementation was checked against the current public `ekwonomist/ivppmlhdfe` v0.9.4 source/help and its distributed Class A/B/C SPJ/bootstrap templates. The upstream estimator uses additive moments `E[q(y-mu)]=0`, iteratively reweighted 2SLS and reghdfe concentration. Public syntax supports pweights/fweights, optional `standardize`, and `separation(all)` including in-loop `mu` detection. SPJ formulas and resampling units follow the companion Kwon–Larch–Yoon–Yotov (2026) materials.

Current upstream sources do not expose a dedicated IV-PPML weak-identification diagnostic analogous to linear KP/SW/Stock-Yogo. Those linear diagnostics are therefore intentionally not re-exported as IV-PPML statistics.

- https://github.com/ekwonomist/ivppmlhdfe
- https://github.com/ekwonomist/ivppmlhdfe/blob/main/ivppmlhdfe.ado
- https://github.com/ekwonomist/ivppmlhdfe/tree/main/data

## Technical-innovation audit references (0.4.6.1)

The 0.4.6.1 package-wide novelty audit uses these references to draw conservative claim boundaries. They are prior-art references, not vendored source.

### HDFE numerical methods

- Current `reghdfe` help documents MAP, CG acceleration, LSMR/LSQR and a degree-one `prune` option. Therefore these numerical primitives and leaf pruning **by themselves** are not econhdfe originality claims:
  - https://github.com/sergiocorreia/reghdfe/blob/master/current-code/reghdfe.sthlp
  - Sergio Correia, *Linear Models with High-Dimensional Fixed Effects: An Efficient and Feasible Estimator*.

The registered econhdfe solver-side claim is limited to the formal arbitrary-G categorical-hypergraph residual-core theorem and exact reconstruction/weight-validity conditions in `docs/technical/hdfe/numerical-residual-core/`.

### Functional dependencies / structure-aware learning

- Mahmoud Abo Khamis, Hung Q. Ngo, XuanLong Nguyen, Dan Olteanu, Maximilian Schleich, *Learning Models over Relational Data using Sparse Tensors and Functional Dependencies*, ACM TODS 45(2), 2020; arXiv:1703.04780.

This prior art prevents econhdfe from claiming functional dependencies or structure-aware regression simplification in general. The registered contribution is limited to the exact HDFE partition-refinement/canonicalization/design-reduction framework proved in `docs/technical/structural-design/`.

### PPML / separation and cluster inference

- Sergio Correia, Paulo Guimarães, Thomas Zylkin, *ppmlhdfe: Fast Poisson Estimation with High-Dimensional Fixed Effects*, Stata Journal 20(1), 2020.
- Sergio Correia, Paulo Guimarães, Thomas Zylkin, *Verifying the existence of maximum likelihood estimates for generalized linear models*, arXiv:1903.01633.
- James G. MacKinnon and Matthew D. Webb, wild-cluster-bootstrap work and related clustered-inference references.

Accordingly, econhdfe's PPML separation implementation and WCR/WCU cluster-inference layer are documented as established-method/compatibility implementations rather than package inventions.
