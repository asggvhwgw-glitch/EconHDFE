# Experimental FE selector v0.5 (research contribution)

The independently maintained, **opt-in** selector lives at
[`contrib/hyperfe_local_coarse/`](../../../../contrib/hyperfe_local_coarse/).
It builds on the separate hypergraph local/coarse solver PR #27 and should be
reviewed only after that branch. Neither the econometric estimand nor the
default FE backend is changed.

## Algorithm

- Safely pass through no-FE, single FE, and ineligible designs; maintain
  the existing original-row sample, weight, singleton, and inference contracts.
- Use native `within-py==0.4.0` for a first real RHS and potentially a
  second warm RHS near the decision boundary.
- Generate bounded normalized-star block partitions and an optional balanced
  global coarse space. Allocate under conservative *known-workspace*
  estimates; these are not hard process-RSS ceilings.
- The group-sampling scout is **advisory**: negative evidence is downgraded
  to `unknown`, never a proof about the full data's rank or solver quality.
- A cost-controlled, continuation-preserving monitored SciPy LSMR pilot
  checkpoints every eight iterations. A pilot stopped for cost reasons
  carries distinct status `8` and is not accepted as converged.
- When a trial does not finish, finish a feasible partial residual on
  the original model with native `within` and rerun original-row auditing.

## Provenance

`monitored_lsmr.py` is an **adaptation of SciPy 1.17.0**, not clean-room
numerical code; see the upstream snapshot, provenance metadata, and full
SciPy license in `contrib/hyperfe_local_coarse/third_party/`.
Balanced coarse spaces, Schur complements, and additive Schwarz are
established numerical methods, not claimed as original inventions here.

## Evidence / limitations

187 archived unit tests passed and independently rerun in the submission
environment. Million-row synthetic results include 54 main, 26 supplemental,
and 6 independent-reference tasks with all passed numerical checks, but
not 86 independent real-world datasets. Retain rejected candidates and
explicit slower examples in `evidence/selector_v05_summary.json`.

Improvements in identified old counterexamples are **regression tests**, not
unrelated holdouts. The two profitable synthetic case families share a
designed multiway additive relationship. Correctness beyond intercept-only
categorical FE and positive finite weights, varying-slope FE, PPML weight
updates, GPU and complete OLS/IV integration is not certified. The new
selection policy is **not** enabled in `econhdfe`'s default backend.

Run `python -m pytest tests -q` from `contrib/hyperfe_local_coarse/`
after installing `selector_requirements.txt`.
