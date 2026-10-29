# fereg 0.5.0 vs within-py 0.4.0: FE topology benchmark (2026-10-29)

**Status:** All 50 solver/size runs valid. **Source SHA:** benchmark run `2fb4f5864843a45131c76877a6e8104d866cc569`. **GitHub Actions:** [run 40805626532](https://github.com/asggvhwgw-glitch/EconHDFE/actions/runs/40805626532). **Source:** [benchmark script](https://github.com/asggvhwgw-glitch/EconHDFE/blob/bench/fereg-within-topologies-20261029/benchmarks/hdfe/fereg_within_topologies.py).

## Setup and fair-comparison boundary

- fereg 0.5.0: Rust source compiled at `0fa59e3b539addbddaee8bd71acd031c497166a0`, Python `fereg.feols`, including formula/FE processing and iid variance estimation.
- within-py 0.4.0: native Python `within.Design` + `within.Solver.solve_batch` (two RHS for y and x), minimal weighted/unweighted FWL OLS slope, **without** variance estimation.
- within default = diagonal-to-adaptive Schwarz; within_additive = forced eager additive Schwarz.
- Linux GitHub-hosted Ubuntu 24.04: 4 visible logical CPUs, 15 GiB memory; **2 Rayon threads**, 1 NumPy BLAS thread; Python 3.12.15. Note runner CPUs vary across topology cases (AMD EPYC 9V45, 9V74, and 7763), while each comparison **within a topology** ran on the same runner VM.
- Synthetic data generated once per process with identical deterministic seed, all tests y ~ x | FE. Observations: 120k / 1M; 5M for four graph classes. All FE classes constructed to minimize singleton confounding. The `reweight` case uses 3 changed weights; others are unweighted.
- Time excludes data generation and import; fereg one complete fit vs within owned design construction, preconditioner setup, batch projection and the light FWL slope calculation. The latter FWL slope cost is not included in the rounded within timing, but is very small and recorded as scope limitation. The two figures are **not kernel-only equivalence**.
- Each case uses a separate runner VM, so ratios must not be compared across cases as a machine-fixed performance regression.

**Ratio definitions:** below `within/fereg >1` means fereg is faster; `<1` means within is faster.

## Cold / first solve (seconds, including setup)

| Topology | Observations | fereg full OLS | within minimal FWL | within/fereg | Peak RSS MB fereg/within |
|---|---:|---:|---:|---:|---:|
| one_big | 120,000 | 0.0156 | 0.0134 | 0.86× | 113/71 |
| one_big | 1,000,000 | 0.0815 | 0.1105 | 1.36× | 281/251 |
| one_big | 5,000,000 | 0.5013 | 0.6510 | 1.30× | 1024/865 |
| two_big | 120,000 | 0.0351 | 0.0234 | 0.67× | 115/69 |
| two_big | 1,000,000 | 0.2446 | 0.2249 | 0.92× | 294/232 |
| two_big | 5,000,000 | 1.8048 | 1.5826 | 0.88× | 1159/878 |
| three_big | 120,000 | 0.0335 | 0.0192 | 0.57× | 114/68 |
| three_big | 1,000,000 | 0.3920 | 0.2143 | 0.55× | 315/235 |
| three_big | 5,000,000 | 3.2415 | 2.1421 | 0.66× | 1103/857 |
| weak_links | 120,000 | 0.0692 | 0.0695 | 1.01× | 151/88 |
| weak_links | 1,000,000 | 0.8397 | 1.1578 | 1.38× | 550/500 |
| weak_links | 5,000,000 | 1.0456 | 9.2115 | 8.81× | 1017/2282 |
| hub | 120,000 | 0.0332 | 0.0240 | 0.72× | 113/65 |
| hub | 1,000,000 | 0.1914 | 0.2059 | 1.08× | 289/220 |
| repeated_cells | 120,000 | 0.0270 | 0.0438 | 1.62× | 111/63 |
| repeated_cells | 1,000,000 | 0.1318 | 0.4099 | 3.11× | 270/210 |
| reweight | 120,000 | 0.0410 | 0.0389 | 0.95× | 121/71 |
| reweight | 1,000,000 | 0.3038 | 0.3524 | 1.16× | 348/268 |

## Repeated OLS solve (median of repeats; seconds)

Within retains the constructed `Design` and same-weight `Solver`; fereg executes new `feols` estimations each call. This is a *workflow-level persistence comparison*, not a fixed preconditioner-reuse comparison of identical low-level APIs. For `reweight`, both rebuild weight-dependent numerical state and the weights change each round.

| Topology | Observations | fereg | within | within/fereg |
|---|---:|---:|---:|---:|
| one_big | 120,000 | 0.0092 | 0.0084 | 0.90× |
| one_big | 1,000,000 | 0.0681 | 0.0802 | 1.18× |
| one_big | 5,000,000 | 0.4446 | 0.5479 | 1.23× |
| two_big | 120,000 | 0.0238 | 0.0188 | 0.79× |
| two_big | 1,000,000 | 0.2267 | 0.1856 | 0.82× |
| two_big | 5,000,000 | 1.8612 | 1.4142 | 0.76× |
| three_big | 120,000 | 0.0225 | 0.0147 | 0.65× |
| three_big | 1,000,000 | 0.3653 | 0.1803 | 0.49× |
| three_big | 5,000,000 | 3.3828 | 2.0107 | 0.59× |
| weak_links | 120,000 | 0.0526 | 0.0226 | 0.43× |
| weak_links | 1,000,000 | 0.8079 | 0.3178 | 0.39× |
| weak_links | 5,000,000 | 1.0479 | 3.0447 | 2.91× |
| hub | 120,000 | 0.0254 | 0.0198 | 0.78× |
| hub | 1,000,000 | 0.1792 | 0.1683 | 0.94× |
| repeated_cells | 120,000 | 0.0168 | 0.0394 | 2.35× |
| repeated_cells | 1,000,000 | 0.1132 | 0.3743 | 3.31× |
| reweight | 120,000 | 0.0303 | 0.0334 | 1.10× |
| reweight | 1,000,000 | 0.2860 | 0.2992 | 1.05× |

## Precision and validity

- Invalid records: **0 / 50**.
- Largest independent normalized FE orthogonality RMS across all repetitions: **3.9821e-9** (absolute scale-free moment score); largest normalized slope score: **2.1609e-11**.
- Maximum absolute fereg-vs-within first-solve coefficient difference: **5.4401e-15**.
- All measured moments pass the stricter post-hoc `5e-8` threshold, not merely the benchmark script's `3e-6` fail threshold. This is a first-order normal-equation certificate; it is not a forward-error bound in near-singular cases.
- Weights are positive in every sample and no groups are intentionally singletons. The reduced fixed-effect coefficient normalization does not affect the estimand.

## Interpretation for an EconHDFE solver planner

1. **One large FE, two small FEs:** fereg wins modestly at 1M and 5M cold starts, consistent with cheap global elimination of the large group. Very small (120k) cases can favor within due to setup overhead.
2. **Two or three comparably large FEs with well-mixed connections:** within default wins in 1M/5M cold starts; three-large FE wins by about 1.5–1.8×. Prefer within adaptive diagonal by default here; avoid eager Additive Schwarz.
3. **Weakly connected two-way network:** fereg wins most decisively at 5M; fereg 1.05s versus within 9.21s cold, about 8.8×. This may reflect topology/sparsity, memory and convergence/preprocessing interactions, not just Krylov iterations. Within's reused same-weight solver becomes faster at 1M, so persistence is important.
4. **Repeated full FE tuples:** fereg gains around 3.1× cold at 1M compared with *raw within*. The EconHDFE native wrapper already implements tuple compression: **do not extrapolate this to existing `strategy=auto` without benchmarking the wrapper**.
5. **Hub-heavy degree structure:** near tie at 1M, few percent; no conclusive route decision at 2 threads. Prefer a low-overhead probe / robust default.
6. **Reweighting (PPML-like but not PPML):** small advantage to fereg for three independent weighted OLS fits, but these tests **do not exercise fereg FeSolver::set_weights across actual PPML IRLS**, and are not a true PPML benchmark.
7. **Eager Additive Schwarz:** up-front preconditioner setup can overwhelm iteration reduction (1M three-large: 4.74s vs default within 0.214s). Use adaptive escalation rather than unconditional Schwarz.

### Candidate planner policy, not yet production validated

- Gather effective FE group sizes after singleton/rank filtering; estimate eliminate-group choices by `K_remaining`, local cross-neighbor squared-degree / Schur fill `E_j`, full-tuple compression factor `U/N`, connectivity and hub concentration.
- If `K_remaining <= 128` and conditioning permits, retain EconHDFE's exact `SmallSchur`. If strongly skewed, consider a fereg-like general reduced PCG.
- If 2–3 FE groups remain large and graph mixing is high, prefer within default adaptive solver (tested strong case).
- If weakly connected with sparse reduced cross-coupling, include fereg-like elimination / PCG as a first-class candidate; validate FE moments independently.
- If `U/N <= 0.7`, compare fereg against EconHDFE's existing native `tuple` branch before selecting.
- Always account for number of RHS and reuse across weights: a one-shot benchmark does not characterize persistent `Design`/IRLS pathways.
- Do not merge or replace existing within on this benchmark alone; acquire direct same-API native-core and PPML parity data.

## Known limitations

Only one regressor and two projected RHS; all data synthetic; seven specific topologies; identical two-thread settings but different hosted CPU SKUs across scenarios. No direct fereg-core residualization API is exposed in the public Python package; no direct EconHDFE `NativeSolver(strategy="auto")` benchmark here; no end-to-end EconHDFE PPML/IVPPML; changed-weight fit test does not reuse a fereg solver state across fits. Selected 10+ RHS and pathological ill-conditioning remain untested.

Raw machine-readable results: [JSON](fereg_within_topologies_results.json). Full [GitHub Actions job logs](https://github.com/asggvhwgw-glitch/EconHDFE/actions/runs/40805626532).
