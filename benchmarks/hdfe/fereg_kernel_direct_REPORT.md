# Low-level fereg-core vs within: second-round benchmarks

- Source commit: `a182fb3744d3baaeff390d516c02c3765c9bb767`. Test date: 2026-10-29.
- [Full successful Actions run](https://github.com/asggvhwgw-glitch/EconHDFE/actions/runs/40809577543); direct Rust job 123236705909.
- Both pinned sources: fereg `0fa59e3b539addbddaee8bd71acd031c497166a0`, within `fc44282cea6124f5f18bb142908605a656d35a7c`.
- Both operate on the **same deterministic Rust-generated FE factors, responses, and weights**, inside the **same Linux runner**, with 2 Rayon threads and `1e-10` requested tolerance, 6000-iteration maximum.
- 7 graph topologies, 120k/1M rows for 2/8 RHS; 5M rows for 2 RHS in 5 scenarios. Direct kernel only, no formula parsing, statistical inference, Python estimators.
- Summary: **66/66 valid records; 66 process exits, 66 exited with 0**. Independent maximum scale-free weighted FE moment across all RHS and rounds **3.8755e-10**.
- Timing excludes synthetic data generation. fereg measures FE structure/setup and input row reordering separately; within measures persistent Design and Solver setup separately, before solution.
- **Critical comparability nuance**: fereg solves fused row-major RHS, while within solves a batch of RHS using shared preconditioning; both use their native optimal memory layouts. fereg input row reordering cost is included separately.
- 3 rounds at 120k and 1M; for weighted reweight structure, weights changed per round and numerical state reused/rebuilt as implemented. For other cases solver/preconditioner persisted.
- Both crates are called directly as Rust libraries; these are not EconHDFE's `NativeSolver(strategy="auto")` results. The latter has its own benchmark.
- Peak RSS is per Rust process. Ratios compare within the same n, RHS, graph structure and runner; first/cold timings may vary from run to run and cannot establish a worst-case complexity theorem.

## 1 million rows; 2 RHS, first solve

| Case | fereg build+marshal (s) | fereg solve (s) | within build (s) | within solve (s) | fereg gain on solve | 
|---|---:|---:|---:|---:|---:|
| one_big | 0.0457 | 0.0203 | 0.0159 | 0.1139 | 5.6219x |
| two_big | 0.0604 | 0.1520 | 0.0154 | 0.2166 | 1.4256x |
| three_big | 0.0741 | 0.4545 | 0.0169 | 0.2828 | 0.6223x |
| weak_links | 0.0506 | 1.0326 | 0.0142 | 1.6551 | 1.6029x |
| hub | 0.0535 | 0.0515 | 0.0138 | 0.1596 | 3.0989x |
| repeated_cells | 0.0255 | 0.0359 | 0.0146 | 0.3824 | 10.6520x |
| reweight | 0.0645 | 0.1546 | 0.0249 | 0.2555 | 1.6532x |


## 1 million rows; 8 RHS, first solve

| Case | fereg build+marshal (s) | fereg solve (s) | within build (s) | within solve (s) | fereg gain on solve | 
|---|---:|---:|---:|---:|---:|
| one_big | 0.0604 | 0.0334 | 0.0162 | 0.4318 | 12.9177x |
| two_big | 0.0840 | 0.4989 | 0.0159 | 0.8469 | 1.6975x |
| three_big | 0.0965 | 1.1743 | 0.0168 | 1.1081 | 0.9435x |
| weak_links | 0.0799 | 1.4290 | 0.0144 | 3.2352 | 2.2640x |
| hub | 0.0710 | 0.1323 | 0.0147 | 0.6171 | 4.6658x |
| repeated_cells | 0.0494 | 0.0691 | 0.0143 | 1.5132 | 21.9142x |
| reweight | 0.0898 | 0.5071 | 0.0250 | 1.0339 | 2.0389x |


## 5 million rows; 2 RHS, first solve

| Case | fereg build+marshal (s) | fereg solve (s) | within build (s) | within solve (s) | fereg gain on solve | 
|---|---:|---:|---:|---:|---:|
| one_big | 0.1894 | 0.0972 | 0.0851 | 0.5509 | 5.6667x |
| two_big | 0.3705 | 1.0163 | 0.0790 | 1.5136 | 1.4893x |
| three_big | 0.3987 | 5.1389 | 0.0926 | 2.4357 | 0.4740x |
| weak_links | 0.3199 | 1.0364 | 0.0921 | 12.5924 | 12.1496x |
| repeated_cells | 0.1353 | 0.2763 | 0.0750 | 2.0312 | 7.3511x |


## Reuse, weighted case (1M, 2 RHS)

| Round | fereg set_weights (s) | fereg solve (s) | within Solver rebuild (s) | within solve (s) |
|---|---:|---:|---:|---:|
| 0 | 0.0000 | 0.1546 | 0.0131 | 0.2555 |
| 1 | 0.0319 | 0.1497 | 0.0056 | 0.2384 |
| 2 | 0.0149 | 0.1514 | 0.0050 | 0.2348 |

## Structural interpretation

- One large and two small FE: fereg can solve the reduced system exactly and gains strongly on 2 RHS; scaling to 8 RHS favors fused operations.
- Two big FE: direct fereg-core PCG **wins**, despite fereg complete estimator being slower in the previous workflow for this shape; packaging, sample construction, or estimator costs explain the different apparent winner. This difference needs profiling, not a premature claim about the FE algorithm.
- Three comparably large FE: within is faster at 1M and 5M even with nearly equal iteration counts; suggests a reduced-PCG iteration cost disadvantage for this shape.
- Weak FE connections: fereg benefits substantially, in particular 5M case; within's **first solve** triggers delayed Schwarz preparation, while subsequent solves re-use that state. Report subsequent rounds separately.
- Many repeated FE tuples: fereg is much faster than raw within; EconHDFE native auto already has tuple compression, so compare with that before integrating fereg.
- Eight RHS increases fereg's relative advantage, indicating fused multi-RHS primitives are important.
- For PPML-like weight updates, fereg's stateful `set_weights` avoids fully rebuilding structure, whereas within preserves `Design` but refreshes the numeric Solver. This is a weighted least-squares proxy and **not** a complete PPML estimator benchmark.

## Methodological guardrails and next planner step

- Prefer overall setup + residualization + memory, not only the favorable kernel time.
- Never enable eager global factorization or Schwarz by default merely because there are many FE groups.
- Route based on skewed dimension counts, connected-component weakness, repeated-tuple compression and number of projected columns.
- Distinguish cold and persistent usage, and set a numerical certification gate.
- Further data needed: native-router (plain, auto), and end-to-end PPML independent of the core timing.

[Raw machine-readable per-solve output](fereg_kernel_direct_20261029.json).
