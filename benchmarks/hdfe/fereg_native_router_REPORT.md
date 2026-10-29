# EconHDFE NativeSolver auto/plain versus base within: native overhead and compression

GitHub Actions run: https://github.com/asggvhwgw-glitch/EconHDFE/actions/runs/40809577543 (native job 123236706221).
Source is `benchmarks/hdfe/native_router_bench.py` on the independent branch. This is a **within-family wrapper comparison**, not a direct fereg comparison.

- Result: **99 of 99 combinations passed**, 99/99 processes exited successfully; maximum independent normalized FE orthogonality RMS **5.9044e-10**.
- Cases: 7; n=120k/1M for 2 and 8 RHS; n=5M for 5 cases with 2 RHS.
- All inputs shared exactly, two threads, prescribed tol 1e-10.
- `within`: upstream Python `within_py==0.4.0` persistent `Design/Solver`, batched RHS.
- `native_plain`: EconHDFE Rust wrapper pinned within revision, single RHS loop with certification and safeguards.
- `native_auto`: same wrapper, automatically SmallSchur or tuple compression; tuple threshold U/N<=0.7.
- `source_setup_s`: preprocessor and solver construction, `setup_s`: weight update (for step>0), `solve_s`: actual projection, including safeguards. Timing excludes DGP generation.
- Do **not** attribute all native wrapper overhead to Python: extra weighted stationarity certification, stricter iteration criteria and fallback/polishing also matter. The upstream version packaged in PyPI may have minor changes versus the native pinned commit.

## 1 million, 2 RHS

| FE graph | upstream within batch s | EconHDFE plain s | EconHDFE auto s | auto route | auto cells/N |
|---|---:|---:|---:|---|---:|
| one_big | 0.172 | 0.260 | 0.195 | schur | 1.000 |
| two_big | 0.293 | 0.415 | 0.416 | within | 1.000 |
| three_big | 0.392 | 0.532 | 0.563 | within | 1.000 |
| weak_links | 1.934 | 2.492 | 2.487 | within | 0.997 |
| hub | 0.239 | 0.326 | 0.324 | within | 1.000 |
| repeated_cells | 0.388 | 0.525 | 0.162 | compressed_within | 0.125 |
| reweight | 0.325 | 0.486 | 0.468 | within | 1.000 |


## 1 million, 8 RHS

| FE graph | upstream within batch s | EconHDFE plain s | EconHDFE auto s | auto route | auto cells/N |
|---|---:|---:|---:|---|---:|
| one_big | 0.692 | 1.096 | 0.747 | schur | 1.000 |
| two_big | 1.136 | 1.653 | 1.667 | within | 1.000 |
| three_big | 1.500 | 2.100 | 2.115 | within | 1.000 |
| weak_links | 3.896 | 6.305 | 6.272 | within | 0.997 |
| hub | 0.884 | 1.275 | 1.254 | within | 1.000 |
| repeated_cells | 1.517 | 2.119 | 0.625 | compressed_within | 0.125 |
| reweight | 1.292 | 1.816 | 1.841 | within | 1.000 |


## 5 million, 2 RHS

| FE graph | upstream within batch s | EconHDFE plain s | EconHDFE auto s | auto route | auto cells/N |
|---|---:|---:|---:|---|---:|
| one_big | 1.069 | 1.735 | 1.741 | within | 0.998 |
| two_big | 2.959 | 4.420 | 4.438 | within | 1.000 |
| three_big | 4.582 | 6.801 | 6.891 | within | 1.000 |
| weak_links | 16.088 | 25.799 | 25.447 | within | 0.999 |
| repeated_cells | 4.309 | 6.000 | 1.262 | compressed_within | 0.125 |


## Weighted iteration (1 million, 2 RHS)

| Method | First setup s | First solve s | Update 1 setup s | Update 1 solve s | Update 2 setup s | Update 2 solve s |
|---|---:|---:|---:|---:|---:|---:|
| within | 0.030 | 0.325 | 0.009 | 0.298 | 0.009 | 0.299 |
| native_plain | 0.079 | 0.486 | 0.046 | 0.444 | 0.041 | 0.457 |
| native_auto | 0.148 | 0.468 | 0.046 | 0.438 | 0.041 | 0.441 |

## Findings

1. Base within `solve_batch` is consistently faster than native `plain` across the tested topologies. At 1M/8 RHS the repeated-column loss is approximately 35–60% on tested graphs.
2. Native `auto` has a strong benefit only on compressible FE tuples. For repeated cells, 1M/2 RHS: auto 0.162s vs raw 0.388s; 5M/2 RHS: auto 1.262s vs raw 4.309s. There is also extra one-time setup cost.
3. On high-uniqueness graphs, `auto` often builds an index but ultimately routes back to uncompressed within: overhead without a benefit, particularly at 5M (two/three big, weak links).
4. Native SmallSchur on one_big at 1M solves with 1 elimination step but is slower than the base `within` batch path after including its construction and row processing; direct fereg-core Schur in the separate test is faster, indicating implementation rather than theory is the main opportunity here.
5. Native weighted refresh spends ~0.041–0.046s on numeric state rebuild per 1M rows, compared with ~0.0085s for base within reusing the design. This can materially impact PPML outer iterations.
6. Raw base within has weaker stationarity RMS than native (e.g. 1e-10 versus 1e-12) even though both are far below the independent 5e-8 gate. Strictly matching achieved accuracy rather than requested tolerance is necessary for final rollout.
7. Because performance under `auto` depends on compression, a planner needs a cheap sample-based lower bound on distinct tuples rather than always building a complete tuple index.

## Recommendations

- First implement or expose batch RHS on the EconHDFE native wrapper, preserving independent numerical certificates.
- Retain the compressed-within branch as a cost-effective fallback; avoid building its index when the topology cannot plausibly compress.
- Tighten SmallSchur admission to be based on predicted total time (construction + RHS count + reuse), not only q<=128 and conditioning.
- Benchmark a fereg-core-derived reduced PCG backend with matching wrapper semantics and multiple RHS before any user-facing backend switch.

[Raw structured results](fereg_native_router_20261029.json).
