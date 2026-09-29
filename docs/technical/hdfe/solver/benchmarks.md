# HDFE solver benchmark evidence

> **Document status:** Current EconHDFE project technical documentation.


Canonical machine-readable evidence lives outside the documentation tree so it can be consumed by tests and release tooling:

- `benchmarks/hdfe/solver_v044_integration.json`: v0.4.3 versus v0.4.4 solver-opt2 integration benchmark.
- `benchmarks/hdfe/exact_rank.json`: exact multiway structural-rank benchmark.

The v0.4.4 solver benchmark records, on the release container, approximately 1.72x warmed speedup for random 200k 3FE, 8.69x for leaf-rich 200k, 2.91x for weighted leaf-rich 1M, and 4.22x for weighted 500k x 20 RHS. An irreducible hard cycle is near parity (~1.05x), and complex mobility end-to-end cells are approximately parity. These numbers are environment-specific evidence, not hardware-independent guarantees.

Release claims should cite the JSON evidence rather than copying timings into code or API documentation.
