# Native within for weighted Poisson-family estimators

PPML and IV-PPML support `PPMLConfig(engine="within")` and
`IVPPMLConfig(engine="within")`. The default `engine="replica"`
and existing `engine="optimized"` remain unchanged. This adds an
engine choice without changing published public dataclass signatures
or the release-0.9.1 contract snapshot.

For the within engine, dense and block-separable Poisson estimators
use the existing `WeightedFEProjector(method="within")`.
The selection applies to post-FE rank screening, IRLS weighted solves,
separation-induced sample rebuilding, and the final high-precision
FWL/inference projection. Separation-specific simplex and ReLU
operators remain unchanged.

Native within requires strictly positive finite numerical weights.
The FE topology is reusable, but numerical weights are updated on each
IRLS iteration. For zero-weight states the existing generic LSMR
backend is selected, returning to native when weights are positive.
Absent the optional `_econhdfe_within` extension, an explicitly
requested within solve raises a configuration error.

Numerical tolerances, rank policy, moment conditions, separation,
sample semantics and VCE remain unchanged. Inspect
`result.diagnostics["projection_resources"]` to see the method
and effective backend. Speedup requires end-to-end validation.
