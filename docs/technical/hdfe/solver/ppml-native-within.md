# Native within for weighted Poisson-family estimators

PPML and IV-PPML now accept `PPMLConfig(absorb_method="within")`
and `IVPPMLConfig(absorb_method="within")`. The default remains
`absorb_method="map"`, and `"lsmr"` is available without the
optional native extension.

Both dense and certified block-separable computation paths pass the
selection to their weighted FE projectors during post-FE rank screening,
IRLS, separation-induced sample rebuilding, and the final high-precision
weighted FWL solve. Separation-specific simplex/ReLU solvers are unchanged.

Within requires positive finite numerical weights. A weighted projector
retains FE topology across IRLS iterations, but calls
`update_weights` on each iteration; weights are *not* held fixed.
When any weight is zero, the explicit within projector selects its
existing generic LSMR fallback for that weight state. On a subsequent
strictly positive state it can return to within.

Native within remains strictly opt-in; if requested without the
`_econhdfe_within` extension, the model raises an actionable
configuration error rather than reporting results from another backend.

Inspect `result.diagnostics["projection_resources"]` to distinguish
the requested method and the backend actually used on the final solve.
Changing the absorber does not modify separation, coefficient rank,
the PPML/IV-PPML moments, the outer IRLS convergence criteria or the
inference formulas. End-to-end speedup is workload-dependent and must
be established with matched accuracy and full-estimator benchmarks.
