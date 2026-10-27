# Third-party numerical implementation

`src/feselection/monitored_lsmr.py` adapts the installed SciPy 1.17.0 LSMR
implementation (David Fong, Michael Saunders and SciPy contributors), including
`_sym_ortho` from the same release. The upstream BSD-3-Clause license and source
snapshot are in `third_party/`. Only a scalar progress monitor and distinct
cooperative stop code 8 have been added; no new LSMR mathematical method is
claimed. This research contribution is not a claim of clean-room authorship of
that upstream routine. The four earlier hyperFE kernels remain byte-identical.
