# Python contribution validation — 2026-10-02

Windows, Python 3.12, NumPy 2.5.3, SciPy 1.18.1; CPU execution. 82 checks passed, 15 optional GPU checks skipped. EconHDFE 0.7.0 full suite: 1,068 passed in a fresh environment with all companions installed. The vendored PyFixest MIT kernel retains its original SHA256.

## Real ACS results, including the difficult primary design

Both designs retain 133,122 California ACS 2024 public-use records and 281 PUMA clusters, at the median. Inputs and original quantreg 6.1 FN reference draws were frozen; no observations or failed enclosures were removed. One-thread CPU solver times exclude loading, baseline fitting and score construction. These are one-run timings, not a new R speed comparison.

| Design | Draws | Python seconds | Finite / infinite radii | Max coefficient difference from frozen R | Max SE relative difference |
|---|---:|---:|---:|---:|---:|
| Primary, 112 columns, including county indicators | 19 | 8.65 | 0 / 19 | 0.00903 | 15.20% |
| Post-hoc pooled sensitivity, 72 columns | 199 | 57.40 | 163 / 36 | 0.00423 | 0.0413% |

**The primary design is not certified for interchangeable empirical inference with the R reference.** All draws are finite but every radius is infinite, and some standard errors differ materially. The pooled design is a different conditional association specification: removing 40 county indicators is not a repair of the primary model. It also has 36 infinite radii. A small reference difference or a finite radius for other draws does not establish statistical coverage. Inspect diagnostics before using estimates in a paper.

The pooled reference uses the corrected C-order fixture `acs_pooled_r_shared_c_b199`; the older similarly named fixture without `_c_` had an R array-layout problem and is not used. The fixed-W references isolate solver behavior; Python-generated bootstrap streams need not equal R streams.

## Reproduce without R

From this package directory, install `.[test]`, then:

```bash
python validation/datasets/prepare_acs.py
python validation/prepare_replay.py
python validation/replay_real_data.py --fixture .cache/acs_primary --reference validation/frozen/primary/reference.npy --output .cache/primary_result
python validation/replay_real_data.py --fixture .cache/acs_pooled --reference validation/frozen/pooled/reference.npy --output .cache/pooled_result
```

Preparation downloads three official Census sources (about 72 MB in total), verifies recorded hashes and reconstructs the original matrix. Generated raw data and arrays are ignored by Git. The small frozen W, baseline coefficient and reference coefficient arrays are included; no ACS person records are committed. `prepare_replay.py` verifies matrix hashes before pairing them with the frozen arrays. JSON result files record inputs, retained draws and errors.

This is an unweighted, model-based PUMA clustering illustration, not ACS survey-design variance or a causal wage estimate. The Python runtime and ordinary examples need neither R nor Stata. Optional R interface instructions are [separate](../../compat/boundedqr-r/README.md). GPU source remains available, but this PR makes no renewed GPU performance claim.

The relocated R example was also run with R 4.6.1 and quantreg 6.1: all nine fixed draws had finite radii, with maximum coefficient difference 5.56e-16. Both ACS matrices were freshly rebuilt from the three pinned Census source files and matched the frozen matrix SHA256 values exactly.
