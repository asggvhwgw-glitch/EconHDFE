# ivgmmgrid: Python IV-GMM candidate grids

This GPL-3.0-only companion evaluates many two-step clustered IV-GMM specifications with **one absorbed categorical fixed effect, one clustering dimension, one endogenous candidate per model, and no weights**. NumPy/SciPy perform every runtime calculation. Python also handles per-candidate missing samples, singleton removal, rank checks, omitted collinear controls and reference fits. Stata is an optional external validation oracle only.

## 与 EconHDFE 的关系

EconHDFE 提供更广泛的 HDFE 估计功能。本工具补充同一研究中反复比较候选变量的网格计算，以及原水需求研究的选点与最终重估规则。两者分别安装；本工具不改变主包接口。特别是，本工具保留原 ivreghdfe 两步 GMM 的协方差口径：用第一步残差得到的聚类矩阵计算第二步有效 GMM 方差，不直接复用主包基于最终残差的协方差结果。

## Install and use

From the repository root:

```bash
python -m pip install ./contrib/ivgmmgrid
python contrib/ivgmmgrid/examples/quickstart.py
```

```python
from ivgmmgrid import fit_grid
result = fit_grid(y, exog, candidates, excluded_instruments, fe, cluster)
print(result.coefficients)  # candidate first, then exog, for each model
print(result.covariance)
print(result.models[0].sample_indices)  # original input row positions
```

Inputs are arrays. `exog` may be N×0; `candidates` and excluded instruments must have at least one column. FE/cluster labels can be numeric or strings. Missing numeric values are dropped per model; an optional boolean `mask` applies an additional common restriction. Do not pre-intersect different candidate samples. An intercept is absorbed. Fixed effects are nuisance variables, not recovered output.

`engine="cached"` shares within transforms and cluster cross-products among models with identical rows. `engine="reference"` reconstructs observation-level scores per fit in Python. `verify=True` compares the routes. The external frozen Stata oracle supplies independent implementation checks. Rank-deficient regressors are omitted in candidate-first input order; an absorbed/unidentified endogenous candidate or singular moment covariance raises `IdentificationError`, without accepting a pseudoinverse result.

The covariance finite-sample multiplier is `(N-1)/(N-k-a) * G/(G-1)`, where k counts retained regressors, and a is the FE level count or 1 when the FE is nested in clusters. RMSE uses N-k-a, inference uses G-1. This deliberately bounded contract is not full Stata syntax compatibility, multiway absorption, survey weights or weak-IV-robust inference.

## Real water-demand replay in Python

Install `./contrib/ivgmmgrid[test]`, download [Do and Jacoby's public v2 data](https://doi.org/10.5281/zenodo.10965745), and run:

```bash
python contrib/ivgmmgrid/examples/replay_water.py --data /path/to/Usage_July15_May18.dta
```

The example reconstructs three frozen bootstrap household draws, differences and instruments from the raw public data entirely in Python. It verifies 780 grid points, final coefficients/covariance, chosen lag/gamma and exact final sample keys against frozen Stata outputs. Lag-length and LASSO instrument **preselection are frozen inputs**, not newly implemented selection algorithms. This is not a replication of all 1,000 bootstrap draws in the paper. Progressive float32 candidate storage, mean ties within order, maximum ties between orders and the fourth-lag final sample restriction are preserved.

See [validation results](VALIDATION.md), [mathematical conventions](METHODS.md), and [source and data notices](THIRD_PARTY_NOTICES.md). From this package directory, run `python -m pytest -q` after installing `.[test]`. Stata `.do` files under `validation/` are optional oracle regeneration aids and are excluded from the Python wheel.
