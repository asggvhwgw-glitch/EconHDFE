# FastSandwich: optional covariance companion

FastSandwich computes Bartlett/Newey-West HAC and one-way clustered score covariance using NumPy/SciPy. It includes the complete runtime, independent numerical checks, a statsmodels OLS/WLS interface, and an opt-in linearmodels 7.0 launcher. Original code is BSD-3-Clause; see [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).

## 与 EconHDFE 的关系

EconHDFE 负责拟合 OLS、IV、PPML 及高维固定效应模型；本目录提供可单独使用的协方差计算工具。它适合作为长带宽 Bartlett HAC 的补充及比较基准。合并本目录后不会自动改变 EconHDFE 的估计结果或默认计算路径。EconHDFE 的多向聚类、缺口时间、面板及自由度规则仍由主包负责。

本项目不是对所有数据都更快的承诺：短带宽可能没有收益；EconHDFE 已编码的聚类内核在一些场景更快。这里不将单个内核的速度解释成完整回归的提速。

## Install and run

From the **EconHDFE repository root**, preferably in a virtual environment:

```bash
python -m pip install ./contrib/fastsandwich
python -m pip install "./contrib/fastsandwich[test]"
python contrib/fastsandwich/examples/hac_policy.py
```

```python
from fastsandwich import cov_hac, hac_meat, cluster_meat
cov = cov_hac(fitted_ols_or_wls, nlags=12, use_correction=True)
S_hac = hac_meat(scores, nlags=12)  # unnormalized sum
S_cluster = cluster_meat(scores, groups)  # divided by number of rows
```

Scores are NOT centered automatically. HAC rows must already be consecutive, equally spaced observations in time order. Missing periods and panel boundaries require explicit handling before calling this interface. Use `bread @ S_hac @ bread.T` for an uncorrected sandwich; apply only the finite-sample correction appropriate to the fitted model.

**Bandwidth conversion:** FastSandwich `nlags=L` gives weights `1-lag/(L+1)`. EconHDFE's Bartlett kernel calls that `bandwidth=L+1`. Equal numeric arguments do not mean equal kernels. Cluster normalization also differs: multiply `cluster_meat` by N to obtain an unnormalized score meat. The checked [example](examples/econhdfe_scores.py) demonstrates these conversions on finite, regularly spaced scores:

```bash
python -m pip install .
python contrib/fastsandwich/examples/econhdfe_scores.py
```

The existing linearmodels path remains available from this directory:

```bash
cd contrib/fastsandwich
python tools/run_optimized.py examples/panel_policy.py
python -m pytest -q
```

The launcher checks the installed linearmodels 7.0 source hash, builds a private copy under `.cache`, and runs a fresh process. It does not edit site-packages. Other versions fail the guard. Tests use `checks/check_*.py` and local pytest configuration so the parent EconHDFE test command does not require companion dependencies. Optional EconHDFE comparison checks skip when the parent package is not installed.

## Numerical scope

The default HAC path uses local moving-window Gram products for longer lags and direct products for short cases. `method="prefix"` is a historical comparator with sensitivity to extreme accumulated offsets; `method="fft"` is an established comparator. Float64 equivalence is not bitwise identity or protection from overflow. `cov_hac` supports OLS/WLS and explicit `(scores, bread)` inputs, not every statsmodels family. Research identification and covariance choices remain the user's responsibility.

`SOURCE_ORIGIN.json` records the imported source hashes. This runnable contribution omits the original study's paper, timing archives, environments and raw data. No runtime depends on those omitted files. [Validation](VALIDATION.md) records checks on this contribution.
