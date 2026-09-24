# BoundedQR: optional quantile-regression companion

BoundedQR provides an unweighted linear quantile-regression process with clustered wild-gradient bootstrap draws, bounded intermediate memory, CPU execution, optional GPU routes and explicit numerical error diagnostics. The complete Python runtime and R bridge are included. Licensed **GPL-3.0-or-later**; see [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).

## 与 EconHDFE 的关系

EconHDFE 的 OLS/IV/PPML 主包覆盖不同的模型。本目录补充研究结果分布不同位置（如中位数）的分位数回归及聚类自助推断。它是单独安装、单独调用的配套项目，不是主包的 HDFE 后端。合并后，用户可从本目录运行 Python 或 R 示例；主包接口和 BSD 许可证保持原有适用范围。

**不能用 OLS 去均值的方法代替分位数回归中的固定效应处理。** 本版本需要有限、满列秩的显式设计矩阵；不支持自动吸收高维固定效应、调查权重、缺失值处理或自动删除共线列。可以显式构造满秩虚拟变量，但其内存成本仍存在。

## Install and run (CPU)

From the EconHDFE repository root, in a virtual environment:

```bash
python -m pip install ./contrib/boundedqr
python contrib/boundedqr/examples/clustered_quantiles.py
```

```python
from boundedqr import bootstrap
result = bootstrap(X, y, cluster, quantiles=(0.25, 0.5, 0.75),
                   reps=199, seed=123, backend="cpu", batch_size=16, threads=1)
print(result.coefficients)
print(result.standard_errors)
print(result.coefficient_radii)  # keep Inf values and inspect diagnostics
```

`X` must include an intercept if required. A full n-by-reps multiplier expansion is avoided; the dense input, requested coefficient draws and certain solver workspaces still occupy memory. This is not a streaming or constant-total-memory claim.

The R bridge preserves `quantreg` baseline residual signs and uses only the `quantreg` and `jsonlite` R packages plus this Python package:

```bash
cd contrib/boundedqr
Rscript -e 'install.packages(c("quantreg", "jsonlite"), repos="https://cloud.r-project.org")'
Rscript examples/clustered_quantiles.R . /path/to/venv/python
```

Replace the last argument with the Python interpreter where BoundedQR is installed. The R example generates data and checks the fixed-multiplier draws against `quantreg::boot.rq`, with every draw retained. The R function lives in `interfaces/R/boundedqr.R`; pass `project_root` when calling it from another directory.

## Validation and limits

```bash
python -m pip install ".[test]"
python -m pytest -q
```

Run these commands inside `contrib/boundedqr`. Checks use `checks/check_*.py` and a local pytest configuration, leaving parent test discovery independent. Included numerical checks cover independent LP references, fixed R draws, block/replay invariance, unit scaling and invalid input. GPU checks skip unless explicitly enabled with an appropriate runtime; see [VALIDATION.md](VALIDATION.md).

The reported radius is an FP64-model-qualified numerical enclosure conditional on supplied data and constructed scores, not a formal interval proof or a statistical coverage guarantee. Infinite radii are retained and mean that this run supplies no finite coefficient-error enclosure. Baseline-score error is outside that enclosure. The Python random stream differs from R's: exact R replay requires supplied baseline coefficients, residual scores and multipliers in the documented cluster order. The R bridge handles R's first-occurrence order internally.

Optional `.[gpu]` targets a separate CUDA 13 environment. EconHDFE's optional CUDA 12 stack should use a different environment; this contribution makes no combined-GPU compatibility claim. CPU installation needs no CUDA, R, Stata or proprietary solver. GPU implementation is shipped, but GPU performance is not re-certified by this PR.

The synthetic R-reference fixture under `experiments/data/n2000_p6_b5` contains no empirical microdata. Original source hashes are recorded in `SOURCE_ORIGIN.json`. Research manuscripts, benchmark archives and development-only experiments are outside this runnable contribution.
