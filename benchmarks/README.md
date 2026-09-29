# Benchmark layout

**Status:** current benchmark index  
**Policy:** benchmark evidence is workload-specific development evidence unless an external comparison records package versions, hardware, workload, parity checks and timing scope.

## Current domain directories

- `hdfe/` — HDFE numerical-solver and exact-rank evidence.
- `ppml/` — PPML benchmark programs and results.
- `repeated/` — repeated-specification/session benchmarks.
- `planner/` — execution-planner and calibration evidence.
- `effects/` — fixed-effect recovery benchmarks.
- `real_world/` — externally executed, user-authorized real-data benchmark records; preserve original runs immutably and append follow-ups.
- `release/` — benchmark material tied to release validation.

## Root-level historical scripts

Several inherited linear/HDFE benchmark scripts remain directly under `benchmarks/`. They are retained at their existing paths to avoid breaking provenance links in historical reports and release evidence.

**Do not add new benchmark families at the root.** New work should use the closest domain directory. A separate provenance migration should precede any future relocation of inherited root-level benchmark files.

## Interpretation rules

A benchmark must separate:

1. statistical/numerical parity;
2. timing scope (end-to-end, setup, projection, solve, inference, etc.);
3. cold versus warm execution;
4. environment and thread/resource settings;
5. workload-specific speed or memory results.

A single benchmark is not a package-wide performance guarantee.

## Real-machine return format

Use `real_world/BENCHMARK_REPORT_TEMPLATE.md` for user-authorized real-data benchmarks. Preserve prior dated records; never overwrite historical evidence. Reports should avoid raw/private data and keep parity evidence separate from speed claims.

## Linear-IV diagnostic decomposition

`bench_iv.py` runs a synthetic IV workload with all existing diagnostics computed,
even when their display mode is `off`. It separates unprofiled warm timings from
a separate stage/call-count profile. For example:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=4 \
PYTHONPATH=. python benchmarks/bench_iv.py --n 96000 --groups 12 --reps 5 \
  --representation structured --vce robust --output /tmp/iv-structured.json
```

Run the same script in separate processes against both source revisions and
repeat with `--representation dense`; do not compare only an old dense run with
a new structured run. `--vce cluster` and `--estimator liml|gmm2s` select other
workloads. `--save-numerics` writes a companion synthetic-results NPZ for
coefficient, VCE, fitted-value, DoF and diagnostic comparisons. Timed dense fits
include role-design construction, as in the heterogeneous-spec benchmark.
Cumulative stage times overlap and must not be added together. Peak RSS is
process-wide (unavailable on Windows), not workspace allocation. These timings
are development evidence, not a universal speed claim or external Stata parity.

### 2026-09-29 workspace check

Baseline: `22e176b771cb9554ec26cc60338a92776c42031d`. On one Linux host,
Python 3.13.5 / NumPy 2.3.5 / SciPy 1.17.0 / Numba 0.65.1, BLAS=1 and
Numba=4, five unprofiled warm repetitions in separate before/after processes gave:

| Synthetic 2SLS workload | Before (s) | After (s) | Before/after |
| --- | ---: | ---: | ---: |
| 96k rows, 12 endogenous slopes, structured, robust | 1.738855 | 0.835278 | 2.08 |
| Same roles, dense, robust | 1.760017 | 0.938512 | 1.88 |
| 96k rows, 12 endogenous slopes, structured, clustered | 1.790198 | 0.890690 | 2.01 |
| 300k rows, one endogenous regressor, dense, robust | 0.224270 | 0.178506 | 1.26 |

All existing diagnostics were computed. Structured robust GELSY calls fell
from 125 to 22. The optimized dense/structured ratio is only 1.12, not 2.08:
the latter is the same structured path before versus after this patch.
Process peak RSS was 570.04 to 571.43 MiB for structured robust and 549.46 to
559.89 MiB for clustered; this is not a memory-reduction claim.

The four before/after pairs had identical coefficients, VCE, first-stage
coefficients/fits and DoF. Diagnostic differences were within rtol=2e-9,
atol=1e-9 (largest absolute difference 3.03e-9 on large F statistics).
The diagnostic test module passed 60 tests; a 132-specification old/new sweep
passed 528 family comparisons; 137 selected historical regression tests passed.
These local checks are not the current full repository suite, cross-platform
CI, or licensed-Stata certification. Public-contract capture for the root and
effects namespaces had zero differences. No version or default changes.

## 2026-09-29：线性 IV 诊断核心收口（未发布）

以 `8cf3afc938cee1f56564e2d0387308d9848ac42f` 为远端基线，本次增量包含已审查的
SW 小系统、KP 投影得分，以及第一阶段/AP 的协方差逆矩阵复用。普通第一阶段与
AP 使用同一 residual/VCE，但检验系数、分子自由度和统计量仍分别计算；SW 使用
自己的残差和协方差。缓存仅存活于一次联合调用，数值负载上限为 4 MiB，超限重算。
所有诊断照常计算；不改变 `diagnostics="off"` 的既有语义。

下表只衡量本轮新增复用相对上一轮 SW/KP 候选的增量。每个版本/场景两个独立进程，
第二轮反转版本顺序；每进程预热后五次，取十次的中位数。Python 3.13.5、
NumPy 2.3.5、SciPy 1.17.0、Numba 0.65.1，BLAS=1、Numba=4。

| 合成 2SLS 场景 | 上轮候选（秒） | 本轮（秒） | 上轮/本轮 |
| --- | ---: | ---: | ---: |
| 96k、12 个内生斜率、structured + robust | 0.428380 | 0.377330 | 1.14 |
| 同一角色设计、dense + robust | 0.609557 | 0.478821 | 1.27 |
| 96k、12 个内生斜率、structured + cluster | 0.598577 | 0.496950 | 1.20 |
| 300k、单内生变量、dense + robust | 0.218830 | 0.224738 | 0.97 |

单内生变量的两轮方向相反，不作提速声明；这些不是外部软件比较或通用倍率。
多内生变量案例的 `ols_vcov` 调用为 36→24，单内生变量为 2→1。
四对比批量化和额外 Shea 交叉乘积重排的收益不足，本轮均不保留；原有 robust/
cluster 流式协方差、HAC/DK 顺序与多向聚类 PSD 修正位置保持不变。

完整源码测试：984 passed，无失败或跳过；定向诊断：101 passed。
128 组常规诊断对照通过；边界为 26 组数值一致、2 组共同异常（不是成功估计）。
16 组独立进程对照的主估计输出逐位一致，诊断最大绝对差约 2.91e-9，满足
rtol=2e-9、atol=1e-9。新增两个精确 Fraction 参考案例不依赖包内求解/VCE。
版本、公共契约、架构门禁通过；未执行本增量的跨平台 CI、隔离安装或 Stata 外部验收。
完整脚本、原始计时、哈希、JUnit、来源与补丁随 `econhdfe-iv-core-optimization.zip` 交付。
