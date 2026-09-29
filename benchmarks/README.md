# Benchmark layout

Benchmark evidence is grouped when a stable domain boundary exists:

- `hdfe/`: HDFE numerical-solver and exact-rank evidence.
- `ppml/`: PPML benchmark programs and results.
- `repeated/`: repeated-specification/session benchmarks.
- `real_world/`: externally executed, user-supplied real-data benchmark records; preserve original runs immutably and append follow-ups.

Some inherited linear/HDFE benchmark scripts remain at this directory level to avoid path churn in historical tooling. New benchmark artifacts should be placed in the appropriate domain subdirectory rather than adding more root-level files.

## Real-machine benchmark return format

Use `real_world/BENCHMARK_REPORT_TEMPLATE.md` for new user-authorized real-data benchmarks. Preserve prior dated records; never overwrite historical benchmark evidence. A benchmark report must separate parity from speed, state timing scope/cold-vs-warm behavior, record machine/package versions and resource settings, and avoid embedding raw/private data.

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

### 2026-09-29 诊断核心收口：SW/KP 与第一阶段/AP 复用

本轮基线是性能分支的 `8cf3afc938cee1f56564e2d0387308d9848ac42f`，
不是未优化的初始版本。以下三列在同一 Linux 宿主、Python 3.13.5、
NumPy 2.3.5、SciPy 1.17.0、Numba 0.65.1、BLAS=1、Numba=4 下重新测量。
每个版本/场景独立进程，先预热，再执行七次完整拟合取中位数；另跑一次
profile，不把 profile 时间混入计时。最终实现加入资源保护后单独复测，
宿主并非硬件隔离环境，数字不代表跨机器保证。

| 合成 2SLS 规格 | 远端基线（秒） | 先前 SW/KP 候选（秒） | 最终实现（秒） | 最终相对远端 | 最终相对候选 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 96k、12 个内生斜率、structured、robust | 0.839472 | 0.316925 | 0.297895 | 2.82x | 1.06x |
| 相同规格、dense、robust | 1.038262 | 0.411136 | 0.385714 | 2.69x | 1.07x |
| 96k、12 个内生斜率、structured、cluster | 0.915043 | 0.388620 | 0.330133 | 2.77x | 1.18x |
| 300k、一个内生变量、dense、robust | 0.179443 | 0.169999 | 0.168716 | 1.06x | 1.01x |
| 96k、24 个内生斜率、structured、robust | 3.888414 | 0.914663 | 0.761515 | 5.11x | 1.20x |

所有原有诊断仍然计算，`diagnostics="off"` 的历史语义未改变。不能将本表与
上一轮在不同计时条件下的秒数相乘。最终 dense/structured 比约为 1.29，
不是表中的 2.82x。单内生变量的本轮增益不足 1%，不作为确定的性能收益。
进程峰值 RSS：12 维 structured/robust 从 580.18 降至 447.04 MiB，
24 维从 1281.64 降至 596.32 MiB；单内生变量从 372.79 增至 378.73 MiB。
RSS 包含预热及独立 profile，不是函数内部峰值，也不是普遍省内存声明。

整合的 SW/AP 算法把重复的 N 行条件回归压缩到工具变量维度的小系统；KP
先投影得分再调用原有协方差引擎。新增复用仅保存相同 reduced-form 残差的
协方差伪逆，让普通第一阶段与 AP 共用，SW 仍单独计算。VCE、聚类、时间、
权重得分缩放、自由度或其他推断设置变化会失效；数组按内容检查，object
标签禁用缓存。缓存矩阵负载最多 8 MiB，SW 条件残差每批最多四列，长样本
自动缩至两列或一列。目标为每张响应数组约 16 MiB，单列已超过时除外；
这些不是全回归的硬 RSS 限额。不缓存全部得分或方程两两协方差张量。

12 维常规案例的诊断协方差调用由候选的 36 次降至 24 次。底层 robust
协方差原本已按行分块；实验中的新批量 Gram 内核没有稳定优势，已撤掉，
`compute/vcov.py` 不变。保留病态/近完全拟合回退，且先决定回退、再计算
优化协方差，避免保护触发前先失败。既有约 1e3 的条件数保护仅为执行选择，
不是新的秩阈值或通用误差保证。

最终验证：当前完整源码测试 988 passed（含 105 项诊断定向测试），源码外
安装 wheel 测试 408 passed。wheel 使用宿主依赖和非隔离构建，不等于干净
依赖解析或跨平台验收。五个场景的十组新旧对照覆盖 2738 个数值字段；系数、
VCE、残差、第一阶段输出及自由度逐位一致，诊断最大绝对差 2.97e-9，满足
rtol=2e-9、atol=1e-9。额外常规 sweep 128 组通过，边界 26 组数值通过、
2 组新旧一致报错（不当作成功估计）。独立 Fraction/Gauss-Jordan 参考在
三种近共线程度下检查 12 个 AP/SW F，最大相对误差 2.14e-10；严格参考
测试使用 rtol=2e-9、atol=1e-16。这不是通用数值证明或外部 Stata 认证。

版本门禁、架构图检查和公共契约对照通过，公共接口、默认行为、版本均不变。
最终诊断源文件 SHA256：
`2ff63d7f1ae31ba0825fbeb580a2d3cceb01955cd9021008330ac41f865bd906`。
本轮只改一个运行时模块及既有测试/benchmark 文件；尚未执行新提交的
16 格跨平台/最低依赖 CI，没有合并、tag、Release 或 PyPI 发布。
