# FE 选择器研究版 0.5

这是在交付的 **0.4目录**上推进的显式实验路径，不是 EconHDFE 的发行版本，
也不修改主包默认吸收器、样本、singleton、自由度、权重或统计推断口径。

## 本轮变化

1. 样本 raw reject 会记入日志，再降级为 effective unknown。它不能独自否决完整
   数据候选，也不能取消临界区热确认。完整数据的资源/数值检查仍然保留。
2. 有用列的局部试探沿同一条 LSMR 递推继续。每8步观察标量收敛和耗时；24步
   不再是强制退出点。保留标准容差，成本退出使用独立状态8并让原生 within
   完成可行的部分残差，不冒充数值收敛。
3. 几何外推不当成保证。在持续进展且一小段续跑成本可承受时，允许多观察一个
   窗口；同时保留相对原生耗时的软上限、绝对秒数上限和原有maxiter。

此处没有新的预条件器数学公式。四个既有hyperFE数学内核保持不变。监测版
LSMR改编自SciPy 1.17.0，作者、原始源码、校验和及BSD-3-Clause许可保存在
`third_party/`，另见 `THIRD_PARTY_NOTICES.md`。不能将该部分宣称为独立原创或
clean-room实现；没有改写原生within的Rust内核。

## 安装与验证

本轮测试环境为Python 3.13.5。精确环境见ENVIRONMENT.json，复制本次依赖版本：

```bash
python -m pip install -r requirements-reproduce.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  python -m pytest tests baseline/tests -q
```

离线时使用已提供的within-py 0.4.0 Linux wheel（其SHA256见ENVIRONMENT.json）。
该wheel和其它依赖需要已经可用；复现包没有自带全部平台的依赖安装器。
JIT首次编译不包含在性能表中，但实际首次使用会有该成本。

单个百万级例子（输出到新目录，不覆盖历史记录）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 RAYON_NUM_THREADS=1 \
  python benchmark_v05.py --case smooth_sparse --seed 93517 --rhs 3 --mode v05 \
  --out results/reproduce/smooth_v05.json
```

原生基线将 `--mode v05` 改成 `--mode native`；冻结0.4改成 `--mode v04`。
`--fixed-pilot`、`--no-early-scout`、`--pilot-native-fraction` 仅用于明确的消融，
不能事后选择最快配置作为在线选择器的成绩。
完整逐进程命令保存在results/formal_jobs.json与supplemental_jobs.json。
run_jobs.py 会将旧 `/mnt/data/fe_selector_v05/` 参考路径重定位到当前目录。
运行整组命令前请备份历史results目录。大输入和参考npy由种子重建，不随包分发。

## 程序接口

```python
import sys
sys.path.insert(0, "src")
from feselection import FESelector, SelectionConfig

# codes: (N,Q), observed dense nonnegative integer IDs; positive finite weights.
# Y: (N,K) float64. This computes FE projection, not a complete OLS/IV estimator.
plan = FESelector(SelectionConfig(mode="auto")).prepare(
    codes, weights, expected_rhs=Y.shape[1]
)
result = plan.residualize(Y)
residuals = result.residuals
trace = result.diagnostics
```

默认mode仍为observe，不自动推广到主包。输入快照、权重重建和缓存语义继承0.4。
改变权重必须调用 `with_weights` 或重新prepare，不复用过时数值分解。

## 如何读证据

- report_zh.html / results/SUMMARY.json：本轮汇总，含慢例和拒绝轨迹。
- results/formal：新种子六类百万级设计，三方法、三次重复。
- results/reference与supplemental：全列参考、旧反例回归、强制成本退出、第二种子。
- development_source_A与results/development：第一版成本外推仍过早退出的失败证据。
- results/SOURCE_FREEZE.json / POLICY_FREEZE.json：正式前冻结的计算源码与参数。
- frozen_v04：未修改的0.4源码与旧测试，便于核对；不进入当前测试收集。

已知残差和正规方程检查不等于每个普通列都有独立参考。只有原始JSON含
`reference_relative_errors` 才做了全部列参考核验。正常退出不等于路由最快。
旧83131/83132案例用于回归与开发，不称为独立holdout；93517/93518仍是已有
合成家族，不是无关生成机制或真实数据上的普遍优势证明。

## 限制

软时间上限不能中断一次底层运算；工作区预算不是全进程RSS硬限制。
对首列热耗时和后续RHS难度的外推仍可能错。仅有用列试探使用成本控制，
后续常规局部解仍遵循原maxiter及数值审计。抽样诊断本身仍有开销。
一般高阶精确零空间、全套near-absorbed X、PPML刷新、slopes、GPU及完整主库
集成/CI均未在本轮认证。没有远端写入，没有变更默认后端或发布包。

增量补丁相对交付的0.4目录，不直接应用到main或跳过前序版本套到PR #27。
