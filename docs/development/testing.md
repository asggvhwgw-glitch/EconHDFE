# 测试指南

当前源码为 0.8.3。测试按公开契约、统计行为、数值参照和工具职责组织，
不按修复日期或历史版本堆积。当前采用约200项的精简回归集；精确提交的执行结果见 [验证状态](test-status.md)；
历史迁移见 [0.6.5 记录](history/test-migration-0.6.5.md)。

## 入口与职责

| 层 | 用例数 | 检查内容 |
| --- | ---: | --- |
| `tests/contracts/` | 24 | 公共 API/兼容入口、配置、错误、因子语法、结果协议和隐私 |
| `tests/behavior/` | 92 | 四类估计器的样本、权重、FE、DoF/VCE、遗漏、预测、session 与缓存语义 |
| `tests/numerics/` | 72 | 独立 dummy/WLS/GLM、SVD/Fraction/moment oracle，秩、尺度、收敛及资源边界 |
| `tests/tooling/` | 17 | 公共契约快照、架构/注册表、发布门禁、制品哈希、安装及来源检查 |

```bash
# 日常公共行为；plain pytest 也只运行这一部分
python scripts/run_tests.py --suite core -- -q

# 数值核或发布工具的定向检查
python scripts/run_tests.py --suite numerics -- -q
python scripts/run_tests.py --suite tooling -- -q

# PR 与发布前：完整四层测试
python scripts/run_tests.py --suite full -- -q
python scripts/version.py gate
python scripts/generate_architecture_map.py --check

# 单项复现直接使用 pytest 节点
python scripts/run_tests.py -- tests/behavior/test_sample_semantics.py -q
```

`core` 为 contracts + behavior；`full` 为整个 `tests/`。
CI 的 Linux/macOS/Windows × Python 3.10–3.13，以及四个最低依赖组合全部运行 `full`；
源码制品重测也运行 `full`。禁止将 core 的通过数登记为完整验收。
Numba 最大线程池须至少为 3，通常设 `NUMBA_NUM_THREADS=4`，
BLAS/OMP/MKL 各设 1。详见[测试环境](test-environment.md)。

## 保留哪些测试

每个独立风险由最接近职责的测试负责。新增修复优先扩展已有案例，
避免重复执行相同准备、公式和随机 DGP。重点位置如下：

| 风险 | 主要测试 |
| --- | --- |
| 四模型入口与论文表协议 | `contracts/test_public_workflows.py` |
| 频数展开、权重语义与拟合统计量 | `behavior/test_weights_and_effects.py`、`test_fit_statistics.py`、`test_fit_degrees_of_freedom.py` |
| 识别、固定 k-class、IV 首阶段和分阶段策略 | `numerics/test_iv_correctness_closeout.py`、`test_iv_primitives.py`、`behavior/test_iv_stages.py` |
| 非线性矩条件、GLM 与 sandwich | `numerics/test_nonlinear_oracles.py`、`test_estimator_oracles.py` |
| 精确秩与结构/数值 FE 拓扑 | `numerics/test_exact_oracles.py`、`test_exact_multiway.py`、`test_fe_canonicalization.py` |
| 投影尺度、预算与线程 | `numerics/test_cg_scale_contract.py`、`test_native_rank_budget.py`、`behavior/test_cg_resource_execution.py` |
| 后估计线性组合/Wald、样本、预测、FE 识别与缓存失效 | `behavior/test_postestimation.py`、`test_sample_semantics.py`、`test_prediction.py`、`test_effects_recovery.py`、`test_data_layer.py`、`test_sessions.py` |

“能运行”或“随机系数接近真值”不能替代独立参照。发布模板等文本契约可以核对内容；
模块依赖按 AST 检查，不以源码出现某词代替行为验证。

## 精简范围与维护原则

当前完整集为 **205项**，由59个模块中的174个测试函数组成；相比上一轮916项，
已实际删除退出常驻集的函数、空模块及不再使用的辅助代码。默认core与full的入口没有变化。
本次不追求与旧矩阵等价覆盖：常驻集聚焦四模型公开工作流、独立数值参照、
样本/权重/DoF、预测与后估计、缓存失效，以及尺度、识别、收敛和预算的关键失败。

- 四模型的具名/数组/复用入口与论文输出协议保留；统计计算优先用dummy/WLS/GLM、SVD、Fraction或显式矩条件验证。
- 参数组合保留明确代表；单一固定情形直接写在函数内。保留函数中的原断言与容差不放宽，不用skip/xfail或收集筛选隐藏旧用例。
- 独立Fraction的2×2×3全部4,095非空子集及结构列空间有限穷举继续作为数学oracle。原分离语料只保留公开反例与关键FE/ReLU/mu路径，其余常规组合、随机种子及精细内部边界退出常驻集。
- FE执行保留14项重点数值/线程/预算回归；原26项中的布局探测、浮点资格、缓存工作区和profile等专项收缩。改动这些细节时，应恢复对应旧专项或增加有针对性的回归。
- 发布门禁、公共契约、实际依赖方向、架构图新鲜度、制品哈希和工作队列子进程仍检查；完整旧工具负面矩阵不再逐项常驻。
- 测试结果交给pytest/JUnit；基准与开发审计显式运行。新增功能优先扩展职责最接近的用例，并说明新增案例对应的独立风险。

原始Git历史仍保留上游测试；本次工作区还将上一轮916项及对应补丁完整归档在源码仓库外。
本页记录当前测试策略，历史报告只对应各自冻结版本。代表集合不能穷尽全部模型×VCE×权重×布局交互，
也不再常驻覆盖所有可选IO后端、低精度/布局资格、重采样分布和发布元数据畸形输入。

## 安装集与验证边界

`verify_installed_numerics.py` 将独立参照、公共工作流、样本语义及共享 fixture 复制到
源码之外，并检查 wheel 的 import provenance。它复用 HOST 依赖；
干净环境安装由 `verify_clean_install.py` 单独验证。
`--extra-test` 接受唯一 basename，例如 `test_weighted_reductions.py`；
缺失、歧义、路径逃逸与符号链接不能绕过选择检查。

本机完整测试不替代跨平台 CI、隔离构建或最终制品验收；内部 oracle 不等于
licensed-Stata/upstream 全部外部认证。历史 benchmark 与 Lean 证明也分别只支持
其记录的工作负载和数学命题。发布流程见[执行验收](../release/acceptance.md)。
