[English](README.md) | [简体中文](README.zh-CN.md)

# econhdfe

**面向现代实证研究的高性能高维固定效应计量经济学工具。**

`econhdfe` 是一个 Python 计量经济学包，当前支持 OLS-HDFE、线性 IV-HDFE、PPML-HDFE 和 IV-PPML-HDFE。它面向大样本、高基数固定效应、多维聚类、复杂交互项以及大量相近规格反复估计的实证工作流。

项目目前仍属于 **alpha research software**。仓库主线是尚未发布的 0.6.5 开发版本；最近一次正式发布为 v0.6.3。仓库内已经有较完整的跨平台、数值和发布工程验证，但 licensed Stata 与完整 upstream external certification 仍是独立的验证边界。

## 性能概览

`econhdfe` 的性能目标不是单纯把最后一个矩阵求解器写得更快，而是尽可能**不做原始计量模型本来就不需要做的计算**。

在一个包含 96,000 个观测、较多交互项的记录 benchmark 中，经严格结构认证后的表示把物理设计矩阵从约 **75.3 MB 降到 7.7 MB**。与本包同一规格下的 dense execution path 相比，记录到的运行时间改善为：

| 估计器 | Structured / dense 加速 | 最大系数差 |
|---|---:|---:|
| OLS-HDFE | **10.51×** | `3.1e-16` |
| Linear IV-HDFE | **1.30×** | `3.5e-15` |
| PPML-HDFE | **6.42×** | `5.8e-16` |
| IV-PPML-HDFE | **2.07×** | `1.7e-15` |

这是一组**特定 workload 的 benchmark**，不是“所有回归都会快十倍”的承诺。不同问题的收益来源不同，包括列投影读取、紧凑 ID 编码、精确结构约简、内存感知执行以及相近规格之间的安全复用。

完整输入、基线、计时规则和 parity 检查见 [benchmark 文档](docs/development/benchmarks.md)。

## 为什么做 econhdfe？

考虑一个 5,000 万行的产品层面海关数据集。基准回归可能吸收 exporter-year、importer-year、product 固定效应，加入若干控制变量，并按贸易对进行聚类。

真正的实证研究通常不会停在第一条回归。研究者会：

- 更换因变量；
- 增减控制变量；
- 调整一个固定效应；
- 更换聚类规则；
- 估计 event-study；
- 最后生成几十列 robustness table。

这些规格在计量上往往高度相近，但传统“每次只跑一条”的执行方式可能不断重复支付：

- 重新读取同样的数据列；
- 重新编码数百万个类别 ID；
- 重新展开相同交互项；
- 重复发现同一组固定效应结构；
- 重新做近似相同的 HDFE 投影；
- 反复分配大型临时数组；
- 重新构造推断对象。

数据一旦足够大，最贵的部分经常不再是最后那个小规模系数求解，而是**数据移动、表示、固定效应投影、内存分配和跨规格重复工作**。

`econhdfe` 围绕这个事实设计：

```text
传统工作流

data
  ↓
regression
  ↓
丢弃中间状态
  ↓
next regression


econhdfe

data source
  ↓
validated encoded state
  ↓
symbolic econometric design
  ↓
exact structural analysis
  ↓
shared HDFE geometry
  ↓
execution planner
  ↓
OLS / IV / PPML / IV-PPML
  ↓
仅复用仍然有效的状态
```

现代 Python 生态使这种设计成为现实。重计算并不意味着解释器里的 Python for-loop：NumPy、SciPy、Numba/JIT、BLAS/LAPACK、Arrow 风格列式数据、现代 dataframe 引擎、并行 runtime、memory mapping 以及可选 GPU 后端都可以位于 Python API 之后。

本包把 Python 当作协调层，重点是：

1. 选对表示；
2. 删掉不必要的工作；
3. 把剩余计算交给适合的数值后端。

**计量模型本身始终是约束。** 性能优化不能静默改变样本、回归变量、工具变量、权重、固定效应或推断方式。

## 快速开始

支持 Python 3.10–3.13。

```bash
pip install econhdfe
```

标准 OLS-HDFE：

```python
from econhdfe import olshdfe

res = olshdfe(
    df,
    y="log_wage",
    x=["experience", "experience2"],
    absorb=["worker", "firm"],
    cluster=["firm"],
    vce="cluster",
)

print(res.params)
print(res.stderr)
```

主要估计入口：

| 模型 | API |
|---|---|
| OLS-HDFE | `olshdfe(...)` |
| Linear IV-HDFE | `ivhdfe(...)` |
| PPML-HDFE | `ppmlhdfe(...)` |
| IV-PPML-HDFE | `ivppmlhdfe(...)` |

线性 IV 示例：

```python
from econhdfe import ivhdfe

res = ivhdfe(
    df,
    y="outcome",
    exog=["control"],
    endog=["price"],
    instruments=["cost_shifter"],
    absorb=["firm", "year"],
    estimator="2sls",
    cluster=["firm"],
    vce="cluster",
)
```

对于回归表中大量相近规格，可以使用 Session：

```python
from econhdfe import OLSHDFESession

session = OLSHDFESession(
    df,
    cluster=["firm"],
    vce="cluster",
)

results = session.fit_many_y(
    y=["outcome_1", "outcome_2", "outcome_3"],
    x=["treatment", "control"],
    absorb=["firm", "year"],
)
```

复用规则是保守的。样本、相关数据状态、权重或 FE geometry 变化时，不允许静默读取过期数值状态。

### 也可以直接交给 AI agent

`econhdfe` 同时面向直接 Python 调用和 **agent-native** 工作流。

仓库内置项目专属 Skill：`skills/econhdfe/`。README 用于解释项目；Skill 则包含安装、模型规格、诊断、高级执行设置、benchmark、validation、隐私安全支持以及第三方开发的操作知识。

给 coding agent 的推荐起始提示可以是：

> Clone or access the EconHDFE repository at `https://github.com/asggvhwgw-glitch/EconHDFE`.
>
> Locate `skills/econhdfe/` and read `skills/econhdfe/SKILL.md` before using the package. Treat that Skill and its linked references as the authoritative operational guide.
>
> Inspect my Python environment and available compute resources, install or configure the appropriate EconHDFE version and optional dependencies, and verify that the package and core estimator interfaces work. Use the Skill when choosing advanced execution settings instead of guessing parameters from general Python knowledge.
>
> Do not modify my data or econometric specification merely to make the computation faster. Performance choices must preserve the requested model.

Agent 只是软件的接口之一，不参与估计器的统计定义。样本、系数、固定效应、收敛规则、推断和数值容差仍由相同的 tested public API 控制。

## econhdfe 在架构上有什么不同？

本包以**完整实证工作流**为优化对象，而不是只优化单个 solver。

支持的 file-backed data source 可以只暴露某一规格真正需要的列；类别 ID 可以在有效条件下复用编码；symbolic design 可以在创建无用的 dense matrix 之前先做精确结构简化；execution planner 再根据 workload 和资源预算，在数学等价的表示之间做物理执行选择。

OLS、线性 IV、PPML 和 IV-PPML 共用 HDFE 基础设施，包括：

- 编码；
- topology；
- projection；
- absorbed DoF；
- weighted projection；
- 底层数值运算。

因此四个模型族不会分别维护四套互不一致的固定效应实现。

重复规格也是一等场景。对很多实证项目而言，“避免第二次昂贵的固定效应变换”往往比把一次小矩阵乘法再提速几个百分点更重要。因此项目严格区分：

- econometric specification；
- reusable computational state。

一旦样本或 FE geometry 失效，复用必须失效。

## 不只是工程优化

构建一般化的 multiway HDFE 系统会遇到不能靠缓存和并行解决的数学问题。

例如多维 categorical FE 吸收的结构自由度，本质上是

[
mathrm{DoF}_{FE} = operatorname{rank}(D_{FE}).
]

一维、二维 FE 的冗余结构有经典图论解释；三维及以上的联合依赖更复杂。`econhdfe` 为任意有限维 categorical FE 提供 theorem-backed 的 exact structural rank / absorbed DoF 框架。

项目还维护另外两条 theorem-backed 数学线：

1. **Residual-core reduction**：在严格正对角权重等明确条件下，可以精确剥离 multiway FE incidence structure 中的叶结构，只在 residual core 上求解，再精确重构完整残差；
2. **Partition-refinement structural design reduction**：利用真实 partition refinement / functional dependency，在 dense materialization 前删除被证明冗余的 FE 或 factor directions，同时保持请求的列空间。

这三条核心数学理论现在已经在 **Lean 4** 中按明确假设完成 machine-checked formalization。

需要明确区分三个概念：

- **数学正确性**：Lean 检查明确写出的定理和假设；
- **实现正确性**：Python/Numba 实现仍依赖测试、numerical oracle 和发布验证；
- **历史原创性**：需要独立 prior-art / literature review，不因形式化证明自动成立。

参见 [技术文档](docs/technical/README.md) 和 [形式验证说明](docs/technical/formal-verification.md)。

## 主要能力

| 领域 | 当前支持 |
|---|---|
| OLS-HDFE | 一般 multiway FE absorption、IID/robust covariance、多维聚类、HAC 与 Driscoll-Kraay 路径 |
| Linear IV-HDFE | 2SLS、LIML、k-class、two-step GMM、常用 weak-IV diagnostics |
| PPML-HDFE | IRLS、高维 FE projection、separation handling、robust/cluster inference |
| IV-PPML-HDFE | 基于共享 weighted-HDFE infrastructure 的 additive-moment IV-PPML |
| Fixed effects | Multiway categorical FE、交互、heterogeneous slopes、可选 exact structural rank/DoF |
| Repeated specifications | 可复用的 OLS / linear-IV Session，并带 FE/sample-aware invalidation |
| Post-estimation | 面向发表结果以及已识别 categorical/indicator FE recovery |
| Execution | Projected data access、memory budgets、structured representation、bounded parallelism、自动 HDFE thread selection |
| Compatibility | `reghdfe`、`ivreghdfe` 与 legacy `pyreghdfe` 入口 |
| Agent interface | 面向 agent-driven 实证和开发流程的原生 EconHDFE Skill |

## 验证原则

性能优化如果静默改变模型，就没有意义。因此项目把以下层次分开验证：

- public-interface contract；
- statistical behavior；
- independent numerical checks；
- release engineering。

测试体系按 **contracts / behavior / numerics / tooling** 组织，而不是按历史版本编号组织。关键数值路径还会对照 dense dummy matrices、SVD reference、exact rational arithmetic、困难 rank cases、extreme scales、rank deficiency 和资源边界失败。

CI 覆盖 Linux、Windows 和 macOS，以及支持的 Python 版本和可行的 minimum-dependency 组合。Source archive 和 installed wheel 也会在不同于普通开发 import path 的环境下测试。

边界仍然明确：大量仓库内测试不等于完整 licensed-Stata / upstream external-corpus certification。

详情见 [Testing](docs/development/testing.md) 和 [validation status](docs/development/test-status.md)。

## 固定效应恢复

有些研究中固定效应本身就是经济对象，例如：

- worker-firm / AKM；
- mobility applications；
- origin-destination models；
- structural gravity 的后续阶段。

`econhdfe.effects` 提供 component-aware 的 identified categorical / indicator FE recovery，并显式区分 identification 和 normalization。

改变 normalization 可以改变报告的 FE level，但不能凭空制造 identification。

详见 [Identified categorical fixed effects](docs/technical/identified-fixed-effects.md)。

## 兼容性

新 Python 代码通常直接导入：

```python
from econhdfe import olshdfe, ivhdfe, ppmlhdfe, ivppmlhdfe
```

也保留兼容入口：

```python
from econhdfe import reghdfe, ivreghdfe
import pyreghdfe
```

这些 alias 用于降低迁移成本，不意味着已经复刻所有 Stata syntax 或所有 upstream edge case。

## 文档

总索引：[docs/README.md](docs/README.md)。

推荐入口：

- [技术总览](docs/technical/overview.md)
- [性能架构](docs/technical/performance-architecture.md)
- [经济学优先的代码架构](docs/development/architecture.md)
- [执行规划器](docs/development/execution-planner.md)
- [Benchmark 规则](docs/development/benchmarks.md)
- [技术文档与数学工作](docs/technical/README.md)
- [Lean 数学形式化](docs/technical/formal-verification.md)

当前开发优先级见 [TODO.md](TODO.md)。正式发布与迁移资料保存在 [docs/release/](docs/release/)。

## 贡献

贡献应继续保持以下层次分离：

- econometric semantics；
- HDFE mathematics；
- estimator-specific equations；
- inference；
- data preparation；
- numerical kernels；
- execution policy。

性能改动应同时提供 numerical parity 和 timing evidence；新的技术 claim 应包含数学陈述、实现对应、测试和 prior-art boundary。

见 [CONTRIBUTING.md](CONTRIBUTING.md) 和 [developer guide](skills/econhdfe/references/developer-guide.md)。

## 许可证

BSD 3-Clause。见 [LICENSE](LICENSE)。
