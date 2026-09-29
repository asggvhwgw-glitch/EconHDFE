# econhdfe

[English](README.md) | [简体中文](README.zh-CN.md)

**面向现代实证研究的高性能高维固定效应计量经济学工具。**

`econhdfe` 是一个 Python 包，支持 OLS-HDFE、线性 IV-HDFE、PPML-HDFE 和 IV-PPML-HDFE。它面向大规模数据、高基数固定效应、聚类推断、复杂交互项，以及大量彼此接近的稳健性规格。

项目目前仍属于 **alpha 阶段研究软件**。仓库中的 0.6.5 开发线尚未正式发布；当前最新公开版本是 v0.6.3。仓库内已经有较完整的验证体系，但 licensed Stata 与完整 upstream external corpus 认证仍是独立的外部验证边界。

## 性能概览

`econhdfe` 的核心目标不是单纯把某一个矩阵核函数做得更快，而是尽量避免执行**用户所请求的计量模型根本不需要的计算**。

在一个 96,000 个观测、交互项较多的基准中，经过认证的结构化表示把物理设计矩阵从约 **75.3 MB 降低到 7.7 MB**。相对于本包同一规格的 dense 路径，实测运行时间分别改善为：

| 估计器 | Structured vs. dense | 最大系数差 |
|---|---:|---:|
| OLS-HDFE | **10.51×** | `3.1e-16` |
| Linear IV-HDFE | **1.30×** | `3.5e-15` |
| PPML-HDFE | **6.42×** | `5.8e-16` |
| IV-PPML-HDFE | **2.07×** | `1.7e-15` |

这只是**特定 workload 下的实测结果**，不是“所有回归都会快十倍”的承诺。其他规格可能通过不同机制获益，包括按需读取数据列、紧凑 ID 编码、精确结构约简、内存感知执行，以及相近规格之间的安全复用。

更完整的输入、基线、计时规则和数值一致性检查见 [benchmark 文档](docs/development/benchmarks.md)。

## 为什么需要 econhdfe？

设想一位研究者使用 5,000 万行产品层面的贸易数据。基准规格可能吸收 exporter-year、importer-year 和 product 固定效应，加入若干控制变量，并按贸易对聚类。

真正的实证工作不会停在第一条回归。研究者会更换因变量、增删控制变量、调整一个固定效应、改变聚类方式、估计 event-study，并最终形成一整张稳健性表格。

从计量上看，这些规格往往彼此非常接近；但传统“一条回归跑一次”的执行方式，可能不断重复支付同样的成本：重新读数据、重复编码几百万个分类标识、再次展开交互项、重新发现固定效应结构、重复做近似相同的 HDFE 投影、反复分配大型临时数组，再重建推断对象。

当数据足够大时，真正昂贵的部分往往已经不是最后那个小型系数求解，而是**数据移动、表示方式、固定效应投影、内存分配，以及相近规格之间的重复工作**。

`econhdfe` 的设计就是围绕这一点展开：

```text
传统工作流

data
  ↓
regression
  ↓
丢弃中间结果
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
reuse what remains valid
```

现代 Python 生态已经不同于早期科学计算时代。机器学习、大规模数据系统与 AI 推动了大量高性能基础设施围绕 Python 接口形成：NumPy、SciPy、Numba/JIT、优化 BLAS/LAPACK、列式数据系统、现代 dataframe engine、并行运行时、memory mapping 和可选 GPU backend，都可以位于 Python API 之后。

因此，`econhdfe` 把 Python 当作**协调层**，而不是让解释器循环去和 Mata 或编译代码竞争。重点是选择更好的表示、消除不必要工作，并把不同子问题送到合适的数值后端。

无论执行层如何优化，**计量模型本身始终是约束**。性能策略不能静默修改回归变量、工具变量、样本、权重、固定效应或用户要求的推断方式。

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

主要估计器入口保持简洁：

| 模型 | 入口 |
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

对于回归表中的重复规格，可以复用仍然有效的变换：

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

复用规则是保守的：如果样本、相关数据状态、权重或 FE 几何结构发生变化，就不能静默读取过期数值状态。

### 也可以直接交给 AI Agent

`econhdfe` 同时面向人类调用与 agent 驱动工作流。

仓库自带专用 Agent Skill：`skills/econhdfe/`。README 用来解释项目本身；Skill 则包含安装、模型设定、诊断、高级执行配置、benchmark、验证、隐私安全支持和第三方开发所需的操作知识。

Agent 只是软件接口，不是估计器的一部分。样本、系数、固定效应、收敛规则、推断方式和数值容差仍由同一套经过测试的公共 API 控制。

## econhdfe 的核心设计差异

本项目围绕**完整实证工作流**设计，而不是只优化一个孤立 solver。文件型数据源可以只暴露当前规格需要的列；分类标识可以在安全条件下编码一次并复用；symbolic design 可以在不必要的 dense matrix 生成之前完成精确约简；execution planner 则在不改变 estimand 的前提下，根据 workload 与资源预算选择经过认证的执行表示。

OLS、Linear IV、PPML 和 IV-PPML 共享底层 HDFE 基础设施，包括编码、拓扑、投影、DoF、加权投影和低层数值计算。这样无需维护四套互不相关的 fixed-effect 实现，而且公共核心的改进可以同时服务多个估计器。

重复规格被视为一等场景。对许多实证项目来说，避免第二次昂贵的 FE 变换，往往比让一个小型矩阵乘法再快一点更重要。因此 `econhdfe` 明确区分计量规格与可复用计算状态，并在样本或 FE 几何变化时主动失效相关缓存。

执行策略与计量语义也被严格分离。内存预算、线程数、结构化表示和缓存策略都是计算决策，不能重新定义模型。

## 不只是工程优化

`econhdfe` 也包含几条由数学理论直接支撑的工作线。

例如，对三个或更多 categorical fixed-effect 维度，吸收的结构自由度是

$$
\mathrm{DoF}_{FE} = \operatorname{rank}(D_{FE}).
$$

一维或二维 FE 的冗余结构有熟悉的图论表达；一般 multiway 情形则更复杂。本项目建立了一个 theorem-backed 框架，在明确假设下计算任意有限维 categorical FE 的 exact structural rank / absorbed DoF。

此外还有两项 theorem-backed 结果：

- **Residual-core reduction**：在满足条件的 multiway FE incidence 结构中精确剥离一部分观测，只在核心上求解，再把外围残差精确重构；
- **Partition-refinement structural reduction**：在 dense materialization 之前识别嵌套或冗余分类结构，用更小的精确基表示同一个请求的列空间。

这三条核心数学理论已经在其明确假设下使用 **Lean 4 完成 machine-checked formalization**。形式化库、逐命题覆盖表和三篇技术文档的形式化附录位于 `formal/`。

这里的 claim 边界非常重要：Lean 验证的是**明确映射的数学命题**，并不等于 Python/Numba 实现被形式验证，也不验证浮点收敛、benchmark 性能、外部 Stata parity 或历史原创性。

更多数学陈述、假设、实现映射、测试与 prior-art 边界见 [技术文档](docs/technical/README.md)。

## 主要能力

| 领域 | 当前支持 |
|---|---|
| OLS-HDFE | 一般 multiway FE absorption、IID/robust covariance、多维聚类、HAC 与 Driscoll-Kraay 路径 |
| Linear IV-HDFE | 2SLS、LIML、k-class、two-step GMM 和常规 weak-IV diagnostics |
| PPML-HDFE | IRLS、高维 FE 投影、separation handling、robust/clustered inference |
| IV-PPML-HDFE | 基于共享 weighted-HDFE infrastructure 的 additive-moment IV-PPML |
| Fixed effects | Multiway categorical FE、交互项、heterogeneous slopes、可选 exact structural rank/DoF |
| 重复规格 | 可复用 OLS / linear-IV session，并带 FE/sample-aware invalidation |
| Post-estimation | 面向论文输出的结果对象，以及可识别 categorical/indicator FE recovery |
| Execution | Projected data access、memory budget、structured representation、bounded parallelism 和 automatic HDFE threads |
| Compatibility | `reghdfe`、`ivreghdfe` 和 legacy `pyreghdfe` 入口 |
| Agent interface | 原生 `econhdfe` Skill，用于 agent 驱动实证与开发工作流 |

## 验证体系

性能优化只有在不静默改变模型时才有意义。因此 `econhdfe` 把公共接口、统计行为、独立数值 oracle 与 release engineering 分开验证。

当前测试体系围绕 **contracts、behavior、numerics、tooling** 组织。统计测试覆盖 realized sample、weights、singleton handling、FE semantics、omitted variables、clustering、DoF、PPML separation、结果语义和 cache invalidation。关键数值路径还会与 dense dummy matrix、SVD reference、exact rational arithmetic、困难 rank case、极端尺度、秩亏和资源边界做独立对照。

CI 覆盖 Linux、Windows、macOS 和支持的 Python 版本，并包含可行的 minimum-dependency 环境。source archive 和 installed wheel 也在普通开发 import path 之外重新验证。

必须强调：大量仓库内测试不等价于完整 licensed-Stata 或 upstream external-corpus 认证。详见 [Testing](docs/development/testing.md) 与 [validation status](docs/development/test-status.md)。

## 固定效应恢复

有些应用需要把固定效应本身作为经济对象，例如 worker-firm、mobility、origin-destination 和 structural gravity。

`econhdfe.effects` 提供 component-aware 的已识别 categorical / indicator FE recovery，并显式处理 normalization。识别与归一化是两个不同问题：改变 normalization 可以改变报告的 level，但不能凭空创造识别。

见 [Identified categorical fixed effects](docs/technical/identified-fixed-effects.md)。

## 兼容性

新代码通常应直接从 `econhdfe` 导入：

```python
from econhdfe import olshdfe, ivhdfe, ppmlhdfe, ivppmlhdfe
```

也提供兼容入口：

```python
from econhdfe import reghdfe, ivreghdfe
import pyreghdfe
```

这些别名用于迁移和互操作，并不表示所有 Stata 语法或 upstream edge case 都完成了外部认证。

## 文档

主文档索引：[docs/README.md](docs/README.md)。

建议入口：

- [技术总览](docs/technical/overview.md)
- [性能架构](docs/technical/performance-architecture.md)
- [经济学优先架构](docs/development/architecture.md)
- [Execution Planner](docs/development/execution-planner.md)
- [Benchmark 文档](docs/development/benchmarks.md)
- [数学与技术文档索引](docs/technical/README.md)
- [Lean 数学形式化](formal/README.md)

当前开发优先级见 [TODO.md](TODO.md)，公开版本之间的迁移说明位于 [docs/release/](docs/release/)。

## 贡献

贡献应继续维持这些边界：计量语义、HDFE 数学、估计器方程、推断、数据准备、数值 kernel 与执行策略彼此分离。

性能改动需要同时携带数值一致性检查和计时证据；新增技术性 claim 应同时给出数学陈述、实现对应、测试和 prior-art 边界。

见 [CONTRIBUTING.md](CONTRIBUTING.md) 与 [developer guide](skills/econhdfe/references/developer-guide.md)。

## 许可证

BSD 许可。见 [LICENSE](LICENSE)。
