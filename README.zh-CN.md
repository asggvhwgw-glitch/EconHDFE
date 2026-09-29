[English](README.md) | [简体中文](README.zh-CN.md)

# econhdfe

**面向现代实证研究的高性能高维固定效应计量经济学工具。**

`econhdfe` 是一个 Python 包，支持 OLS-HDFE、线性 IV-HDFE、PPML-HDFE 和 IV-PPML-HDFE。它面向大规模数据、高基数固定效应、聚类推断、复杂交互项以及大量相近规格反复估计的实证工作流。

项目目前仍属于 **alpha 研究软件**。仓库当前是尚未正式发布的 0.6.5 开发线，最新公开发布版本为 v0.6.3。仓库内已经有较完整的测试和数值验证，但 licensed Stata 以及完整 upstream external corpus 认证仍属于独立的外部验证边界。

## 性能概览

`econhdfe` 的核心目标不是单纯让某一个矩阵乘法更快，而是尽可能避免执行经济模型本身并不需要的计算。

在一组 96,000 个观测、交互项较多的已记录 benchmark 中，经过认证的结构化表示把物理设计矩阵从约 **75.3 MB 降到 7.7 MB**。相对于包自身在同一规格下的 dense 执行路径，实测速度提升为：**OLS-HDFE 10.51×**、**PPML-HDFE 6.42×**、**IV-PPML-HDFE 2.07×**、**线性 IV-HDFE 1.30×**；最大系数差异约在 `1e-16` 到 `1e-15` 之间。

| 估计器 | Structured 相对 dense | 最大系数差异 |
|---|---:|---:|
| OLS-HDFE | **10.51×** | `3.1e-16` |
| 线性 IV-HDFE | **1.30×** | `3.5e-15` |
| PPML-HDFE | **6.42×** | `5.8e-16` |
| IV-PPML-HDFE | **2.07×** | `1.7e-15` |

这是特定 workload 下的开发 benchmark，不代表所有回归都会快十倍。不同工作负载会通过不同机制受益，包括按需读取数据、紧凑类别编码、精确结构约简、内存感知执行，以及相近规格之间的安全复用。

详细输入、基线、计时规则和 parity 检查见 [benchmark 文档](docs/development/benchmarks.md)。

## 为什么做 econhdfe？

考虑一个使用 5,000 万行产品级贸易数据的研究者。基准回归可能吸收 exporter-year、importer-year 和 product 固定效应，加入若干控制变量，并按贸易对聚类。

真实的实证工作通常不会停在第一条回归。研究者会换因变量、增加或删除控制变量、调整固定效应、改变聚类方式、估计 event study，最后生成整张 robustness table。

从计量模型看，这些规格往往彼此非常接近；但如果把每条回归都当成完全独立的问题，程序可能一遍遍付出相同成本：重复读取同一批列、重新编码数百万类别变量、重新展开相同交互项、重新识别固定效应结构、再次投影相近变量、重新分配大数组，并重新构造推断对象。

当数据足够大时，真正昂贵的部分往往已经不是最后的小型系数求解，而是**数据搬运、表示方式、固定效应投影、内存分配，以及规格之间的重复工作**。

`econhdfe` 的架构围绕这个事实设计：

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
只复用仍然有效的状态
```

现代 Python 数值生态已经与早期科学计算时期非常不同。机器学习、大规模数据系统和 AI 的发展推动了大量高性能基础设施围绕 Python 接口形成。重计算并不需要运行在解释型 Python 循环里：NumPy、SciPy、Numba/JIT、BLAS/LAPACK、列式数据系统、现代 dataframe engine、并行运行时、memory mapping 以及可选 GPU backend 都可以位于 Python 接口之后。

因此，`econhdfe` 把 Python 当作协调层，而不是试图用 Python 循环和 Mata 或编译语言拼速度。重点是选择更合适的数据表示、删除不必要的工作，并把不同计算交给合适的数值后端。

**经济模型始终是约束。** 执行策略可以改变“怎么计算”，但不能静默改变研究者请求的 regressors、instruments、sample、weights、fixed effects 或 inference。

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

主要估计器入口保持尽可能小：

| 模型 | 入口 |
|---|---|
| OLS-HDFE | `olshdfe(...)` |
| 线性 IV-HDFE | `ivhdfe(...)` |
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

对于重复回归表，可以安全复用仍然有效的变换：

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

复用规则是保守的。样本、相关数据状态、权重或 FE 几何发生变化时，不允许静默读取已经失效的数值状态。

### 也可以交给 AI Agent

`econhdfe` 同时面向直接 Python 调用和 **agent-native** 工作流。

仓库在 `skills/econhdfe/` 下提供包专用 Agent Skill。README 用来解释项目本身；Skill 则保存安装、模型设定、诊断、高级执行参数、benchmark、验证、隐私安全支持和第三方开发所需的操作知识。

给 coding agent 的起始提示可以是：

> Clone 或访问 EconHDFE 仓库 `https://github.com/asggvhwgw-glitch/EconHDFE`。
>
> 在使用包之前找到 `skills/econhdfe/` 并阅读 `skills/econhdfe/SKILL.md`。把 Skill 及其链接的 reference 视为操作层面的权威说明。
>
> 检查我的 Python 环境和可用计算资源，安装或配置合适的 EconHDFE 版本及可选依赖，并验证包和核心估计接口可用。高级执行参数应根据 Skill 和实际环境选择，不要根据通用 Python 经验猜。
>
> 不要为了提升速度修改我的数据或计量规格。性能策略必须保留请求的模型。如果出现失败，使用 EconHDFE 的结构化诊断与隐私安全支持流程。
>
> 配置完成后，再询问我要运行的实证规格或研究任务。

Agent 只是软件接口，不是估计器的一部分。样本、系数、固定效应、收敛规则、推断和数值容差仍由同一个经过测试的公共 API 控制。

## econhdfe 的设计差异

这个包针对的是完整实证工作流，而不是一个孤立 solver。

文件型数据源可以只暴露当前规格实际需要的列；类别编码可以在仍然有效时复用；symbolic design 可以在 dense matrix 物化前进行结构约简；execution planner 可以根据实际 workload 和资源预算，从数学上等价的表示中选择物理执行方案。

OLS、线性 IV、PPML 和 IV-PPML 共享同一套 HDFE 基础设施，包括编码、拓扑、投影、DoF、加权投影和底层数值计算。这样避免维护四套互不相关的固定效应实现，也让公共计算核心的改进能够同时服务多种估计器。

重复规格是一等公民。很多实证项目中，避免第二次昂贵的 FE transformation，比把一次很小的 dense matrix multiplication 再快一点更重要。因此，`econhdfe` 明确区分“经济规格”和“可复用计算状态”，并在样本或 FE 几何变化时显式失效缓存。

包还严格区分 execution policy 和 econometric semantics。内存预算、线程数、结构化表示和 cache strategy 都只是计算决策，不能重新定义模型。

## 不只是工程优化

`econhdfe` 也包含一部分由通用 multiway HDFE 问题自然产生的数学工作。

例如，对于三个及以上 categorical fixed-effect 维度，吸收的结构自由度应由联合 FE design 的秩决定：

[
mathrm{DoF}_{FE} = operatorname{rank}(D_{FE}).
]

一到两个 FE 维度具有熟悉的图结构；一般多维 FE 的精确 rank 则更复杂。EconHDFE 包含一套 theorem-backed 框架，用于在明确假设下计算任意有限 (G) 个 categorical FE 的精确结构秩和 absorbed DoF。

这一区分非常重要：**数值表示的简化和经济模型的 rank 不是同一个对象。** 项目把用户请求的 FE topology、结构 rank/DoF、数值表示以及投影 solver 分开维护，从而避免性能优化静默改变待估设计。

目前项目还有另外两条 theorem-backed 数学工作：

1. **Exact residual-core reduction**：在 multiway categorical FE 投影中精确剥离满足条件的部分，只在 residual core 上求解，再重构完整残差；
2. **Exact partition-refinement structural reduction**：在完整设计矩阵物化前识别嵌套或冗余类别结构，用更小但列空间完全等价的表示替代请求设计。

这三条工作的**核心数学命题已经在 Lean 4 中按文档列明的假设完成 machine-checked formalization**。逐命题覆盖和假设见 [形式化验证文档](docs/technical/formal-verification.md) 与 [Lean 数学库](formal/README.md)。

这里的边界是严格的：

- Lean 验证的是数学命题；
- 不等价于 Python / Numba 实现的形式验证；
- 不验证浮点收敛、benchmark 性能或 licensed-Stata parity；
- machine-checked correctness 也不会自动证明历史原创性。

数学正确性、软件实现正确性和独立原创性始终作为三个不同问题处理。

## 主要能力

| 领域 | 当前支持 |
|---|---|
| OLS-HDFE | 一般 multiway FE absorption、IID/robust covariance、多向聚类、HAC 和 Driscoll-Kraay 路径 |
| 线性 IV-HDFE | 2SLS、LIML、k-class、two-step GMM 和常规 weak-IV diagnostics |
| PPML-HDFE | IRLS、高维 FE projection、separation handling、robust 与 clustered inference |
| IV-PPML-HDFE | 基于共享 weighted-HDFE 基础设施的 additive-moment IV-PPML |
| Fixed effects | multiway categorical FE、交互项、heterogeneous slopes、可选 exact structural rank/DoF |
| 重复规格 | OLS 和线性 IV session，带 FE/sample-aware invalidation |
| Post-estimation | 面向论文输出的结果对象，以及可识别 categorical/indicator FE recovery |
| Execution | projected data access、memory budgets、structured representation、bounded parallelism 和自动 HDFE 线程选择 |
| Compatibility | `reghdfe`、`ivreghdfe` 和 legacy `pyreghdfe` 入口 |
| Agent interface | 原生 `econhdfe` Skill |

## 验证

性能优化只有在不改变模型的情况下才有意义。EconHDFE 因此分开维护：

- 公共接口测试；
- econometric behavior 测试；
- 独立数值 oracle；
- release engineering。

当前测试按 **contracts / behavior / numerics / tooling** 组织，而不是按历史版本号堆叠。统计测试覆盖 realized sample、weights、singletons、FE semantics、omitted variables、clustering、DoF、PPML separation、result semantics 和 cache invalidation。关键数值路径还会与 dense dummy matrix、SVD reference、exact rational arithmetic、困难 rank case、极端尺度、rank deficiency 和资源边界等独立构造比较。

CI 覆盖 Linux、Windows 和 macOS，并覆盖支持的 Python 版本和可行的最低依赖环境。源码包和安装后的 wheel 也会离开普通开发 import path 单独检查。

边界仍然明确：仓库内大量测试不等价于完整 licensed-Stata 或 upstream external-corpus certification。没有完成的外部检查不会因为内部测试很多就被视为完成。

详见 [Testing](docs/development/testing.md) 和 [validation status](docs/development/test-status.md)。

## 固定效应恢复

有些研究需要把固定效应本身作为经济对象，而不是 nuisance parameter，例如 worker-firm、mobility、origin-destination 和 structural gravity。

`econhdfe.effects` 支持按 connected component 恢复可识别的 categorical / indicator fixed effects，并显式指定 normalization。

识别与归一化是两件不同的事：改变 normalization 可以改变报告的水平，但不能凭空创造 identification。

详见 [Identified categorical fixed effects](docs/technical/identified-fixed-effects.md)。

## 兼容接口

新 Python 代码通常应直接导入：

```python
from econhdfe import olshdfe, ivhdfe, ppmlhdfe, ivppmlhdfe
```

同时保留：

```python
from econhdfe import reghdfe, ivreghdfe
import pyreghdfe
```

这些别名用于迁移和 interoperability，并不表示所有 Stata syntax 或 upstream edge case 都已完成外部认证。

## 文档

总文档入口见 [docs/README.md](docs/README.md)。

常用技术入口：

- [Technical overview](docs/technical/overview.md)
- [Performance architecture](docs/technical/performance-architecture.md)
- [Economics-first architecture](docs/development/architecture.md)
- [Execution planner](docs/development/execution-planner.md)
- [Benchmark documentation](docs/development/benchmarks.md)
- [Technical documentation](docs/technical/README.md)
- [Lean / formal verification](docs/technical/formal-verification.md)

当前开发优先级见 [TODO.md](TODO.md)，公开版本之间的迁移说明位于 [docs/release/](docs/release/)。

## 贡献

贡献应保持 econometric semantics、HDFE mathematics、estimator-specific equations、inference、data preparation、numerical kernels 和 execution policy 之间的边界。

性能改动应同时提供数值 parity 和 timing evidence；新的技术结论应包含数学陈述、实现对应、测试以及 prior-art boundary。

参见 [CONTRIBUTING.md](CONTRIBUTING.md) 和 [developer guide](skills/econhdfe/references/developer-guide.md)。

## 许可证

BSD License。见 [LICENSE](LICENSE)。
