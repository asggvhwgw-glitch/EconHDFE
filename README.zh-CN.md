# econhdfe

[English](README.md) | [简体中文](README.zh-CN.md)

**面向现代实证研究的高性能高维固定效应计量经济学工具。**

`econhdfe` 是一个用于 OLS-HDFE、线性 IV-HDFE、PPML-HDFE 和 IV-PPML-HDFE 的 Python 包，面向大规模数据、高基数固定效应、聚类推断、复杂交互项以及大量相近规格的实证工作流。

项目目前仍属于 **alpha 阶段研究软件**。仓库中的开发线为尚未正式发布的 0.7.0.dev0，最近一次公开发布版本是 v0.6.3。仓库内已经进行了较广泛的验证，但 licensed Stata 对照和完整 upstream external certification 仍属于独立的外部验证边界。

## 性能概览

`econhdfe` 的核心目标不是单纯让某一个线性代数 kernel 更快，而是尽可能**避免执行计量模型本身并不需要的计算**。

在一组已经记录的复杂交互 benchmark 中，样本量为 96,000。经过严格结构认证后，物理设计矩阵由约 **75.3 MB 降至 7.7 MB**。与包自身在同一规格上的 dense execution path 相比，测得运行时间分别改善为：**OLS-HDFE 10.51×**、**PPML-HDFE 6.42×**、**IV-PPML-HDFE 2.07×**、**线性 IV-HDFE 1.30×**，同时最大系数差异保持在约 `1e-16` 到 `1e-15`。

| 估计器 | Structured vs. dense | 最大系数差异 |
|---|---:|---:|
| OLS-HDFE | **10.51×** | `3.1e-16` |
| 线性 IV-HDFE | **1.30×** | `3.5e-15` |
| PPML-HDFE | **6.42×** | `5.8e-16` |
| IV-PPML-HDFE | **2.07×** | `1.7e-15` |

这只是一个特定 workload 的 benchmark，并不意味着所有回归都会快十倍。其他规格可能通过不同机制获益，例如按需读取数据列、紧凑 ID 编码、精确结构约简、内存感知执行，以及相近规格之间的安全复用。

完整 benchmark 输入、基线、计时规则和数值一致性检查见 [benchmark 文档](docs/development/benchmarks.md)。

## 为什么做 econhdfe？

设想一个研究者正在处理 5,000 万行产品层面的贸易数据。一个基准规格可能吸收 exporter-year、importer-year 和 product 固定效应，加入若干控制变量，并按 trade pair 聚类。

但真实的实证工作通常不会停在第一条回归。研究者会更换因变量、加入或删除控制变量、调整某个固定效应、尝试其他聚类方式、估计 event study，最后形成一整张 robustness table。

这些规格在计量上往往非常接近，但如果每次都把回归当作完全独立的任务，就可能反复承担同一批成本：重复读取数据列、编码数百万个分类 ID、重新展开相同交互项、重新发现固定效应结构、对类似变量重复做 HDFE 投影、重新申请大块临时内存，以及反复构造推断对象。

当数据规模足够大时，最贵的部分往往已经不是最后那个很小的系数求解，而是：

- 数据移动；
- 数据与设计表示；
- 固定效应投影；
- 内存分配；
- 相近规格之间的重复工作。

`econhdfe` 的架构正是围绕这一点设计。

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
安全复用仍然有效的状态
```

现代 Python 生态使这种架构比早期科学计算时代更现实。机器学习、大数据和 AI 推动了大量围绕 Python 接口的高性能基础设施。重计算并不需要运行在解释型 Python 循环里：NumPy、SciPy、Numba/JIT、优化后的 BLAS/LAPACK、Arrow 风格列式数据、现代 dataframe engine、并行 runtime、memory mapping，以及可选 GPU backend，都可以隐藏在统一的 Python 接口之后。

因此，`econhdfe` 把 Python 主要作为**协调层**，而不是试图让解释型 Python 循环与 Mata 或编译代码硬拼。重点是选择更好的表示、避免无意义的计算，并把不同任务交给更合适的数值后端。

但计量模型始终是约束。执行方式可以改变，**请求的样本、回归变量、工具变量、权重、固定效应和推断规则不能因为性能优化而被静默改变**。

## 快速开始

支持 Python 3.10–3.13。

```bash
pip install econhdfe
```

标准 OLS-HDFE 可以直接对 pandas DataFrame 调用：

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

主要估计入口保持刻意简洁：

| 模型 | 入口 |
|---|---|
| OLS-HDFE | `olshdfe(...)` |
| 线性 IV-HDFE | `ivhdfe(...)` |
| PPML-HDFE | `ppmlhdfe(...)` |
| IV-PPML-HDFE | `ivppmlhdfe(...)` |

线性 IV 例如：

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

对于重复回归表，`econhdfe` 可以在规格变化后继续复用仍然有效的变换：

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

复用规则是保守的。样本、相关数据状态、权重或 FE 几何发生变化时，旧的数值状态不能被静默复用。

### 也可以交给 AI agent

`econhdfe` 除了可以直接从 Python 调用，也按照 **agent-native** 的方式组织。

仓库在 `skills/econhdfe/` 中提供项目专用 Agent Skill。README 主要用于帮助研究者理解项目；Skill 则保存更完整的操作知识，包括安装、规格设定、诊断、高级执行参数、benchmark、validation、隐私安全支持流程和第三方开发说明。

使用 coding agent 时，可以从下面的提示词开始：

> Clone or access the EconHDFE repository at `https://github.com/asggvhwgw-glitch/EconHDFE`.
>
> Locate `skills/econhdfe/` and read `skills/econhdfe/SKILL.md` before using the package. Treat that Skill and its linked references as the authoritative operational guide.
>
> Inspect my Python environment and available compute resources, install or configure the appropriate EconHDFE version and optional dependencies, and verify that the package and core estimator interfaces work. Use the Skill when choosing advanced execution settings instead of guessing parameters from general Python knowledge.
>
> Do not modify my data or econometric specification merely to make the computation faster. Performance choices must preserve the requested model. If something fails, use EconHDFE's structured diagnostics and privacy-safe support workflow.
>
> Once setup is complete, ask me for the empirical specification or research task I want to run.

Agent 只是软件的一个接口，不属于估计器本身。样本、系数、固定效应、收敛规则、推断方法和数值容差仍由与人工调用相同的、经过测试的公开 API 控制。

## econhdfe 与普通实现有什么不同？

项目面向的是**完整实证工作流**，而不是一个孤立 solver。

文件型数据源可以只暴露当前规格真正需要的列；分类 ID 可以在语义允许时编码一次并复用；符号化设计可以在不必要的大型 dense matrix 生成之前进行精确约简；execution planner 可以在等价的物理表示中，根据工作量和资源预算进行选择。

OLS、线性 IV、PPML 和 IV-PPML 共用同一套 HDFE 基础设施，包括编码、拓扑、投影、自由度、加权投影和底层数值模块。这样既不需要维护四套互不关联的固定效应实现，也意味着对公共核心的优化能够同时服务多个 estimator family。

重复规格被视为一等工作流。对很多实证项目来说，避免第二次昂贵的 FE transformation，可能比把一次很小的 dense matrix multiplication 再优化几个百分点更重要。因此，`econhdfe` 明确区分 econometric specification 与 reusable computational state，并在样本或 FE 几何变化时显式失效相关缓存。

执行策略同样与计量语义分离。内存预算、线程数、结构表示和缓存策略都是计算决策，不能重新定义模型。

## 不只是工程优化

`econhdfe` 不只是一个工程项目。实现任意多维 HDFE 时会出现一些无法仅靠缓存或并行解决的结构问题。

一个典型问题是三维及以上分类固定效应的 absorbed degrees of freedom。若 $D_{\mathrm{FE}}$ 是组合固定效应设计矩阵，则核心目标是：

$$\operatorname{DoF}_{\mathrm{FE}} = \operatorname{rank}(D_{\mathrm{FE}}).$$

对于一维和二维 FE，冗余具有熟悉的图结构。到了任意多维 FE，精确结构秩问题更加复杂。项目中已经有一个 theorem-backed 框架，在明确假设和资源边界下计算任意有限维 categorical FE 的 exact structural rank 和 absorbed DoF。

这一点重要，是因为“为了计算方便而简化设计”和“计量意义上的设计秩”不是同一个对象。实现上会明确区分：

- 用户请求的 FE topology；
- structural rank / DoF；
- 数值表示；
- projection solver。

这样可以避免优化过程静默改变实际估计的设计。

项目还有另外两条 theorem-backed 数学工作：

1. **Exact residual-core reduction**：在昂贵的多维 FE 数值求解之前，精确消去满足条件的 incidence 结构，并在求解后重构完整残差；
2. **Exact partition-refinement structural reduction**：在完整 materialization 之前识别嵌套和冗余分类结构，用更小的精确 basis 表示同一请求列空间。

这三条工作的核心数学命题目前已经在各自明确假设下使用 **Lean 4 完成 machine-checked formalization**。定理、证明覆盖、实现映射、测试与 prior-art boundary 分别维护在 [技术文档](docs/technical/README.md) 和 [形式化验证状态](docs/technical/formal-verification.md) 中。

这里必须明确区分几件事：

- Lean 验证的是明确陈述的数学定理；
- 它不等价于 Python/Numba 实现已经被形式验证；
- 它不替代 floating-point 数值验证或 benchmark；
- 它也不自动证明历史原创性。

数学正确性、实现正确性、数值验证和独立的历史优先权是四类不同的 claim。

## 主要能力

| 领域 | 当前支持 |
|---|---|
| OLS-HDFE | 一般多维 FE absorption、IID/robust covariance、多向聚类、HAC 和 Driscoll-Kraay 路径 |
| 线性 IV-HDFE | 2SLS、LIML、k-class、two-step GMM 和常规 weak-IV diagnostics |
| PPML-HDFE | IRLS、高维 FE 投影、separation handling、robust 与 clustered inference |
| IV-PPML-HDFE | 基于共享 weighted-HDFE infrastructure 的 additive-moment IV-PPML |
| Fixed effects | 多维分类 FE、交互项、heterogeneous slopes、可选 exact structural rank/DoF |
| 重复规格 | 可复用的 OLS 和 linear-IV session，并带 FE/sample-aware invalidation |
| Post-estimation | 论文结果输出、线性组合/Wald、分块 OLS/IV 预测，以及显式保存的分类 FE（0.7 开发线） |
| Execution | projected data access、memory budgets、structured representations、bounded parallelism、automatic HDFE thread selection |
| Compatibility | `reghdfe`、`ivreghdfe` 与 legacy `pyreghdfe` 入口 |
| Agent interface | 项目原生 `econhdfe` Skill，用于 agent-driven empirical 与 development workflow |

## 验证

性能优化只有在不改变模型时才有意义。因此 `econhdfe` 把公开接口测试、统计行为测试、独立数值 oracle 和 release engineering 分开管理。

当前测试体系按照 **contracts、behavior、numerics、tooling** 组织，而不是按历史版本号组织。统计测试覆盖 realized sample、weights、singleton handling、FE semantics、omitted variables、clustering、DoF、PPML separation、result semantics 与 cache invalidation。

关键数值路径还会与独立构造进行对照，例如：

- 显式 dense dummy matrix；
- SVD-based reference；
- exact rational arithmetic；
- 特意构造的困难 rank case；
- extreme scale；
- rank deficiency；
- resource-boundary failure。

CI 覆盖 Linux、Windows 和 macOS，以及支持的 Python 版本和可行的最低依赖环境。source archive 与 installed wheel 也会脱离普通开发 import path 单独验证。

边界仍然明确：仓库内部的大规模验证并不等价于完整 licensed-Stata 或 upstream external corpus certification。

详见 [Testing](docs/development/testing.md) 和 [validation status](docs/development/test-status.md)。

## 后估计与预测

尚未发布的 0.7 开发线提供 `linear_combination()`、`wald()`，以及标准 OLS 和线性 IV
的分块预测。`result.predict()` 返回最终估计样本的拟合值；`restore_sample=True`
按原始物理行位置恢复结果。新数据的 `kind="xb"` 使用冻结设计；`kind="stdp"`
只包含 beta 协方差，不包含固定效应的不确定性。

新数据的 `response` 和 `fe` 要求通过 `save_fe=True` 显式保存受支持的分类固定效应；
未知水平和未识别的新组合会被拒绝。本候选不支持 PPML/IV-PPML 预测、varying-slope
或 group+individual FE 样本外预测，也不包含 margins 和 AME。
详见[后估计契约](docs/development/postestimation-0.7.md)。安装已发布的 PyPI 版本
不代表已包含这些开发功能。

## 固定效应恢复

一些研究并不只把固定效应当作 nuisance parameter，而是直接把它作为经济对象。例如 worker-firm、mobility、origin-destination 和 structural gravity。

`econhdfe.effects` 提供 component-aware 的 identified categorical / indicator fixed-effect recovery，并要求明确 normalization。

识别与归一化是两个问题：改变 normalization 可以改变报告的 FE level，但不能把本来未识别的对象变成已识别。

详见 [Identified categorical fixed effects](docs/technical/identified-fixed-effects.md)。

## 兼容接口

新的 Python 代码通常应直接从 `econhdfe` 导入：

```python
from econhdfe import olshdfe, ivhdfe, ppmlhdfe, ivppmlhdfe
```

同时保留兼容入口：

```python
from econhdfe import reghdfe, ivreghdfe
import pyreghdfe
```

这些 alias 用于降低迁移和互操作成本，并不表示所有 Stata syntax 或 upstream edge case 都已经完成独立外部认证。

## 文档

总文档入口：[docs/README.md](docs/README.md)。

主要技术入口包括：

- [Technical overview](docs/technical/overview.md)
- [Performance architecture](docs/technical/performance-architecture.md)
- [Economics-first architecture](docs/development/architecture.md)
- [Execution planner](docs/development/execution-planner.md)
- [Benchmark documentation](docs/development/benchmarks.md)
- [Technical documentation / theorem-backed work](docs/technical/README.md)
- [Machine-checked mathematical verification](docs/technical/formal-verification.md)

当前开发优先级见 [TODO.md](TODO.md)。公开版本之间的迁移说明位于 [docs/release/](docs/release/)。

## 贡献

贡献应继续保持以下层次的边界：

- econometric semantics；
- HDFE mathematics；
- estimator-specific equations；
- inference；
- data preparation；
- numerical kernels；
- execution policy。

性能改动应同时提供数值一致性检查和计时证据。新的技术性 claim 应包含数学陈述、实现对应、测试和 prior-art boundary。

详见 [CONTRIBUTING.md](CONTRIBUTING.md) 和 [developer guide](skills/econhdfe/references/developer-guide.md)。

## 许可证

BSD 许可证。见 [LICENSE](LICENSE)。
