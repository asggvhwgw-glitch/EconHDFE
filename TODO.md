# EconHDFE 统一 TODO / ROADMAP

更新：2026-09-29。开发基线：`main@06cd0a5efe34de50c52d8b7c0db68c240b9d2673`。
目标：**0.6.5 已合并并完成主分支跨平台验收，尚未发布**。任务完成、代码审阅、本机执行和跨平台验收分开登记；本文件不承诺发布日期或尚未分配的 Owner。

## 0.6.3 已完成与保留边界

v0.6.3 已发布，冻结提交为 `2782d93f901ec1eed28ac0f963834701e0a44b09`。
原 REL-01（隔离构建/干净安装）、REL-02（12 格平台 + 4 格最低依赖）、
REL-03（执行门禁）、REL-04（提交/制品绑定），以及 CI-01/02/03 修复，
属于 **0.6.3 历史完成项**，不再作为当前开发阻塞。
PERF-01 与 QA-01 已合入。发布工作流已泛化，后续发布仍需明确授权。

来源：[0.6.3 Release](https://github.com/asggvhwgw-glitch/EconHDFE/releases/tag/v0.6.3)、
[基线 CI](https://github.com/asggvhwgw-glitch/EconHDFE/actions/runs/36299787125)。
旧发布维护记录 `RELEASE_CLOSEOUT.md` 保留为历史，不改写旧测试、哈希或数学文稿。
**旧版本通过不等于 0.6.5 通过**；新版本必须重新取得全部必要执行证据。
VAL-01 的 licensed-Stata/upstream 外部认证仍未完成；alpha 声明保持，不以普通发行冒充外部认证。

## A. 0.6.4 已合并开发基线：稳健性与有证据的内存优化

| ID | 优先级 | 当前状态 | 范围与验收 |
|---|---|---|---|
| ROADMAP-01 | P1 | 已更新 | 归档旧发布状态；将开发与发布门禁分开；保留全部历史 ID |
| QA-02 | P1，优先 | native 已实施；整项部分完成 | modular/rational 共用单个 residual component 的工作量预算；初始字典、fill-in 和整数增长在分配/运算前检查；超限显式失败，不返回模秩下界。SymPy/FLINT 的执行中资源限制及全任务硬 RSS/时间限制未实现，不宣称完成 |
| QA-03 | P1，优先 | 0.6.5 测试按契约/行为/数值/工具重组并完成 PR/main 跨平台验收；高风险边界持续补齐 | 预算不足、坏素数、恢复后重试、输入不修改、尺度/列置换、分块末尾非有限、空矩阵；缓存失效、Gram 和收敛边界继续按风险补齐 |
| PERF-03 | P1 | 已实施；0.6.4/0.6.5 跨平台与 bundle 验收通过 | 分块尺度/有限性检查，缩放直接写入 QR buffer；不改 QR/SVD、数值秩阈值或最小范数定义；独立稠密 SVD、C/F/sliced 输入与独立进程内存/耗时对照 |
| PERF-06 | P1 | 待实施验收 | 用完整回归表工作流测 session / persistent reuse / PPML warm-start；区分读入、编码、设计、投影、求解、推断；样本/FE/权重变化不得复用过期数值状态 |
| PERF-04 | P1 | 待 profile | 先区分单次拟合内部复用与跨规格 session 复用；仅在聚类编码/交互映射成本显著时实施缓存；保留标签、最终样本及多向交互语义 |
| PERF-05 | P1 | 待跨机器校准 | 多规模、不平衡 FE、不同核心数和内存；记录冷/热、真实路由、阶段耗时及 RSS；不依据本机结果改全局默认值 |
| PERF-02 | P1，低优先级 | 暂不合入 | 投影私有 z buffer 原型收益不稳定；本轮未重测，不宣称已有最终裁决；只有独立稳健收益才重开 |
| MATH-01 | P1 | 数学理论形式化已完成 | 三项 theorem-backed 工作的核心数学命题已映射到 Lean 4，并通过整库编译、逐定理公理审计与附录引用检查；软件实现、浮点行为和算法资源管理不属于本项完成标准 |
| VAL-01 | 按声明 | 并行待验收 | 独立数值 oracle 不等于 Stata 认证；保留参考版本、样本、权重、DoF/修正与差异分类 |

本轮实现说明及复现入口：`docs/development/maintenance-0.6.4.md`。

## B. 0.6.5 主分支验收与发布边界

REL-01/02/03/04 是可重复的门禁职责，不另造同义 ID。0.6.5 已完成本轮全量源码测试、installed-wheel 数值复验、构建/干净安装矩阵、16 格 CI、兼容性/Skill/技术文档/架构图检查，以及 PR 与合并后主分支 bundle 验证。PR #7 合并提交为 `1ef1c3998bd1eec6710e845949a965221184549d`；主分支 run #44（`36407706397`）结论为 success。

0.6.5 **尚未发布**：没有创建 tag、GitHub Release 或 PyPI 发布，也没有把普通 CI 候选制品冒充正式 detached release binding。若未来决定发布 0.6.5，仍需按发布流程冻结提交、绑定最终制品哈希并取得明确发布授权。

## C. 0.7 后估计开发（0.7.0.dev0，尚未合并/发布）

| ID | 范围 | 进入条件 |
|---|---|---|
| PRED-00 | 只读 fitted-design / prediction state | 标准 OLS/IV、session 与 persistent 已接入；保留原始标签、列、物理行语义。0.7.0.dev0 契约与执行证据独立登记 |
| PRED-01 | 明确尺度的预测 API | OLS/IV 的 response/xb/fe/stdp 已实现；categorical FE 显式 save_fe，unknown/跨分量/嵌套/extra-nullity 拒绝。PPML、varying-slope、group+individual 和通用可估函数仍待实施 |
| POST-00 | 单模型线性约束、Wald、contrasts | 名称/矩阵约束、非零原假设和单一有效秩判据已实现；零方差/非有限/非 PSD 拒绝；不包含跨模型联合协方差 |
| POST-01 | margins、AME、非线性变换 | 先定义预测目标与 FE 不确定性边界，不能用 beta-only 协方差冒充完整预测推断 |
| VCE-00/01 | 只读 score/bread/sample 契约与外部 VCE | 至少一个实际 provider；模型特定矩条件、权重、DoF 和归一化明确；不暴露可变 estimator 内部对象 |

## D. 需求驱动的研究/扩展候选

| ID | 主题 | 进入条件 |
|---|---|---|
| FUT-01 | 线性 IV 的 block-native 首阶段/弱识别诊断 | 实证 profile 证明必要；KP/SW 等完整语义与独立对照 |
| FUT-02 | extra-nullity、稀疏零空间、可估函数 | 明确应用/识别/预算，不把不可识别系数当唯一估计 |
| FUT-03a/b/c/d | FE SE；AKM leave-out；CRV3；更多 bootstrap | 每项独立文献、设计和 oracle，不混成一个承诺 |
| FUT-04 | GPU/IO/FLINT 等可选后端认证和扩展 | 实际设备/依赖；原 ID 保留，区分已有支持验证与新增功能 |
| FUT-05 | 跨模型联合协方差/suest-like | 共样本/重叠 cluster/矩条件对齐明确 |
| FUT-06 | PPML group + individual FE | 明确的实证需求和识别/分离设计 |
| FUT-08 | 通用 out-of-core HDFE | 真实瓶颈和资源预算；不为理论规模重构 solver |
| MATH-02 | 独立 prior-art/原创性复核 | 数学正确性已有 machine-checked 支持；历史原创性仍须独立文献复核，不因 Lean 验证而自动提升原创性状态 |

## 通用完成标准

不放宽既有测试、精度、停止容差或统计契约。性能补丁独立测量并保留全部重复计时；
计时与 profiling 分离。明确本机与跨平台边界，不承诺未经测量的速度。
每项变更检查 API、后端、错误、Skill、联动模块、结果、默认配置、缓存、测试、性能、
迁移说明、依赖和发布制品。新版本用 `scripts/version.py bump` 重置审阅/执行记录，
不覆盖历史 baseline、证据或已发布制品。

### 0.6.5 测试体系调整

见 `docs/development/testing.md` 和逐项迁移表。压缩重复测试不改变统计契约；
core 只用于日常开发，所有平台与 source archive 继续跑 full。
本机验收不能代替远端执行；该要求现已由 PR #7 及合并后 `main` run #44 满足。0.6.4 的旧证据未迁移为 0.6.5 通过。


## E. 数学形式化收尾（2026-09-29）

三项 theorem-backed 工作的数学理论形式化已经在 `dev/lean-foundations` 完成收口。完成标准仅覆盖数学命题及其明确假设，不包含 Python/Numba 程序验证、浮点收敛、资源预算、外部软件 parity 或历史原创性。权威映射见 `formal/mathematical-coverage.md`；三篇手稿的 Lean 附录片段位于 `formal/appendices/`。形式化工作并不改变 0.6.5 的未发布状态，也不触发 tag、GitHub Release 或 PyPI。
