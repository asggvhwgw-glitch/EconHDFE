# EconHDFE 统一 TODO / ROADMAP

更新：2026-09-27。开发基线：`main@34d5138aa771f707ebd300bed34aa2ccad32c478`。
目标：**0.6.5 开发中，尚未发布**。任务完成、代码审阅、本机执行和跨平台验收分开登记；本文件不承诺发布日期或尚未分配的 Owner。

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
| QA-03 | P1，优先 | 测试按契约/行为/数值/工具重组，本机验收；新 CI 待运行 | 预算不足、坏素数、恢复后重试、输入不修改、尺度/列置换、分块末尾非有限、空矩阵；缓存失效、Gram 和收敛边界继续按风险补齐 |
| PERF-03 | P1 | 已实施，本机测量；跨平台待验收 | 分块尺度/有限性检查，缩放直接写入 QR buffer；不改 QR/SVD、数值秩阈值或最小范数定义；独立稠密 SVD、C/F/sliced 输入与独立进程内存/耗时对照 |
| PERF-06 | P1 | 待实施验收 | 用完整回归表工作流测 session / persistent reuse / PPML warm-start；区分读入、编码、设计、投影、求解、推断；样本/FE/权重变化不得复用过期数值状态 |
| PERF-04 | P1 | 待 profile | 先区分单次拟合内部复用与跨规格 session 复用；仅在聚类编码/交互映射成本显著时实施缓存；保留标签、最终样本及多向交互语义 |
| PERF-05 | P1 | 待跨机器校准 | 多规模、不平衡 FE、不同核心数和内存；记录冷/热、真实路由、阶段耗时及 RSS；不依据本机结果改全局默认值 |
| PERF-02 | P1，低优先级 | 暂不合入 | 投影私有 z buffer 原型收益不稳定；本轮未重测，不宣称已有最终裁决；只有独立稳健收益才重开 |
| MATH-01 | P1 | 本轮映射已更新；持续任务 | 资源拒绝不改变秩证明；分块最大值与缩放保持原公式；测试不是数学证明 |
| VAL-01 | 按声明 | 并行待验收 | 独立数值 oracle 不等于 Stata 认证；保留参考版本、样本、权重、DoF/修正与差异分类 |

本轮实现说明及复现入口：`docs/development/maintenance-0.6.4.md`。

## B. 0.6.5 发布门禁（每个新版本重新执行）

REL-01/02/03/04 是可重复的门禁职责，不另造同义 ID。
当前 0.6.5 必须完成：全量源码测试、已安装 wheel 数值复验、隔离 sdist→wheel、
干净依赖环境安装、16 格 CI、兼容性/Skill/技术文档/架构图检查、最终外置制品绑定。
`docs/release/maintenance.json` 记录影响审阅；`execution.json` 记录执行。
未运行/阻塞不能用 reviewed 代替，旧证据不能迁移；创建 tag、合并和发布需维护者授权。

## C. 0.7 后估计基础（本轮不实施）

| ID | 范围 | 进入条件 |
|---|---|---|
| PRED-00 | 只读 fitted-design / prediction state | 复用现有 design/effects，冻结列、样本、类别与归一化语义；不保存不必要的原始数据 |
| PRED-01 | 明确尺度的预测 API | 先 categorical FE；新 level、不识别组合/跨组件组合必须明确拒绝或 NaN，不默认为零；varying-slope 单独设计 |
| POST-00 | 单模型线性约束、Wald、contrasts | 参数/协方差对齐与有效秩；跨模型比较需要联合协方差，不能默认独立 |
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
| MATH-02 | 独立正确性与 prior-art/原创性复核 | 两类结论分开；强化原创主张前前置，不把现有登记视作新颖性认证 |

## 通用完成标准

不放宽既有测试、精度、停止容差或统计契约。性能补丁独立测量并保留全部重复计时；
计时与 profiling 分离。明确本机与跨平台边界，不承诺未经测量的速度。
每项变更检查 API、后端、错误、Skill、联动模块、结果、默认配置、缓存、测试、性能、
迁移说明、依赖和发布制品。新版本用 `scripts/version.py bump` 重置审阅/执行记录，
不覆盖历史 baseline、证据或已发布制品。

### 0.6.5 测试体系调整

见 `docs/development/testing.md` 和逐项迁移表。压缩重复测试不改变统计契约；
core 只用于日常开发，所有平台与 source archive 继续跑 full。
本机验收不能代替本轮测试/CI 修改后的远端执行。0.6.4 合并前后的通过证据不迁移为 0.6.5 通过。
