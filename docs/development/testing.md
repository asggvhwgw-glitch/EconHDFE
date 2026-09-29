# 测试设计：以当前契约组织，而非以历史版本堆积

## 当前范围与历史来源

本页说明 0.7 的测试入口与职责。分层结构在 0.6.5 开发阶段建立，
历史迁移过程见 [迁移记录](history/test-migration-0.6.5.md)。
当前已纳入 Post/Prediction、IV 识别与数值修复；实际执行状态见 [test-status.md](test-status.md)。

测试首先回答：功能是否实现，调用入口是否对齐，同一经济计量对象的语义是否稳定。
“调用没有报错”“Monte Carlo 系数大致接近真值”“源码包含某字符串”不能替代这些验收。
文档模板及命令声明等本身就是文本契约的检查仍保留；模块依赖改用 AST 解析。

## 四层测试与运行入口

| 层 | 职责 | 典型检查 |
|---|---|---|
| `tests/contracts/` | 对研究者和调用端的承诺 | 根 API/兼容别名、参数边界、命名数据/数组/复用入口、结果和错误、隐私 |
| `tests/behavior/` | 功能与统计语义 | OLS/IV/PPML/IV-PPML、样本/权重/FE、VCE 与 DoF、遗漏变量、separation、session/SPJ |
| `tests/numerics/` | 不能以接口烟雾测试替代的独立守卫 | 稠密 dummy/SVD/Fraction oracle、坏素数、秩亏、尺度、资源拒绝、block/solver |
| `tests/tooling/` | 开发和制品完整性 | 当前公共契约快照、架构/注册表、候选与正式门禁、来源和哈希、安装验证入口 |

```bash
# 日常开发：当前公共行为，不等于完整发布验收
python scripts/run_tests.py --suite core -- -q
# 无路径的 pytest 也默认只运行 core
python -m pytest -q

# 数值核修改：定向检查之后必须跑完整集
python scripts/run_tests.py --suite numerics -- -q
python scripts/run_tests.py --suite tooling -- -q
python scripts/run_tests.py --suite full -- -q

# 单项复现继续使用正常 pytest 节点，不需专用 DSL
python scripts/run_tests.py -- tests/behavior/test_sample_semantics.py -q
```

`core` = contracts + behavior。`full` = 整个 tests 目录，包含数值及发布工具守卫。
无路径的 `pytest` 默认行为自 0.6.5 起发生变化，不能再把它的通过数登记为“完整测试”。
CI 的 12 个 OS/Python 组合、4 个最低依赖组合，以及源码制品重测均明确选择 `full`。
保留线程要求 `NUMBA_NUM_THREADS>=3`（通常设 4），不偷偷跳过并行测试。

## 当前公共能力矩阵

| 功能 | 成功结果的判据 | 拒绝/特殊语义 | 主要位置 |
|---|---|---|---|
| 四类模型与入口 | 同一规格的命名数据、数组、可复用对象得到相同系数/VCE/样本 | 不强行统一模型本来不同的参数和字段 | contracts/test_public_workflows.py |
| 回归表输出 | 系数/SE/统计量/p/CI/星号及模型量按名称和次序对齐 | 诊断默认隐藏；IV 首阶段不输出巨型观测矩阵；归一化常数不报告假精度 | contracts/test_public_workflows.py |
| 样本 | 明确最终保留行；递归 singleton 与显式子样本一致 | singleton、分离和外部 DataFrame index 不混淆 | behavior/test_sample_semantics.py、test_effects_diagnostics.py、numerics/test_separation_corpus.py |
| 权重 | fweight 与真实展开样本等价；a/pweight 和通用权重按各自契约 | 非整数频数、非有限值和非法质量明确失败 | behavior/test_weights_and_effects.py、numerics/test_estimator_oracles.py |
| FE 与 DoF | 请求的 FE/推断拓扑与数值简化分离；独立秩对照 | 嵌套 FE 的拟合自由度不混用 cluster 推断自由度 | behavior/test_fit_degrees_of_freedom.py、numerics/test_exact_oracles.py |
| 因子与遗漏项 | 明确 reference、交互、保留系数名称及顺序 | 默认告知遗漏，显式 drop/raise 语义独立 | contracts/test_factor_variables.py、behavior/test_*omissions.py |
| 聚类与推断 | 单向/多向、校正、cluster 数、弱 IV 诊断有独立参照 | 单 cluster、非正推断自由度、秩亏结果不伪装有效 | behavior/test_cluster_inference.py、numerics/test_inference_boundaries.py |
| 重复回归 | 换 y/控制项/FE 的结果及拟合统计量与独立拟合一致 | 数值列、FE、权重等变化不可读到旧缓存 | behavior/test_sessions.py、test_data_layer.py、test_weight_updates.py |
| PPML / IV-PPML | eta/mu、offset/exposure、dense/block、矩条件结果一致 | separation/refit、极端数值、未收敛和标准化常数有明确语义 | behavior/test_ppml*.py、test_ivppml*.py、numerics/test_nonlinear_oracles.py |
| Post/Prediction | 线性组合/Wald、分块 xb/response/fe/stdp 与独立 dummy/WLS 对照一致 | 未知类别、样本映射、未识别/嵌套 FE 组合与 beta-only 不确定性 | behavior/test_postestimation.py、test_prediction.py |
| FE recovery | 重建贡献、归一化和可识别分量匹配 | 不可识别系数用明确状态/NaN；关闭诊断不改变求解结果 | behavior/test_effects_recovery.py、numerics/test_inference_boundaries.py |
| 资源限制 | 在可行范围内与独立 oracle 一致 | 达到预算只能明确失败，不能将模秩下界当 exact | numerics/test_native_rank_budget.py、test_resource_boundaries.py |
| 安装与发布 | wheel 来自独立 target，源码外执行真实契约与数值子集 | 旧证据、错误哈希、缺失制品和歧义路径拒绝 | scripts/verify_installed_numerics.py、tests/tooling/ |

这里的“入口对齐”不等于改 API：线性模型使用 `cluster`，非线性模型使用 `clusters`；
PPML 与 IV-PPML 的 weight_type 支持不同；线性 `.params` 数组与非线性 `.coef` 数组也不伪装成同一结构。
这些差异应被测试明确约束，未来有意更改时再走公共兼容性流程。

## 压缩规则与任务清单

已将原 67 个平铺、经常按版本命名的文件迁移为按职责命名的四层目录。
40 个冗余或陈旧测试函数由更强的当前检查覆盖；10 个矩阵减少重复随机种子、恒等尺度
或改用显式代表组合，原保留案例的断言和容差不放宽。详见 [逐项迁移表](history/test-migration-0.6.5.md)。

组合缩减不是穷尽检验：两两组合覆盖不能保证所有三因素相互作用。为此单独保留
空矩阵、只读/非连续输入、极端单位、近秩亏、坏素数、整数资源耗尽、分离和缓存失效等反例。
4095 子集的独立 Fraction 秩穷举与既有 separation corpus 不以“数量多”为理由删掉。
不将大量案例塞进不可定位的循环，也不通过 xfail/skip 把错误隐藏。

实施任务：审计与旧→新映射；当前公共工作流测试；重复函数/组合压缩；运行/安装/CI 入口联动；
完整执行、逐行/逐分支迁移核查、定向故障注入；最后重新生成架构图与重置新候选执行记录。
新的验证报告独立保存，历史报告和已发布兼容基线不改写。

## CI 与安装集

分支 push 不再和同一 PR 触发两套相同矩阵；只保留 main push、PR 和手动触发。
同一 PR 的过期运行可取消，main 与 PR 的并发组不同，不互相冒充验证结果。
本轮不缩小 OS/Python/最低依赖支持面，也不以 core 替换原有完整 CI。

installed-wheel 测试保持独立目录和 import provenance 检查，复制新的分层文件以及共享 fixture，
纳入四模型公共工作流及样本测试。`--extra-test` 仍接受唯一 basename；歧义、路径逃逸或缺失应失败。
例如 `--extra-test test_weighted_reductions.py`。该测试仍使用 HOST 依赖，不等于干净依赖解析。

## 验收证据与限制

基线来源、完整/核心计数、逐行逐分支覆盖变化、独立进程计时、故障注入和制品检查
见本轮随附证据。测试数量不是质量指标；保留关键失败语义比追求一个整数目标重要。
覆盖率也不是正确性证明：本轮逐项检查丢失行/分支，并以故意引入错误验证关键测试确实会失败。
六种定向错误被捕获不等于系统性 mutation score=100%。
本机验证不替代新的 16 格 CI，HOST 构建不替代隔离构建，内部 oracle 不替代 licensed-Stata 认证。
