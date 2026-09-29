# EconHDFE v0.7.0 — Alpha research release

## 中文

本次发布整合 Post/Prediction、IV 正确性修复以及中英文文档。项目仍为 Alpha 阶段研究软件。

- 标准 OLS/线性 IV 支持冻结设计状态、分块 `predict()`、原始样本行位置恢复，以及显式 `save_fe=True` 保存的分类固定效应预测。未知类别、违反嵌套关系和未识别组合明确拒绝。
- 线性组合与 Wald 检验保留非有限值、负方差、秩亏和零方差保护；`stdp` 仅包含 beta 协方差，不包含固定效应不确定性或完整预测区间。
- IV 增加矩条件识别与数值秩保护；敏感 2SLS 使用压缩工具变量空间 QR/SVD；修正加权 first-stage fitted 值的坐标及 fixed-k 协方差，并使用调用内有界诊断缓存。
- 整合此前 0.6.4/0.6.5 开发线的资源边界守卫与测试体系整理。测试入口为 `python scripts/run_tests.py --suite full`；普通 pytest 只运行日常核心集。
- 中英文首页、技术文档、历史目录和发布打包入口统一。旧持久化结果缓存按 ABI 安全失效；已有 within 数值缓存 ABI 不变。

支持 Python 3.10–3.13。升级命令：`pip install --upgrade econhdfe==0.7.0`。

发布流程使用成功的 16 环境 CI（Linux/macOS/Windows 与可行最低依赖环境）及 bundle 验证产出的冻结制品，经独立验收后发布；GitHub Release 与 PyPI wheel/sdist 字节相同，哈希见 `SHA256SUMS.txt`。

边界：不包含 PPML/IV-PPML、varying-slope/group+individual FE 样本外预测、margins/AME；不新增 licensed-Stata/upstream 外部认证、普遍提速或全流程内存预算保证。Lean 证明覆盖明确假设下的数学命题，不等于 Python/Numba 浮点程序的形式验证或历史原创性认证。

## English

This alpha research release brings together Post/Prediction, IV correctness fixes, and bilingual documentation.

- Standard OLS and linear IV gain frozen prediction state, chunked prediction, original-row restoration, and explicitly saved categorical fixed effects. Unknown categories and unidentified or inconsistent FE combinations fail explicitly.
- Linear contrasts and Wald tests retain finite-value, variance and rank safeguards. Prediction `stdp` is beta-only; it is not a full prediction interval or an estimate of FE uncertainty.
- IV estimation adds cross-moment identification/rank guards and a compressed instrument-coordinate QR/SVD solve for sensitive 2SLS systems. Weighted first-stage coordinates and fixed-k covariance are corrected, with bounded call-local diagnostic reuse.
- Includes resource guards and the reorganized full test suite from the unreleased 0.6.4/0.6.5 development lines, together with English/Chinese documentation and packaging cleanup.

Python 3.10–3.13 is supported. Install with `pip install --upgrade econhdfe==0.7.0`.

Publication consumes the frozen candidate from successful matrix and bundle validation after detached acceptance; it does not rebuild after tagging. GitHub and PyPI share the exact wheel/sdist bytes.

PPML/IV-PPML prediction, varying-slope/group+individual FE prediction, margins/AME, complete FE uncertainty and full external Stata/upstream certification remain outside this release's claims. No universal performance, end-to-end memory-bound or implementation-formal-verification claim is added.
