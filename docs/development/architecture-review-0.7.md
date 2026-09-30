# 0.7 代码结构审查

审查基线：`e738333ad8a4c07fa15e3587238e4997ba21c19f`（2026-09-30）。
范围是模块职责、重复逻辑、接口复杂度与外围文档，不是全包数值正确性证明。
用 AST 扫描 122 个运行模块，人工核对命中项，并用现有依赖契约测试检查层次约束。
生成的 [架构图](architecture-map/architecture.md) 记录 import 证据；下述职责判断是人工审阅。

## 结论与本次处理

主要层次仍可辨认：模型负责估计方程，HDFE 负责吸收/结构秩，IV 提供结果模型无关的
工具变量原语，compute 负责低层计算。没有必要另建通用 estimator 框架来容纳预测。
但文档滞后、少量重复代码和较长的入口函数确实存在。

| 发现 | 依据与影响 | 处理 |
| --- | --- | --- |
| 两份相同的 union-find 合并核 | `effects/topology.py::_union_pairs` 与 `hdfe/rank.py::_union_edge_pairs` 的去注释 AST 函数体相同；两者维护路径压缩和按大小合并 | effects 复用已有 HDFE 内部核，保留原内部别名；effects 原本已依赖 rank，不增加反向依赖、公共接口或另一套图模块 |
| PPML 与 IV-PPML plan 刷新重复 | 两个 `api.py` 的 `_ensure_plan_current` 均为 15 行，同样检查行数/签名、重建 FEPlan 并清空 context | 记录为后续生命周期重构；不为 15 行重复引入持有任意 model 对象的新抽象 |
| OLS/IV 入口偏大 | AST 计数：`olshdfe` 300 行/40 个具名参数，`ivhdfe` 360 行/48 个具名参数，包含 keyword-only 参数 | 保留兼容签名；后续先固化标量参数与 config 优先级，再抽取内部准备、求解、结果装配阶段 |
| “根模块都是薄包装”不成立 | `design.py` 为 938 行、`sessions.py` 为 1003 行；pipeline 还承担样本、列和 FE 准备 | 修正文档定位；按编译、样本准备、复用生命周期划分后续工作，不按文件长度机械切分 |
| 预测模块名称相近，职责说明不足 | state、新行执行、拟合时 FE 认证、参数推断分别有不同输入与时机 | 明确下表边界，不合并成一个包含全部状态和统计逻辑的大 result 类 |
| 用户文档与真实入口不一致 | README 写 `wald()`，实际公开结果方法是 `wald_test()`；状态/manifest 页面仍以旧版为当前标题 | 修正 API 名称、当前入口与历史归档，明确源码版本不等于已发布版本 |

长度和参数数只用于定位审阅热点，并不证明错误。重复扫描比较去除文档字符串后的
函数体 AST，限定至少 8 行；它发现完全相同的语法体，不识别全部语义重复，也不证明
其他代码无冗余。没有为了降低这些计数而改变估计器、数值容差或公开接口。

## 后估计和预测的职责边界

| 模块 | 职责 | 不应承担 |
| --- | --- | --- |
| `results.py` | 结果字段和薄方法转发 | 新行设计编译或 FE 识别算法 |
| `postestimation.py` | 对既有系数/协方差做线性组合与 Wald 推断 | 模型重拟合、跨模型联合协方差 |
| `prediction.py` | 冻结样本、active design、类别与 FE 元数据 | 保留原始 DataFrame 或求解估计方程 |
| `prediction_api.py` | 校验目标，分块重建新行设计，计算 response/xb/fe/stdp | 重新选择因子基准或静默填补未知类别 |
| `effects/prediction.py` | 显式 save_fe 后构造分类 FE 映射与识别证书 | 改写已完成估计，或将资源失败当识别成功 |
| `hdfe/rank.py` | 分类结构秩与其内部图操作 | 结果展示和新行预测策略 |

`predict_linear` 本身118 行，已集中多个目标和输入分支。若扩展非线性预测，先建立
各模型的目标/识别/不确定性契约，再决定如何拆分；不把 PPML 强塞进 identity-link 分支。

## 接口复杂度：下一步边界

1. 公共入口的兼容性优先。`cluster`/`clusters`、线性 `params`/非线性 `coef` 的区别
   已被测试约束，不应为了名字一致而悄悄改签名或结果字段。
2. 标量参数与 `HDFEConfig`/`InferenceConfig`/`ExecutionConfig` 存在重叠。
   先明确冲突时哪个值生效，并覆盖 array、DataFrame、group/individual 路径；
   再抽内部参数解析。不要新增第四个平行配置对象来掩盖优先级。
3. 合并 PPML plan 生命周期逻辑前，补齐两模型对修改 FE 标签、行数变化、
   `cache_validation="signature"/"none"` 和无 FE 情形的行为验证。
4. session、design 的后续拆分以状态生命周期/编译阶段为单位，保留唯一的样本与
   FE 身份来源。不要在 results、session 和 prediction 中复制一套可变模型状态。

## 验证入口

```bash
python scripts/run_tests.py -- tests/tooling/test_dependency_contract.py tests/behavior/test_effects_recovery.py tests/behavior/test_prediction.py tests/numerics/test_exact_oracles.py -q
python scripts/run_tests.py --suite full -- -q
python scripts/compatibility.py diff
python scripts/version.py gate
python scripts/generate_architecture_map.py --check
```

完整回归与 FE/预测独立 oracle 用于守住共享核的行为；不添加仅断言“两个私有函数
是同一对象”的实现镜像测试。实际执行结果以本 PR 的固定提交和 CI 为准。
本 PR 改变了运行/验证指纹，不能沿用冻结 `e738333` 的发布验收或制品哈希。
