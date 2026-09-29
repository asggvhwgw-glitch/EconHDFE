# 0.7 后估计：预测目标、状态与识别边界

当前版本：`0.7.0`。开发候选为 `0.7.0.dev0`，原始工作基线是 `main@22e176b` 与 PR #12 的 `3ba2b4c`。
这是开发实现，不代表 0.7 已发布；0.6.5 的快照及历史证据不改写。

## 预测接口

本轮扩展标准 OLS/线性 IV 的 `RegressionResult`，不扩展 PPML/IV-PPML 结果类。

```python
res.linear_combination({"treatment": 1.0, "control": -1.0}, value=0.0)
res.wald_test([{"treatment": 1.0}, {"control": 1.0}], values=[0.0, 0.0])
res.predict()                           # 最终估计样本 fitted，含 FE
res.predict(restore_sample=True)        # 按原始物理行位置补回 NaN
res.predict(newdata, kind="xb")         # 保留参数化下的 X beta
res.predict(newdata, kind="stdp")       # beta-only 标准误
res.predict(newdata)                    # 含 FE，需要对应保存状态
res.predict(newdata, kind="fe")         # 保存归一化下的 FE 总贡献
res.predict(newdata, unknown="nan", chunk_size=65536)
res.predict(X=active_design, kind="xb") # 显式已变换矩阵，按 res.names 顺序
```

`data` 是原始标签的 DataFrame 或列映射，不能同时提供 `X`。`X` 为二维且列数等于系数数目的
已变换矩阵；不会添加常数、猜测公式、重选基准类别。裸矩阵不支持含 FE 的样本外预测。
返回 NumPy 一维数组，保持输入行顺序；DataFrame index（包括重复 index）不是样本身份。

无输入的 `predict()` 复制已有 fitted，不重新拟合。无输入的 `xb`/`fe` 需要保存 FE 贡献
（无 FE 时例外）；`stdp` 始终要求数据或显式矩阵，因为结果不保留原始设计矩阵。
`restore_sample` 只用于无新数据时，恢复原始物理行，不是 fweight 展开后的观察值。

## 实现与资源

`prediction.py` 保存冻结的样本/设计/类别/FE 契约；`prediction_api.py` 按冻结规则分块重建；
`effects/prediction.py` 是拟合结束后的 FE 映射适配器，复用 topology 和 native exact rank，
不另造 FE 求解器。`results.py` 只作薄转发。

默认拟合不新增原始 DataFrame 或 N×K 设计缓存，也不为预测重跑 FE 识别。标准 OLS/IV 显式
`save_fe=True` 才构建 FE 原始标签和识别映射；额外成本取决于 level 数、样本规模及精确秩
工作量，并非免费。level 映射按类别存储，删行样本信息使用 packed bits。预测工作矩阵为
`chunk_size × K`，最终返回向量仍占 O(N) 内存。后估计不反向调用估计器，不重复吸收 FE。

文件源/EncodedEconometricDataset 可能将标识符先编码为整数。本实现只在小型预测映射中
恢复原始标签，保留实际拟合使用的类别顺序、参考组和系数名称，不重定义模型。新数据必须
提供原始标签，不是另一次 factorize 得到的 codes。

结果缓存 ABI 为 `linear-session-result-3`，旧结果安全失效；within 数值 ABI 不变。
使用允许列表式 dataclass + JSON/NPY，不开放 pickle。数值和字符串映射以不可写 bytes
缓冲区持有；其他 object 标签只提供隔离副本及只读标志，不声称任意 Python 对象深度不可变。

## 固定效应预测的识别条件

设有效 FE 设计为 D，新行 FE 线性函数为 d；含 FE 的预测要求 d 对 D 的零空间正交，否则
归一化会改变预测值。本实现使用现有 exact component rank：只有通常平移自由度的
categorical component 内的新组合可接受；跨 component 或额外零空间（extra-nullity）
分量按策略拒绝/NaN。extra-nullity 分量采取保守拒绝，尚未提供通用可估函数分析。因此
显式 `newdata` 中即使提供原样本某行，也可能保守拒绝；已存 fitted 仍可直接读取。

数值规范化若删除嵌套/等价 FE，新行还必须保持原请求 FE 的 level 和 fine→coarse 映射。
只检查有效 FE 会错误接受违反训练嵌套关系的新组合。这里复用 `save_fe` 已有的 minimum-norm
系数，不另选经济归一化；保存 beta 副本以拒绝修改参数后继续叠加旧 FE 的样本外预测。

识别使用有资源守卫的 native 后端。遇到其资源上限，回归及已有 fitted 保留，仅将新行 FE
预测能力标为不可用，原因是 `identification_resource_budget`。该守卫不是进程级硬 RSS/
时间限额。非收敛 FE、opaque FE 和 varying-slope FE 不冒充可用。

## 推断与失败语义

`stdp = sqrt(diag(X V_beta X'))` 只计入 beta 协方差，不是完整 FE-inclusive 预测的标准误，
也不包含未来因变量噪声；本轮没有完整预测区间和 FE SE。保留参数化下的 `xb` 不自动等于
原始秩亏设计的可估函数。自动遗漏回归变量时，新行 `response` 首版明确拒绝，而允许明确
请求 `xb`；用户主动指定的参考项/遗漏项属于模型定义，允许。IV 中遗漏的排除工具变量不
影响结构结果预测，预测不要求重新提供这些工具变量。线性 IV 的 response 是估计的结构指数加 FE，
不承诺等于给定内生变量的条件均值，也不自动赋予任意新数据情景因果解释。

未知 factor/FE level、未观察到的 factor interaction cell、缺失/非有限数值、未识别 FE
组合默认报错；`unknown="nan"` 只将相应行设为 NaN，不将未知类别当作零。缺列、opaque
设计、形状错误、不支持目标和缺少 FE 状态始终报错，不被 NaN 策略掩盖。错误复用既有
InputError/SpecificationError/ShapeError/InferenceError，stage 为 postestimation；错误载荷
只含必要计数，不包含原始观察值/类别样本。

主要代码：`prediction.unknown_level`、`unknown_fe_level`、`unobserved_cell`、
`unidentified_combination`、`omitted_estimability`、`fe_unavailable`、`nonfinite`、
`missing_column`、`coefficient_alignment`、`matrix_shape`、`invalid_variance`。

结果对象的后估计方法使用现有 error_boundary，保留可定位的 postestimation stage，不掩盖编程错误。
线性约束使用已保存的 V 和推断自由度。Wald 用同一次特征分解决定有效秩及逆，拒绝显著
负特征值、无方差方向中的非零限制差及非有限输入；重复约束不重复计自由度。零估计方差
不再返回伪装确定的 p=1。F/χ² 仍是常规参考约定，不是弱 IV 稳健检验；跨模型联合检验仍
需要联合协方差，不能假设各模型独立。

## 验证与保留范围

行为测试覆盖命名/矩阵、factor/交互、block-native、OLS/IV、session/persistent、物理行/
fweight、文件标签、未知/嵌套/断开/extra-nullity、输入不修改、资源拒绝和不支持路径。
数值测试用独立稠密 dummy/WLS 检查新行预测及 beta 协方差，不只比较包内矩阵乘法。
执行证据按本次代码另存，旧 CI run #67 不代表本轮通过。

未实现：PPML/IV-PPML 预测、group+individual 样本外预测、varying-slope FE 样本外预测、
秩亏模型的通用可估函数、完整 FE/预测不确定性、margins/AME 和跨模型联合协方差。

接口参考为 reghdfe 官方帮助的 xb/stdp/xbd/d 区分及固定效应可识别性说明：
https://scorreia.com/help/reghdfe.html 。接口参照不构成 licensed-Stata 外部认证。
