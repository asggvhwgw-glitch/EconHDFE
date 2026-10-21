# IV 分阶段结果与诊断计算开关

## 计算范围

`InferenceConfig.iv_advanced_statistics` 默认为 `True`，保持既有完整诊断。
设为 `False` 时真正跳过高级统计量的计算，适用于只需要常用诊断的重复估计。

| 内容 | `False` 时 |
| --- | --- |
| 系数、协方差、标准误、样本定义与必需秩/识别检查 | 保留 |
| 普通首阶段 F（经典及当前 VCE 对应的稳健/聚类 F）、partial R² | 保留 |
| Kleibergen–Paap（KP）、过度识别检验 | 保留 |
| Shea partial R²、AP/SW、Cragg–Donald（CD）、Stock–Yogo | 跳过 |

KP 等检验仍遵循原有适用条件与不可用值规则。保留一个检验不等于保证所有模型下
都能计算出有限数值；本开关不改变检验的计量定义。

关闭时，高级结果键不出现在对应结果字典中，同时
`result.diagnostics["advanced_statistics"]` 明确记录
`enabled=False`、`status="not_requested"` 和 `omitted` 字段名元组。
不以零、NaN 或空表伪装成已执行检验。开启时不新增这一状态项，保持既有默认结果结构。

`InferenceConfig.diagnostics="off" | "publication" | "full"` 继续仅控制显示，
与计算开关相互独立。关闭高级计算后，`publication_output()` 和显式完整诊断输出
都只包含实际完成的统计量；`diagnostics="full"` 不会触发补算。

## 分阶段读取

```python
from econhdfe import ivhdfe, InferenceConfig

def show_stage(event):
    if event["stage"] == "estimation":
        print(event["names"], event["params"], event["stderr"])
    elif event["stage"] == "diagnostics":
        print(event["diagnostics"]["kleibergen_paap"])

result = ivhdfe(
    data, y="y", exog=["x"], endog=["d"], instruments=["z1", "z2"],
    absorb=["firm", "year"], cluster="firm", vce="cluster",
    inference_config=InferenceConfig(iv_advanced_statistics=False),
    on_stage=show_stage,
)
```

`on_stage` 默认为 `None`。回调在同一线程中同步执行，`ivhdfe` 最终仍返回
`RegressionResult`。不保留用于后台补算的 N 级矩阵，也不引入后台任务。
回调自身的处理时间计入完整拟合耗时；绘图、磁盘或网络工作会延迟估计继续执行。

按顺序发送以下独立快照。`status="complete"` 表示该阶段完成，只有
`stage="complete"` 才表示完整拟合结束。

| `stage` | 内容 |
| --- | --- |
| `estimation` | 原单位 `params`、`vcov`、`stderr`、`nobs`、`nobs_raw`、`rank`、`df_resid`、`df_absorbed`、`vce`；`diagnostics_status="pending"` |
| `first_stage` | 首阶段 `diagnostics` 列表和 `advanced_statistics` 布尔值 |
| `diagnostics` | 识别检验字典 `diagnostics` 和 `advanced_statistics` 布尔值 |
| `complete` | `result.publication_output(include_diagnostics=True)` 的独立副本，放在 `result` 键内；另有 `cached` 标志 |

各阶段附有 `names` 和 `outcome`。具名因变量使用其列名；直接传数组时
`outcome=None`。`complete.result["coefficients"]` 保持既有 publication schema，
是系数表 DataFrame；需要 JSON 时由调用方按其输出格式序列化。
早期阶段不包含尚未完成的最终拟合统计量，也不包含完整结果对象。
所有快照排除逐观测拟合值和残差；修改快照不影响估计结果。

若后续诊断、结果组装或回调自身失败，公共错误边界照常抛出异常，没有
`complete` 事件。收到早期系数不能当作完整拟合成功。回调不用于取消某项
诊断；计算范围应通过配置在拟合前确定。

支持既有 dense、heterogeneous block、group/individual，以及 2SLS、
LIML/k-class、GMM2S 路径原本允许的组合；本功能不扩展各路径的估计器支持范围。

## Session 与缓存

`IVHDFESession(data, inference_config=..., on_stage=...)` 将相同策略应用于
`fit` 和 `fit_many`；通过 `outcome` 区分因变量。回调逐个估计发送，
不并行调用。

计算开关包含在完整结果缓存键内，关闭后的结果不会命中开启时的完整结果缓存。
回调本身不进入缓存键。持久缓存命中时只发送
`stage="complete", cached=True`，不伪造未执行的中间阶段；
仅 FE/within 缓存命中时仍正常执行估计并发送四阶段。

## 验证入口

- `tests/behavior/test_iv_stages.py`：计算跳过、保留数值、单位、事件顺序、
  快照隔离、异常、各路径与持久缓存。
- `tests/numerics/test_iv_primitives.py` 和 `test_iv_correctness_closeout.py`：
  原有数值与识别/协方差回归验证。

默认完整计算和最终返回类型不变。已接入的矩乘积复用、聚类聚合共享及数值边界
见[单次拟合内的计算复用](iv-performance-reuse.md)。
