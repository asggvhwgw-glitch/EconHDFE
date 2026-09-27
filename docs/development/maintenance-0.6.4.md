# 0.6.4 第一轮开发：native 精确秩保护与线性代数内存

状态：**开发候选，未发布**。基线是 `main@669f166db717b2ec553a4c07dd6be7faa1ae5e43`，
源码取自该提交成功 CI 的 artifact `10924194551`，下载 ZIP 的 SHA256 为
`419e6f04bb8421262c5e7528aafac3d949b8baff45a74fb31288d746a01e470d`。
不是旧 submission 包，也不继承 0.6.3 的发布验收结论。

## 范围与停止线

计算逻辑仅修改两个既有模块：`econhdfe/hdfe/rank.py`、
`econhdfe/compute/stable_linalg.py`；另有 `__init__.py` 的版本元数据更新。
没有新增 estimator、依赖、公共参数、
错误码、结果字段或缓存层。API 与 0.6.3 baseline 的自动对照无差异。
Prediction / margins / VCE 插件仍留在后续 minor 周期。

### QA-02：native 部分完成，不是全任务硬资源上限

modular 与 rational 两个 native 消元阶段在一个 residual component 内共享预算。
上限为 50,000,000 个计费工作单位、256 MiB 的保守 Python 稀疏行存储估算、
16,384 位整数中间量。初始行字典创建前做准入检查，fill-in 的容器扩张与整数
交叉乘积在分配/运算前检查；计费是系数访问/更新的确定性代理，不是实际 CPU 指令计数。

超限异常只包含阶段、资源、计数和上限，不包含 FE 标签/观测。
异常意味着没有获得 exact rank，不能返回 GF(2)/mod-p 下界，不能静默改为 pairwise DoF。
直接 rank 调用抛 RuntimeError；估计器边界沿用既有错误封装，不增加公共错误代码。

**限制**：这是每个 component 的协作式保护，不是整个回归/多次 prefix-rank 的统一预算。
输入与拓扑构造、Python/JIT/分配器占用、SymPy/FLINT 的执行过程不在计费范围内；
不提供硬 RSS、超时或取消保证。FLINT 原有 cell cap 不变。
因此 QA-02 整项仍为部分完成，不能据此认证所有可选后端的资源安全。

### PERF-03：保持原求解公式，减少临时数组

原算法先创建整张 `abs(X)` 和有限性掩码；现在按既有 row-block 预算计算各列
绝对最大值并验证有限性，完成所有块验证后才调用 QR。缩放通过 `np.divide(..., out=...)`
直接写入私有 Fortran QR buffer。原 QR/SVD、数值秩 cutoff、最小范数定义、系数单位与
协方差公式不变。空设计仍检查 y 的有限性。没有采用 fastmath 或放宽任何容差。

数学/代码/测试映射：有限数据的全列最大值等于各块最大值再取最大值；其尺度向量不变。
`tests/test_stable_linalg_workspace.py` 用独立 dense NumPy SVD/lstsq、尺度/列置换、
秩亏、宽矩阵、C/F/sliced/只读输入、非有限值及 allocation 守卫验证实现。
`tests/test_native_rank_budget.py` 用 Fraction oracle 和坏素数验证 exact 回退与预算失败。
原 `test_math_resource_validation.py` 只调整私有 mock 的参数转发，原断言均保留。
没有新增定理或原创性主张，注册的三个技术文稿未改动。

## 同机测量（不是跨平台速度承诺）

每个工作负载、实现运行三次独立进程，交替 baseline/candidate 顺序；单线程。
另各运行一次 tracemalloc profiling，不把其耗时混入计时。原始重复值与环境见
`../release/evidence/0.6.4/development/performance.json`。

| 工作负载 | 基线 ms | 候选 ms | 候选/基线 | 基线/候选临时峰值 MiB |
|---|---:|---:|---:|---:|
| 2,000 × 8，C | 2.252 | 2.032 | 0.902 | 0.32 / 0.26 |
| 240,000 × 32，C | 262.479 | 199.432 | 0.760 | 58.60 / 16.53 |
| 240,000 × 32，F | 231.446 | 188.919 | 0.816 | 58.60 / 16.53 |
| 120,000 × 32，sliced | 154.378 | 139.670 | 0.905 | 29.36 / 16.53 |
| 1,000 × 128，C | 18.691 | 17.384 | 0.930 | 2.03 / 1.88 |

本组所有系数、bread 和 rank 对照一致，观测最大系数/bread 差异为零；
这不是对所有输入和平台的逐位一致承诺。临时峰值是 tracemalloc 跟踪的函数内分配，
**不是整个进程 RSS**。F-order 转换或 sliced backing storage 的既往峰值会影响
process high-water；相应 RSS 另列在 JSON，不混用两个指标。
本轮未预注册百分比速度门槛，结论是同机内存假说得到支持、五组未观察到计时回退；
保留候选继续跨平台验收，不主张统计显著的通用加速。

复现：从基线提交提取 `econhdfe/compute/stable_linalg.py`，然后执行：

```bash
python benchmarks/stable_linalg_workspace.py \
  --baseline /path/to/baseline/stable_linalg.py \
  --output /path/to/performance.json --rounds 3
```

本轮没有重测 PERF-02，没有 profile PERF-04，也没有完成 PERF-05/06 多机/完整工作流；
这些状态在 TODO 中保留，不以微基准冒充整项完成。

## 验证与发布状态

基线：931 passed。本轮定向测试：111 passed。最终源码：993 passed；
源码目录外 installed-wheel 数值子集：360 passed；六项 HOST smoke 全部通过。
最终源码/安装测试与门禁结果登记在 `../release/evidence/0.6.4/development/`；
执行状态以 `../release/execution.json` 为准。
构建若使用宿主 setuptools、已安装依赖或 `--no-build-isolation`，仅属 HOST 诊断；
不能登记为隔离构建或全新依赖解析安装。新版本的 16 格 CI、正式 artifact binding、
合并、tag 和发布均不由本地通过自动获得授权。
