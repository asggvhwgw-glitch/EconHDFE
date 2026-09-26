# 0.6.3 发布前维护记录

本轮是尚未正式放行的 0.6.3 候选修复，不覆盖已交付包，不发布 tag、GitHub Release 或 PyPI。
GitHub 基线：`213e33817e26bbd47ff8a2b892e07e678650aae8`。
本地输入：`econhdfe-v0.6.3-source.zip`；运行包、脚本、测试、CI、Skill、兼容性目录
的 Git tree SHA 与上述主分支一致。主分支的公开文档及历史证据脱敏增量应保留。

## 修复范围与接口

- `scripts/compatibility.py`：统一非空标准 Enum 的继承构造签名表示，避免 Python
  3.10/3.11 的解释器反射差异导致 `VariableRole` 假阳性。自定义 metaclass、显式
  signature 和普通函数变更仍检查；历史 baseline/approval 文件不改写。
- `econhdfe/hdfe/projection.py`：在既有线程控制上下文中，仅为非线程安全的 Numba
  `workqueue` 加模块级可重入锁。即使线程 mask 为 1，并发进入 parallel kernel 仍不安全。
  OpenMP/TBB 不加锁、不改默认线程数。此锁只协调本包受管的 parallel region，
  不是对任意外部 Numba 线程程序的安全承诺。
- `econhdfe/models/ols_bootstrap.py`：将进程级 BLAS 限额放到整个 joblib 批次外侧，
  避免工作线程相互覆盖/恢复线程设置。抽样种子、统计公式与返回结构不变。
- Windows 模板测试按 UTF-8 文本比较，仅归一化换行；新增 CRLF 与内容篡改反例。
  `.gitattributes` 固定普通源码 LF；历史执行证据和二进制文件不作换行转换。
- `scripts/build_release.sh` 使用 `python -m build --outdir` 默认 sdist→wheel 链路；
  不再同时显式指定 `--sdist --wheel` 从工作树分别构建两个制品。

- 发布包 benchmark 校验读取已声明的 `public_copy_sha256`；原始 `source_files`
  哈希保留不变。声明了脱敏就必须完整匹配公开副本，缺失、篡改或偷偷退回原始副本均拒绝。
  CI 额外保留完整 bundle 校验日志，失败时也上传。

没有增加运行依赖、估计器、公共参数、结果字段或错误码；没有放松原有数值断言。
API/错误/Skill 的运行命令、配置默认值和模块依赖边界维持不变。

## CI 与依赖下限

原 3 OS × Python 3.10–3.13 的 12 格最新兼容依赖矩阵保留，增加 Ubuntu 上每个
Python 版本的 4 格最低可安装运行依赖组合。每格均运行接口门禁、全量测试、
隔离 sdist→wheel 构建、新 venv 安装与六项 smoke。最低组合也约束 clean-install，
不能偷偷升级依赖后当作最低版本验收。

| Python | NumPy | SciPy | pandas | Numba |
|---|---|---|---|---|
| 3.10 / 3.11 | 1.26.0 | 1.11.1 | 2.0.0 | 0.59.0 |
| 3.12 | 1.26.0 | 1.11.2 | 2.1.1 | 0.59.0 |
| 3.13 | 2.1.0 | 1.14.1 | 2.2.3 | 0.61.0 |

共同下限：joblib 1.3.0、threadpoolctl 3.2.0。测试/构建工具不在此下限认证范围内。
SciPy 1.11.0 已被维护者撤回；较新 Python 使用有相应二进制发行的依赖版本。
实际成功之前，这些是待执行的验收组合，不是已认证的平台。

CI 保留 source commit、代码指纹、Python/依赖版本、JUnit、测试/构建/安装日志、
wheel/sdist。`validation-complete.txt` 只在上述步骤全部成功后产生；失败作业也上传已有证据。
16 格全部通过才允许 candidate bundle 作业开始；仍不自动发布。

## 证据规则与尚待完成

旧 macOS 本地测试、旧 Linux 执行记录和旧 CI 是历史基线证据，不能替修复后的源码背书。
修改受指纹覆盖的代码/测试/CI 后，旧 `passed` 记录必须失效；本轮重新记录验证。
候选 manifest 可保留未完成项，正式门禁不允许把它们当通过。

本地修复曾以 `NUMBA_THREADING_LAYER=workqueue` 复现原崩溃（子进程退出码 134），
修复后原加权/非加权并行 bootstrap 测试均通过；完整的最终计数见交付证据。
无网络且缺少完整 wheelhouse 的宿主安装测试不算隔离构建或新依赖解析。

尚需以最终选定提交收齐 16 格 CI 和干净安装的原始证据，生成外置 manifest，绑定
wheel、sdist、source ZIP、bundle 的 SHA256，再执行正式 `--mode release` 门禁。
外部 Stata/golden parity 不新增认证主张；保持 alpha 与外部验证边界。

参考：
- Numba threading layers: https://numba.readthedocs.io/en/stable/user/threading-layer.html
- Numba version support: https://numba.readthedocs.io/en/0.61.0/user/installing.html
- PyPA build: https://build.pypa.io/en/stable/
- SciPy withdrawn release: https://pypi.org/project/scipy/1.11.0/
- SciPy Python 3.12 wheels: https://pypi.org/project/scipy/1.11.2/
- pandas Python 3.12 wheels: https://pypi.org/project/pandas/2.1.1/
