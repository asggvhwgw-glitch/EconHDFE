# Optional R interface for BoundedQR

此目录独立保留旧的 R 使用入口。BoundedQR 主体在 `contrib/boundedqr`，计算内核是 Python；不使用 R 的用户无需安装本目录。

从 EconHDFE 仓库根目录运行：

```bash
python -m pip install ./contrib/boundedqr
Rscript -e 'install.packages(c("quantreg", "jsonlite"), repos="https://cloud.r-project.org")'
Rscript compat/boundedqr-r/clustered_quantiles.R contrib/boundedqr /absolute/path/to/venv/python
```

最后一个参数须指向已安装 NumPy、SciPy、threadpoolctl 的 Python 可执行文件。Windows 可用 `C:/.../Scripts/python.exe`。示例从自己的目录加载 `boundedqr.R`，因此无需把 R 文件放回主体目录。

直接调用时，`source("compat/boundedqr-r/boundedqr.R")`，再调用 `boundedqr_cluster(..., project_root="contrib/boundedqr", python=...)`。`project_root` 指向 Python 包源码目录，而非本兼容目录。Python 子进程执行计算，R 负责 quantreg 基准残差及抽样约定。完整输出保留所有抽样和无限误差界；请检查诊断。

示例固定 9 组乘数，核对 `quantreg::boot.rq` 与 Python 抽样结果。R 与 Python 的随机数流不同，固定种子本身不保证跨语言逐次相同。此桥接保留 R 的聚类顺序、基准残差符号与乘数；它不意味着所有 ACS 设计已获得有限误差界，见 [真实数据验证](../../contrib/boundedqr/VALIDATION.md)。

本目录采用 GPL-3.0-or-later，见 [LICENSE](LICENSE)。R、quantreg 和 jsonlite 分别按其许可证安装。主 Python 包及其发行归档不含这些 R 源码。
