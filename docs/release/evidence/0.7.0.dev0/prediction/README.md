# 0.7.0.dev0 本机预测验证

以 validation.json 的代码指纹、时间及结果为准。构建/安装复用 HOST 依赖，不是隔离构建或跨平台证明。
原始日志包含测试路径和公开模拟测试的例行 warnings；不含用户研究数据。
benchmark_prediction.py 可在源码根目录用 PYTHONPATH=. 运行；只比较 xb 分块，不声明峰值 RSS 或估计器提速。
