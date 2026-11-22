# GPU TF-IDF 本机实测结果（2026-11-22）

## 实验范围与机器

基于仓库 `Dundun123123/435` 的源码快照 `6549c63c1f5cabca8b3fae7a749f7bbe2ad0ff0e`，新增 `gpu_tfidf.py` 等 GPU 后端与实验脚本。源码由 GitHub ZIP 获取，因此本地没有 `.git` 元数据；新增内容尚未推送到远端。原有 CPU 代码与历史结果未覆盖。MPI 双机实验仍未执行。

本机为 Windows 11、Python 3.13.1、Intel64 Family 6 Model 183 Stepping 1（20 逻辑处理器）、约 31.84 GiB 内存、NVIDIA GeForce RTX 4060（约 8 GiB 显存）、驱动 610.88。CuPy 14.2.0、CUDA Runtime 13.2（`runtimeGetVersion=13020`）、NumPy 2.4.6。GPU 数值计算使用 `float64`；系统未安装 CUDA Toolkit，运行所需 CUDA 组件由项目环境中的 PyPI 包提供。

这是 **CPU 分词/词频与 CSR 构建 + GPU 稀疏数值计算 + CPU 关键词恢复** 的混合实现。GPU 执行 DF、平滑 IDF、TF-IDF 权重和完整向量 L2 归一化；每批 500 篇，没有构造文档数乘词表大小的稠密矩阵。

数据使用仓库脚本生成：20 Newsgroups 有效 18,311 篇，`data/newsgroups.jsonl` SHA-256 为 `79229d2a00fe99bfd689c90dc1d6177a4f9457af041b2ea98a2fc8d606c56db6`；IMDb 清理去重后 98,463 篇，`data/imdb.jsonl` SHA-256 为 `1a36bece88d473792b3fc947a14d155685c47a02d38b18c58ee77ffc134f1263`。实验使用原有顺序的嵌套子集；IMDb 90,000 篇包含 21,238,059 个匹配 token、132,229 个词、12,571,163 个 CSR 非零项。

## 正确性

- CPU 原有测试和新增 GPU 契约测试共 6 项均通过；`verify.py` 的完整串/并行向量参考核对与 `validate_large.py` 的 98,463 篇 top-10 核对也通过。
- 20 Newsgroups 与 IMDb 各取 100/1,000 篇，比对 CPU 与 GPU 的完整稀疏向量、全部 IDF、L2 范数，容差 `rtol=1e-10, atol=1e-12`。IMDb 最大绝对权重误差 `7.22e-16`，IDF 最大误差 0。
- 五组重复实验前逐文核对同机 CPU 串行与 GPU 的 top-10 词及权重；9 万篇输出已生成 90,000 行 JSONL。
- 设备探针实际在 GPU 上做数组运算、同步并回传；设备识别为 RTX 4060。

## 同机全流程时间

每组 1 次预热、10 次正式测量。以下均为中位数，单位秒。输入为相同的内存文本，输出为同样的 top-10 关键词字典。计入分词、CSR 构建、数据传输、CUDA 计算、同步、回传和结果恢复；不计 JSONL 读取与写盘。4 进程 CPU 是仓库现有实现。每轮执行顺序由固定随机种子打乱。

|数据与篇数|CPU 串行|CPU 4 进程|GPU 混合|CPU 串行 / GPU|
|---|---:|---:|---:|---:|
|Newsgroups 1,000|0.0558|0.4284|0.0691|0.81×|
|Newsgroups 18,311|1.2434|1.2681|1.5462|0.80×|
|IMDb 10,000|0.8274|1.0449|1.0847|0.76×|
|IMDb 50,000|4.1786|2.0473|5.4134|0.77×|
|IMDb 90,000|7.7240|3.1380|9.8579|0.78×|

9 万篇时，CPU 4 进程相对 CPU 串行为 2.46×。这台机器上的 GPU 混合流程比 CPU 串行慢约 27.6%，更慢于 CPU 4 进程。没有观察到 GPU 全流程加速。完整原始数据见各 `results/local/gpu_*/raw.csv`、`summary.json` 与 `time_comparison.svg`。

## 相同 CSR 上的数值阶段

`gpu_benchmark.py` 中的同 CSR 对照还包含关键词字典恢复；IMDb 90,000 篇时，CPU 为 3.0190 秒，GPU 为 3.2781 秒。`gpu_numeric_benchmark.py` 进一步把关键词恢复从计时路径移除：CPU NumPy 从 DF 到完整向量归一化为 **0.1470 秒**；GPU CuPy 包含 H2D、设备计算、同步、D2H 为 **0.2679 秒**。这两项均排除分词、CSR 构建、关键词字典恢复。GPU CUDA event 测得设备区间中位数约 0.1642 秒，仅用于拆解，不能代替含传输的时间。两种数值计算同样 1 次预热、10 次正式测量。

在完整 GPU 流程的 9 万篇重复实验中，CPU 分词与 CSR 阶段中位数为 6.6012 秒，结果恢复中位数为 2.8988 秒；设备 event 区间中位数约 0.2487 秒。各阶段中位数来自不同轮次，不能直接相加当成总时间。CPU 前后处理占主要部分，同时当前稀疏分批算法的传输和 GPU 分配开销也抵消了设备计算收益。

## 复现与限制

见 [GPU_RUNBOOK.md](GPU_RUNBOOK.md)。`results/local/` 保留 CSV、JSON 与 SVG，原始 IMDb 与 Newsgroups 正文及下载缓存不打包。GPU 关键词 JSONL 可以单独压缩交付。所有速度结论仅适用于此硬件、软件版本、语料、批量与 top-10 任务；旧笔记本 CPU 历史数据没有作为 GPU 加速分母。没有测试 GPU 分词、多卡、双机 MPI 或其他 GPU 算法。
