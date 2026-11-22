# 四模式 CPU GPU TF-IDF 实测与复现

## 概念与实现

`cpu-single` 和 `cpu-multi` 分别用一个与四个 CPU 进程。`gpu-single` 用一个进程调用一张 RTX 4060 的 CUDA 并行数值运算；`gpu-multi` 用四个进程共享**同一张** RTX 4060，各进程有独立 CUDA 上下文并处理不同批次。后者不代表四张 GPU，也不代表单进程 CUDA 是串行算术。两种 GPU 模式的分词、CSR 构建、关键词结果恢复均在 CPU；数值计算在 GPU。GPU 多进程的全局 DF/IDF 在主进程 GPU 上完成，再向四个子进程传递 IDF。

四模式均使用相同 IMDb JSONL 输入、原顺序嵌套子集、`[a-z]+` 词法、原始 TF、平滑 IDF、完整向量 L2 归一化及 top-10 排序。`data/imdb.jsonl` 的 SHA-256 为 `1a36bece88d473792b3fc947a14d155685c47a02d38b18c58ee77ffc134f1263`。每组测量前逐篇比对四种模式的关键词权重和全部 IDF，容差为 `rtol=1e-10, atol=1e-12`；全部通过。四个 CUDA 子进程实际参与每轮 10k/50k/90k 测量。

## 同机全流程结果

机器：Windows 11、Python 3.13.1、20 逻辑 CPU、RTX 4060 8 GB、CuPy 14.2.0、`float64`、batch 500。计时从相同内存文本到关键词字典就绪；包含 CPU 分词、CSR、进程启动和通信、GPU 传输/同步、CPU 恢复，排除 JSONL 文件读写。每组先预热一次，轮次内打乱模式顺序。表中为中位数，单位秒；10k 有 10 次正式测量，50k/90k 各 5 次。

|IMDb 篇数|CPU 单进程|CPU 4 进程|GPU 单进程|GPU 4 进程|
|---:|---:|---:|---:|---:|
|10,000|0.8376|1.0580|1.0486|2.7314|
|50,000|4.2768|2.0888|5.5785|7.1212|
|90,000|7.7890|3.1536|9.9525|11.7226|

9 万篇时 CPU 4 进程相对 CPU 单进程加速 **2.47×**，进程效率 `2.47/4=61.7%`；GPU 单进程相对 CPU 单进程为 **0.78×**，GPU 4 进程为 **0.66×**。GPU 4 进程比 GPU 单进程耗时约 **1.18×**。四个进程争用同一张卡，额外 CUDA 上下文、启动和进程通信没有带来加速。这是本实现、本数据、本机的结果；不能推广成“GPU 一般比 CPU 慢”。GPU 多进程不按 CPU 的 `S/p` 公式计算 GPU 核心效率。

`results/local/four_mode_imdb_{10000,50000,90000}/raw.csv` 保存每轮实际值与进程计数；各自 `summary.json` 保存中位数、均值、标准差、Q1/Q3/IQR、加速比和环境。单次 CLI 输出见 `results/local/demo_four_*.txt`；单次时间可能包含首次 CUDA 上下文启动，不能替换正式中位数。图表 `GPU_CPU_四模式折线图.png` 从上述正式 CSV/JSON 绘制，误差线为 IQR。原有三模式与纯数值阶段实验仍保留在 `results/local/gpu_*`，见 `GPU_LOCAL_RESULTS.md`。

## 如何运行与录制演示

完整 PowerShell 命令与三分钟视频顺序见 [FOUR_MODE_DEMO.md](FOUR_MODE_DEMO.md)。最快的课堂演示方式是在准备好 `.venv` 和数据后，用 `four_mode_main.py` 对五篇 `data/sample.jsonl` 依次运行四个模式；随后打开真实的 `raw.csv`、`summary.json` 与折线图讲解 9 万篇的重复实验。短演示的单次秒数会随启动和系统负载波动，以保存的正式测量为准。视频提交由小组在课程网站操作。
