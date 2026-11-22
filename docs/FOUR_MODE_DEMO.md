# 四种 TF-IDF 模式的 CLI 运行与三分钟演示

## 模式含义

- `cpu-single`：一个 CPU 进程完成分词、DF/IDF、权重和关键词。
- `cpu-multi`：多个 CPU 工作进程按现有 v3 方案协作。
- `gpu-single`：一个进程使用一张 GPU；CPU 分词/CSR，GPU 数值计算，CPU 恢复关键词。
- `gpu-multi`：多个独立进程共享**同一张** GPU，每个进程拥有 CUDA 上下文并处理不同 CSR 批次；全局 DF/IDF 先在主进程 GPU 计算，之后传给子进程。此模式不是多卡，也不是“GPU 核心串行/并行”的切换。GPU 单进程中的 CUDA 操作本身已并行。

四个模式使用相同 JSONL 输入、`[a-z]+` 分词、平滑 IDF、完整向量 L2 范数、top-10 排序和输出格式。计时从内存原文到关键词字典；不包含文件读写。GPU 多进程包括进程启动、通信、传输和结果恢复，因此不能用 CUDA event 时间替代总时间。

## 在 Windows PowerShell 中运行

在项目根目录执行。需要 Python 3.13、兼容的 NVIDIA GPU 与驱动。第一次在可联网环境安装依赖并准备数据：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu-local.txt
.\.venv\Scripts\python.exe prepare_data.py
.\.venv\Scripts\python.exe prepare_large_data.py
New-Item -ItemType Directory -Force .tmp, .cupy, results/local | Out-Null
$env:TEMP = (Resolve-Path .tmp).Path
$env:TMP = $env:TEMP
$env:CUPY_CACHE_DIR = (Resolve-Path .cupy).Path
nvidia-smi
.\.venv\Scripts\python.exe diagnostics/gpu_probe.py --output results/local/gpu_probe.json
```

先用仓库自带的五篇小样例检查四种输出：

```powershell
foreach ($mode in 'cpu-single','cpu-multi','gpu-single','gpu-multi') {
  .\.venv\Scripts\python.exe four_mode_main.py --mode $mode --input data/sample.jsonl --workers 2 --batch-size 2 --top-k 3 --output "results/local/demo_$mode.jsonl"
}
```

再用同一份 IMDb 子集跑性能实验。9 万篇会花较长时间；课堂三分钟视频建议演示 1,000 或 10,000 篇，并展示预先保存的 9 万篇结果与原始 CSV。

```powershell
.\.venv\Scripts\python.exe four_mode_benchmark.py --input data/imdb.jsonl --limit 10000 --workers 4 --batch-size 500 --warmups 1 --repeats 10 --output-dir results/local/four_mode_imdb_10000
Get-Content results/local/four_mode_imdb_10000/summary.json
```

`summary.json` 提供四组中位数、IQR、CPU 串行基准的加速比与正确性状态，`raw.csv` 保留每次正式/预热测量。`gpu-multi` 的 `active_gpu_processes` 和进程 ID 存在每轮 profile 中；在单卡上多进程未必能加速，必须以实际测量为准。进程效率 `S_p/p` 仅用于 CPU 多进程的 CPU 核/进程分析；GPU 多进程应报告吞吐/耗时及与单进程 GPU 的比值，不把 GPU SM 数当 CPU 工作进程数。

## 三分钟视频顺序

1. 约 20 秒：展示 JSONL 输入一行与四种模式定义。
2. 约 30 秒：运行 `nvidia-smi` 与 GPU 探针，说明只有一张 RTX 4060。
3. 约 60 秒：运行四模式小样例，展示相同关键词与 GPU 多进程实际进程数。
4. 约 50 秒：展示真实 `raw.csv`、`summary.json` 和折线图，讲明 CPU 串行基线、加速比及未加速的 GPU 结果。
5. 约 20 秒：总结原因：CPU 分词/结果恢复、CUDA 上下文启动、同卡多进程竞争。

录屏只录程序、数据和报告；关闭其他私人窗口。视频与课程网站提交需要由小组自行完成。
