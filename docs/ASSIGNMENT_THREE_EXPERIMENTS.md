# CST435W 三组实验复现与三分钟视频指南

本指南对应交付源码 `435_GPU_源码与实验结果.zip` 和 Word 报告。所有操作都在 **PowerShell CLI** 中完成。先用小样例验证程序，再看已经保存的正式重复实验；三分钟视频不需要现场重跑 90,000 篇。

## 1. 先理解要展示的三个实验

|实验|比较对象|90,000 篇 IMDb 中位时间|
|---|---|---|
|1 单机 CPU|串行 1 个进程 vs CPU 并行 4 个进程|7.7890 s vs 3.1536 s|
|2 单进程 GPU 加速|同机 CPU 串行 vs 1 个 Python 进程调用 GPU|7.7890 s vs 9.9525 s|
|3 单机与双机 CPU|同一 socket 多进程程序：单机 4 个工作进程 vs 两台电脑各 2 个|2.6034 s vs 4.7670 s|

实验 1 的加速比 `7.7890/3.1536 = 2.47×`，4 进程效率 `2.47/4 = 61.7%`。实验 2 的全流程加速比 `7.7890/9.9525 = 0.78×`，这次没有实现 GPU 全流程加速。实验 3 的双机相对单机比值 `2.6034/4.7670 = 0.55×`，双机更慢。三个实验的 10,000、50,000、90,000 篇时间折线图见 `assignment_three_experiments.png`。

**“GPU 单进程”指一个 Python 主进程，不代表 GPU 芯片串行计算。**程序在 CPU 分词和恢复关键词，在 RTX 4060 上用并行 CUDA 内核做稀疏 TF-IDF 数值计算。实验 3 的单机基线必须是同一 socket 程序的单机运行；不能拿实验 1 的 `cpu-multi` 时间充当实验 3 的分母，因为实现和计时边界不同。

实验 3 已在两台真实 Windows 电脑上通过小样例及正式重复实验。电脑 A：`Computer-A`，Intel Core i5-14600KF，20 逻辑处理器，Python 3.13.1；电脑 B：`Computer-B`，Intel Core i5-11300H，4 物理核，Python 3.13.12。两台通过同一 `192.168.1.x` WLAN 通信。电脑 A 的真实输出记录两台主机名各出现两次，证明每台参与 2 个 CPU 工作进程。两机 CPU 性能不同，网络传输也计入双机时间，因此不能单独把差值归因于网络。

## 2. 项目与输入准备，只需做一次

电脑 A、B 均可从 GitHub 克隆本仓库；以下示例假设 A 已进入克隆目录，B 在 `C:\CST435\435-GPU`。视频中可以使用现有目录，不必重新安装。若在新电脑重新搭建，把源码包复制或下载到各电脑，解压到自己选择的目录后进入项目根目录，确保 `dir four_mode_main.py`、`dir socket_tfidf_benchmark.py` 能看到文件。

电脑 A 首次安装：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu-local.txt
.\.venv\Scripts\python.exe -m pip install matplotlib==3.11.2
.\.venv\Scripts\python.exe prepare_large_data.py
New-Item -ItemType Directory -Force .tmp, .cupy, .mpl, results\local | Out-Null
$env:TEMP = (Resolve-Path .tmp).Path
$env:TMP = $env:TEMP
$env:CUPY_CACHE_DIR = (Resolve-Path .cupy).Path
$env:MPLCONFIGDIR = (Resolve-Path .mpl).Path
nvidia-smi
.\.venv\Scripts\python.exe diagnostics\gpu_probe.py --output results\local\gpu_probe.json
```

电脑 B 只运行 CPU 工作进程，使用 Python 标准库，无需 `mpi4py` 或 GPU 包。进入 `C:\CST435\435-GPU` 后运行 `python -m venv .venv`。本次两机曾尝试 MS-MPI，但远程启动返回 Windows `Access denied (errno 5)`；正式双机实验采用同一 Python 多进程工作函数加经过一次性连接码认证的 TCP 通道，不把本机 MPI 结果冒充双机结果。

输入为 UTF-8 JSONL，每行至少有唯一 `id` 和字符串 `text`。正式输入 `data\imdb.jsonl` 的 SHA-256 应为 `1a36bece88d473792b3fc947a14d155685c47a02d38b18c58ee77ffc134f1263`。电脑 A 可以执行：

```powershell
Get-Content data\sample.jsonl -TotalCount 1
Get-FileHash data\imdb.jsonl -Algorithm SHA256
```

所有模式使用 `[a-z]+` 小写分词、原始词频、`IDF = ln((N+1)/(DF+1))+1`、完整向量 L2 归一化、前 10 个关键词。输出 JSONL 保持输入顺序；每轮正式计时保存到 `raw.csv`，中位数和正确性结果保存到 `summary.json`。**计时从已经读入内存的文本开始，到关键词字典完成；不包含读写文件。**GPU 计入传输和同步；实验 3 计入工作进程启动、进程通信、WLAN 传输和结果恢复，但排除最初的 TCP 建连。

## 3. 实验 1 与 2：同机 CPU 和 GPU

在电脑 A 的项目根目录先用五篇小样例看清三种模式的输出：

```powershell
foreach ($mode in 'cpu-single','cpu-multi','gpu-single') {
  .\.venv\Scripts\python.exe four_mode_main.py --mode $mode --input data\sample.jsonl --workers 4 --batch-size 2 --top-k 3 --output "results\local\sample_$mode.jsonl"
}
```

每次会打印 `Mode`、`workers`、`elapsed: ... s`、第一篇关键词和输出路径。小样例的时间只用于现场演示，不能代替下列 IMDb 正式结果。正式运行三种模式，先预热，再轮流重复并逐篇核对输出：

```powershell
.\.venv\Scripts\python.exe four_mode_benchmark.py --input data\imdb.jsonl --limit 10000 --workers 4 --batch-size 500 --top-k 10 --warmups 1 --repeats 10 --modes cpu-single cpu-multi gpu-single --output-dir results\local\four_mode_imdb_10000
.\.venv\Scripts\python.exe four_mode_benchmark.py --input data\imdb.jsonl --limit 50000 --workers 4 --batch-size 500 --top-k 10 --warmups 1 --repeats 5 --modes cpu-single cpu-multi gpu-single --output-dir results\local\four_mode_imdb_50000
.\.venv\Scripts\python.exe four_mode_benchmark.py --input data\imdb.jsonl --limit 90000 --workers 4 --batch-size 500 --top-k 10 --warmups 1 --repeats 5 --modes cpu-single cpu-multi gpu-single --output-dir results\local\four_mode_imdb_90000
```

这三组的完整同机运行已保存于交付源码包的 `results\local\four_mode_imdb_*`；GitHub 仓库另提供去除机器标识后的 `results/assignment_2026/` 原始计时。原测量还包含额外的 GPU 多进程模式，但本报告只从其相同试次提取 `cpu-single`、`cpu-multi`、`gpu-single` 三列。任何重新测量都有正常波动，以新生成的 `summary.json` 为准，不要修改或删除慢试次。

## 4. 实验 3：一台 4 进程对两台各 2 进程

电脑 A、B 都运行 `socket_tfidf.py` 中的同一个 CPU 工作函数；单机和双机只改变四个工作进程的放置方式。**先完成单机，然后完成双机**，每种输入规模配对比较。

### 4.1 单机 4 进程，在电脑 A 运行

```powershell
.\.venv\Scripts\python.exe socket_tfidf_benchmark.py --mode one-host --input data\imdb.jsonl --limit 10000 --warmups 1 --repeats 10 --output-dir results\local\socket_one_host_10000
.\.venv\Scripts\python.exe socket_tfidf_benchmark.py --mode one-host --input data\imdb.jsonl --limit 50000 --warmups 1 --repeats 5 --output-dir results\local\socket_one_host_50000
.\.venv\Scripts\python.exe socket_tfidf_benchmark.py --mode one-host --input data\imdb.jsonl --limit 90000 --warmups 1 --repeats 5 --output-dir results\local\socket_one_host_90000
```

### 4.2 双机各 2 进程，两台分别运行

两台需在同一可互通 WLAN；以下地址仅为示例，运行前用各电脑的 ipconfig 查到的实际 IPv4 地址替换命令中的地址。示例 A=`192.168.1.10`、B=`192.168.1.11`，地址变化时同步替换命令。电脑 A 在项目根目录生成一次随机连接码：

```powershell
New-Item -ItemType Directory -Force results\local | Out-Null
$key = .\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_hex(16))"
Set-Content results\local\socket_key.txt $key
$key
```

把最后一行显示的 **32 位十六进制连接码**只在两台自己的 PowerShell 间复制，不必发到聊天或拍进视频。电脑 B 在 `C:\CST435\435-GPU` 执行 `Set-Content results\local\socket_key.txt '你刚复制的32位连接码'`。两台的文件内容须相同。实验程序还会检查 B 的局域网 IP 和主机名，拒绝把同一台电脑伪装成双机。

对每个规模按下列顺序操作：**先在电脑 A 运行一条接收端命令，看到 `Waiting for laptop B`；再在电脑 B 运行同一条工作端命令；等双方结束后再做下一个规模。**电脑 B 的工作端命令每次都相同：

```powershell
cd C:\CST435\435-GPU
.\.venv\Scripts\python.exe socket_tfidf_worker.py --server 192.168.1.10 --key-file results\local\socket_key.txt
```

电脑 A 的三条正式接收端命令，**不要一起启动**：

```powershell
.\.venv\Scripts\python.exe socket_tfidf_benchmark.py --mode two-host --listen 192.168.1.10 --expected-peer 192.168.1.11 --key-file results\local\socket_key.txt --input data\imdb.jsonl --limit 10000 --warmups 1 --repeats 10 --output-dir results\local\socket_two_host_10000
.\.venv\Scripts\python.exe socket_tfidf_benchmark.py --mode two-host --listen 192.168.1.10 --expected-peer 192.168.1.11 --key-file results\local\socket_key.txt --input data\imdb.jsonl --limit 50000 --warmups 1 --repeats 5 --output-dir results\local\socket_two_host_50000
.\.venv\Scripts\python.exe socket_tfidf_benchmark.py --mode two-host --listen 192.168.1.10 --expected-peer 192.168.1.11 --key-file results\local\socket_key.txt --input data\imdb.jsonl --limit 90000 --warmups 1 --repeats 5 --output-dir results\local\socket_two_host_90000
```

电脑 A 最终应显示 `Connected: Computer-B (192.168.1.11), 2 remote CPU workers`、每轮 `correct=yes` 和 `Median: ... s`。电脑 B 显示 `Coordinator finished; worker exiting.`。用两个 `summary.json` 核对 `host_count` 分别为 1 和 2、`worker_processes` 均为 4、`dataset_sha256` 相同。电脑 A 的正式数据见交付源码包 `results\local\socket_{one,two}_host_*`；GitHub 仓库的 `results/assignment_2026/` 提供去除机器标识的计时记录。B 不必持有 IMDb 原文。

## 5. 画折线图与核对时间

在电脑 A 的项目根目录运行：

```powershell
.\.venv\Scripts\python.exe plot_assignment_experiments.py --results results\local --output results\local\assignment_experiments.png
Get-Content results\local\four_mode_imdb_90000\summary.json
Get-Content results\local\socket_one_host_90000\summary.json
Get-Content results\local\socket_two_host_90000\summary.json
```

图分为实验 1、2、3 三个折线小图，横轴为 10,000／50,000／90,000 篇，纵轴为正式试次的中位秒数。报告采用已保存的同一输入和正确性已通过的数据。实验 3 的单机和双机是**同一 socket 程序、同样四个工作进程**；实验 1 的 3.154 秒不能代替实验 3 的 2.603 秒。若重新运行，实际时间随负载、CPU 和 WLAN 变化，必须从新 `summary.json` 重画图。

## 6. 三分钟视频按这个顺序录

正式 90,000 篇已提前跑完。录视频时打开两台 PowerShell；录屏可以依次切换窗口，也可用第二台的终端画面作为简短补充。

1. **0:00–0:25** 在 A 展示 `Get-Content data\sample.jsonl -TotalCount 1`，解释输入 `id`、`text`，输出为每篇前 10 个关键词。
2. **0:25–1:10** 在 A 运行第 3 节的三个小样例模式，镜头显示 `elapsed`；运行 `nvidia-smi`，说明 GPU 单进程仍启动并行 CUDA 内核。
3. **1:10–1:50** 在 A 展示第 4 节双机的 **5 篇小样例**：A 用 `socket_tfidf_benchmark.py --mode two-host --input data\sample.jsonl --limit 5 --warmups 1 --repeats 2 --listen 192.168.1.10 --expected-peer 192.168.1.11 --key-file results\local\socket_key.txt --output-dir results\local\video_two_host_sample`；看到 `Waiting` 后在 B 启动工作端命令。指出 A 输出的两个不同主机名和 `correct=yes`。这次小样例时间只作程序演示。
4. **1:50–2:30** 在 A 展示报告中的正式时间表，读出 90,000 篇的三个结论：CPU 7.789 vs 3.154 秒；CPU 7.789 vs GPU 9.953 秒；单机 2.603 vs 双机 4.767 秒。说明这些是预热后的重复实验中位数。
5. **2:30–3:00** 展示 `assignment_experiments.png` 的三个折线图，指出双机慢于单机。展示 `socket_two_host_90000\summary.json` 的 `rank_hosts`、`host_count=2`、`worker_processes=4`、`correctness`。

如需保存当前终端原始文字，可在 A 开始录制前运行 `Start-Transcript -Path results\local\video_transcript.txt`，结束时执行 `Stop-Transcript`。报告中的白底 CLI 图是从保存的真实 CLI 输出生成的**文字图像**；若课程要求原生屏幕截图，请在录视频时直接截取 PowerShell 窗口。不要把 5 篇现场小样例的时间当作 90,000 篇正式时间。
