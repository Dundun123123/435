# GPU TF-IDF 本机实验复现

本扩展使用 CPU 进行 `[a-z]+` 分词与 CSR 构造，使用 NVIDIA GPU 与 CuPy 做全局 DF、IDF、稀疏权重和全词 L2 归一化。输出回到 CPU 后按原文档顺序生成关键词。实验不创建 `N×V` 稠密矩阵；`--batch-size` 控制每次 GPU 处理的文档数。GPU 包是可选依赖，原 `main.py` 的 CPU 命令仍可使用。

## 环境

本机实测：Windows、Python 3.13.1、RTX 4060 8 GB、NVIDIA 驱动 610.88。项目本地环境的精确包版本见 `requirements-gpu-local.txt`。CUDA 组件来自 PyPI，本次没有安装系统级 CUDA Toolkit。请先核对新机器的 GPU、驱动与 CuPy 兼容性。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu-local.txt
New-Item -ItemType Directory -Force .tmp, .cupy | Out-Null
$env:TEMP = (Resolve-Path .tmp).Path
$env:TMP = $env:TEMP
$env:CUPY_CACHE_DIR = (Resolve-Path .cupy).Path
.\.venv\Scripts\python.exe diagnostics/gpu_probe.py --output results/local/gpu_probe.json
```

`TEMP`、`TMP`、`CUPY_CACHE_DIR` 指向项目内目录，是为了让临时编译文件留在可写位置。缺少 GPU 或 CuPy 时，探针与 GPU 命令应报告失败，不使用 CPU 结果冒充 GPU。

## 数据与正确性

```powershell
.\.venv\Scripts\python.exe prepare_data.py
.\.venv\Scripts\python.exe prepare_large_data.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe gpu_verify.py --input data/newsgroups.jsonl --sizes 100 1000
.\.venv\Scripts\python.exe gpu_verify.py --input data/imdb.jsonl --sizes 100 1000 --output results/local/gpu_imdb_correctness.json
.\.venv\Scripts\python.exe gpu_main.py --input data/imdb.jsonl --limit 90000 --top-k 10 --output results/local/gpu_imdb_keywords.jsonl
```

`gpu_verify.py` 检查完整向量、全局 IDF 和非空向量范数，容差为 `rtol=1e-10, atol=1e-12`。边界输入和 top-k 排序由 `tests/test_gpu_contract.py` 检查。`gpu_main.py` 的 `.summary.json` 记录总时间、阶段时间及设备信息。首次调用包含 CUDA 上下文和 JIT 编译，不能与预热后的正式轮次混用。

## 重复实验

```powershell
.\.venv\Scripts\python.exe gpu_benchmark.py --input data/imdb.jsonl --limit 10000 --workers 4 --batch-size 500 --repeats 10 --warmups 1 --output-dir results/local/gpu_imdb_10000
.\.venv\Scripts\python.exe gpu_benchmark.py --input data/imdb.jsonl --limit 50000 --workers 4 --batch-size 500 --repeats 10 --warmups 1 --output-dir results/local/gpu_imdb_50000
.\.venv\Scripts\python.exe gpu_benchmark.py --input data/imdb.jsonl --limit 90000 --workers 4 --batch-size 500 --repeats 10 --warmups 1 --output-dir results/local/gpu_imdb_90000
.\.venv\Scripts\python.exe gpu_numeric_benchmark.py --input data/imdb.jsonl --limit 90000 --batch-size 500 --repeats 10 --warmups 1 --output-dir results/local/gpu_numeric_90000
.\.venv\Scripts\python.exe gpu_plot.py results/local/gpu_imdb_90000
```

每次运行先核对相同 CSR 上的 CPU/GPU 数值结果，再按固定种子打乱五种方案的试验顺序。`raw.csv` 保留全部预热与正式轮次；`summary.json` 给出中位数、均值、样本标准差和 IQR；`time_comparison.svg` 展示中位数与 IQR。

`full_pipeline` 从相同的内存文本开始，到关键词字典就绪为止；包含 CPU 分词、CSR、传输、GPU 运算、回传与重建，但不含文件读取和 JSONL 写盘。`numeric_shared_csr` 从已构造的同一 CSR 开始，两边都包含数值计算与关键词字典重建；GPU 还包含 H2D/D2H。`device_kernel_seconds` 用 CUDA events 测量设备部分，只作辅助解释，不能代替全流程时间。正式测量前先做一轮完整预热。

`gpu_numeric_benchmark.py` 在相同 CSR 上单独计时计算，不生成关键词字典。CPU 计算包含 DF、IDF、权重和归一化；GPU 计算还包含传输、同步和回传。两者都排除 CSR 构建和关键词字典重建。包含字典重建的数值阶段另见 `gpu_benchmark.py`。

本机新测量保留在 `results/local/`，不覆盖仓库历史 CPU 结果；数据正文与下载缓存也不入库。
