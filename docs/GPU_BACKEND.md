# GPU对接与实施指南

**状态（2026-11-22）：GPU 混合后端已实现并在 RTX 4060 实测。实测数据见 [GPU 本机实验](GPU_LOCAL_RESULTS.md)，执行步骤见 [复现指南](GPU_RUNBOOK.md)。以下保留原设计与验收要求。**

## 1. 先检测硬件，不盲装CUDA

新电脑先运行：

```powershell
.\.venv\Scripts\python.exe doctor.py
nvidia-smi
```

建议先走NVIDIA CUDA + CuPy路线。没有兼容NVIDIA设备时，不把CPU模拟结果写成GPU结果；Intel/AMD显卡需要单独设计支持路线。若使用云GPU，取得用户对服务/费用的授权后再选择，不能默认开付费实例。

记录GPU型号、显存、驱动、OS、Python、CuPy及运行时版本。`nvidia-smi`中的CUDA Version表示驱动支持能力，不等于已安装Toolkit版本；若有nvcc再记录`nvcc --version`，并以CuPy实际配置验证。

## 2. 环境选择

CPU环境保持可用，先安装requirements-gpu.txt（只包含基础依赖），再由接手Codex根据真实硬件和驱动选择**一个**CuPy发行包。以下是官方文档当前示例，不是无需检测即可照抄的通用安装命令：

```powershell
# CUDA 12兼容路线，且该GPU/驱动满足当前CuPy要求时：
.\.venv\Scripts\python.exe -m pip install "cupy-cuda12x[ctk]"

# CUDA 13兼容路线的替代选项，不与上面同时安装：
# .\.venv\Scripts\python.exe -m pip install "cupy-cuda13x[ctk]"
```

新版驱动/Toolkit可能停止支持旧GPU，不能仅看数字更大就升级。重新核对官方安装页和设备兼容性，固定最终实测版本到新的GPU环境锁文件。不要把CUDA11/12/13的多个CuPy包混装。`[ctk]`按当前文档提供所需CUDA组件，但仍需兼容系统驱动。

## 3. 现有设备探针

```powershell
.\.venv\Scripts\python.exe diagnostics/gpu_probe.py --output results/local/gpu_probe.json
```

探针做一次真实GPU数组运算并同步、回传核对；找不到库/设备时非零退出并写unavailable。它不是TF-IDF实现，也不是正式性能测试。

## 4. 接手Codex需实现的方案

已实现的独立 GPU 入口：

```text
python gpu_main.py --input data/imdb.jsonl --limit 90000 --top-k 10 --output results/local/gpu_keywords.jsonl
```

建议先增加独立GPU入口或延迟导入的可选后端，CPU程序不应因缺少CuPy而无法运行。

### 推荐第一版：CPU文本处理 + GPU稀疏数值阶段

```text
原始文本
  → CPU分词和词频
  → 构建CSR（每篇每词最多一个非零项）
  → 主机到GPU传输
  → GPU全局DF / IDF / 权重 / 全词L2归一化
  → 回传CPU，按原文档顺序选择/输出关键词
```

这是混合CPU/GPU流水线，必须如实标注。不要声称英文正则分词已经在GPU执行。

- CSR包含indptr、indices、data；词表与文档顺序确定且可复现。
- 若CSR每行每词仅一次，DF可以对indices按词编号计数；若同词重复未合并，不能把原始token出现次数当DF。
- `IDF=log((N+1)/(DF+1))+1`；raw TF乘IDF，再使用每行全词平方和的平方根归一化。
- 使用float64先对齐当前CPU权重容差；float32仅作明确标注的额外精度/性能实验。
- top-k必须使用完整向量范数，不能截断后重新归一化。相同权重按词字母顺序。
- 先采用CPU端确定性top-k也可以，但其耗时与回传必须计入整体时间。
- 不能分配`N×V`稠密矩阵：90000篇×约十几万词远超普通显存。采用稀疏表示，必要时分批，记录显存预算、批大小及中间数组开销。
- CPU分词/CSR构建尽量复用现有计数规则；CPU参考和GPU后端共享输入合同，而不是偷偷换成另一套tokenizer。

## 5. 必须分开的两个实验

### A. 全流程对比（主要结论）

从相同内存原文开始，到完整输出字典/关键词就绪为止；包含CPU分词、CSR构造、H2D、GPU计算、D2H和top-k/重建。CPU串行、多进程与GPU在**同一台GPU电脑**重新测量。旧笔记本结果只作历史背景，不能直接当纯GPU加速分母。

### B. 数值阶段对比（辅助解释）

准备相同CSR后，比较CPU NumPy/SciPy与GPU CuPy数值阶段。把纯设备内核、传输和端到端数值处理时间分开。若CSR准备或传输被排除，必须对CPU/GPU采用一致边界并标注。GPU上数值内核快不等于整个文本项目快。

## 6. 同步与预热

CuPy调用通常异步。仅在调用前后执行perf_counter而不等待GPU完成会严重低估耗时。

```python
cp.cuda.Stream.null.synchronize()
start = time.perf_counter()
# H2D、GPU计算、D2H、结果恢复等需要计入的工作
cp.cuda.Stream.null.synchronize()
elapsed = time.perf_counter() - start
```

若使用非默认stream，等待真正参与计算的stream，或用设备同步；不要同步错stream。设备内核时间采用CUDA events或cupyx.profiler.benchmark，作为单列辅助指标。先完成实际内核的预热再做10次正式测量，同时另报首次CUDA上下文初始化/JIT冷启动开销。各模式的冷/热计时条件明确区分，不能与CPU含启动历史值混为一谈。

## 7. 验收

- 探针证明真实GPU执行，并记录设备信息。
- sample、边界输入及100/1000篇真实文档的完整向量与CPU核对，float64参考容差rtol=1e-10、atol=1e-12；若达不到先分析误差，不随意放宽到掩盖错误。
- 独立检查文档顺序、词表映射、空行、重复词合并、global DF、top-k排序。
- 每篇完整向量L2范数为1（非空），截断top-k的范数可以小于1。
- 同机器CPU基线、相同SHA256输入、相同输出任务、原始10次时间、显存/批次配置。
- 记录失败、显存不足、无设备等状态；不得把CPU fallback写作GPU成功。

## 官方参考
- https://docs.cupy.dev/en/stable/install.html
- https://docs.cupy.dev/en/stable/user_guide/performance.html
- https://docs.cupy.dev/en/stable/reference/scipy_sparse.html
- https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/

