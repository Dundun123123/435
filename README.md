# CST435W：TF-IDF 串行、单机多进程与后续 MPI/GPU 实验

本仓库用于小组作业，以及在第二台电脑上由 Codex 接手继续开发。**请先读 [项目交接](docs/HANDOFF.md)。**

## 当前状态

|部分|状态|
|---|---|
|Python 串行 / multiprocessing|已实现，并完成正确性核对|
|真实文本数据准备|已实现；大语料不存入Git，脚本可重新获取|
|CPU重复实验|已完成，250次正式测量；见 [研究结果](results/study_v3/REPORT.md)|
|双机MPI|已提供设计、环境与验收指南、连通性探针；**TF-IDF MPI后端尚未实现、未实测**|
|GPU|已提供设计、计时规则与验收指南、设备探针；**TF-IDF GPU后端尚未实现、未实测**|

不要把本机MPI多进程写成双机实验，不要把GPU数值阶段的耗时当作整个文本处理耗时。

## 1. Windows 新电脑安装与首次运行

需要 Git 和64位Python 3.13。PowerShell中逐条执行：

```powershell
git clone https://github.com/Dundun123123/435.git
cd 435
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe doctor.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe main.py --input data/sample.jsonl --mode sequential --output results/local/sample_seq.jsonl
.\.venv\Scripts\python.exe main.py --input data/sample.jsonl --mode parallel --workers 2 --batch-size 2 --output results/local/sample_par.jsonl
```

不需要激活虚拟环境，因此不依赖PowerShell激活脚本策略。Linux/macOS使用 `.venv/bin/python` 替代 `.venv\Scripts\python.exe`。

## 2. 获取真实数据

```powershell
.\.venv\Scripts\python.exe prepare_data.py
.\.venv\Scripts\python.exe prepare_large_data.py
.\.venv\Scripts\python.exe verify.py
.\.venv\Scripts\python.exe validate_large.py
```

第一次需要联网。20 Newsgroups有效18311篇；IMDb清理去重后98463篇。来源、SHA256和预处理见 [数据说明](data/README.md)。目录中只有示例与清单，没有完整语料，所以仅clone不能直接运行大数据命令。

## 3. 正式运行与性能实验

```powershell
.\.venv\Scripts\python.exe main.py --input data/imdb.jsonl --limit 90000 --mode sequential --output results/local/imdb_seq.jsonl
.\.venv\Scripts\python.exe main.py --input data/imdb.jsonl --limit 90000 --mode parallel --workers 4 --output results/local/imdb_par.jsonl
.\.venv\Scripts\python.exe benchmark.py --input data/imdb.jsonl --sizes 10000 50000 90000 --workers 1 2 4 8 --repeats 10 --warmups 1 --top-k 10 --output-dir results/local/scale
```

`python study.py --output-dir results/local/study` 可重做规模和长短文档整套实验，需要较长时间。公开的已有结果保留不覆盖。完整说明见 [CPU指南](docs/GUIDE_V3.md)。

## 4. 第二台电脑的接手顺序

1. 阅读 [HANDOFF.md](docs/HANDOFF.md) 和根目录 [AGENTS.md](AGENTS.md)。
2. 按本README跑通CPU示例和测试。
3. 运行 `doctor.py`，确认操作系统、MPI环境及NVIDIA GPU情况。
4. 双机任务按 [MPI对接指南](docs/MPI_TWO_NODES.md)；GPU任务按 [GPU对接指南](docs/GPU_BACKEND.md)。
5. 使用 [可以直接复制给Codex的提示词](docs/CODEX_PROMPTS.md)。
6. 使用 [实验协议](docs/EXPERIMENT_PROTOCOL.md) 验收，遵守 [Git协作说明](docs/GIT_WORKFLOW.md)。

## 程序结构

```text
tfidf.py                 共用词频/IDF/归一化，以及串行和多进程调度
main.py                  现有CLI；目前只支持sequential/parallel
benchmark.py             预热、10次重复、原始数据与误差图
study.py                 按顺序运行整套CPU实验
prepare_data.py          20 Newsgroups
prepare_large_data.py    IMDb + 长短文档样本
verify.py                参考库核对及边界情况
validate_large.py        全量IMDb关键词核对
doctor.py                不安装、不修改系统的本机诊断
diagnostics/             MPI/GPU探针（不是TF-IDF后端）
docs/                    接手、MPI、GPU、实验和Git文档
tests/                   无需大语料的CPU接口测试
results/                 已公开的CPU实测结果；新测量放results/local/
```

## 算法合同 / Algorithm contract

英文单词正则 `[a-z]+`，小写，不做停用词删除或词干化。TF为出现次数；`IDF=ln((N+1)/(DF+1))+1`；全词L2归一化。top-k的权重仍以完整向量范数计算。TF-IDF是NLP特征提取，不是分类模型训练。

The CPU implementation is complete. The two-node MPI and GPU TF-IDF backends are planned extensions, not completed experiments. All backends must share the same corpus, preprocessing, mathematical definitions, output task and correctness checks. See the handoff documents before extending the project.

