# 第二台电脑与Codex项目交接

## 2026-11-22 最新结果

三组最终实验（单机 CPU、CPU/GPU、单机/双机 CPU）已完成，见 [正式复现指南](ASSIGNMENT_THREE_EXPERIMENTS.md)。双机采用 authenticated TCP，不是 MPI；下文保留初始交接时的历史记录。

## 2026-11-22 最新结果

三组最终实验（单机 CPU、CPU/GPU、单机/双机 CPU）已完成，见 [正式复现指南](ASSIGNMENT_THREE_EXPERIMENTS.md)。双机采用 authenticated TCP，不是 MPI；下文保留初始交接时的历史记录。

## 目标与当前边界

用户希望在CST435W小组作业中比较四种方案：
1. CPU串行；2. 单机multiprocessing；3. 两台实体电脑的MPI；4. GPU。

当前只完成1、2；3、4等待硬件后实现和实测。用户此阶段要求的是完整准备与交接，而不是声称已经有双机/GPU结果。后续Codex应根据用户在新电脑上的实际授权继续实现，不需重新从头设计项目。

## 已完成且必须保留

- 共用核心：tfidf.py的count_batch、weight_batch，以及compute_tfidf。
- compute_tfidf(texts, mode='sequential'/'parallel', workers=4, batch_size=500, top_k=None, profile=None) 返回 vectors、idf、seconds 三项。
- top_k=None返回所有稀疏向量；正整数返回top-k，但使用全部词权重的L2范数。
- 输入UTF-8 JSONL，每行唯一字符串id和字符串text，可有额外元数据。
- main.py命令行、关键词输出和相邻summary.json；新的后端不能破坏现有模式。
- 正确性：参考scikit-learn、空文章、重复内容、单字母词、空词表、乱序分配后的恢复；全量IMDb top-10核对通过。
- CPU v3：本地词频保留、全局IDF每进程一次、紧凑数字数组传输、按字符量分配批次。
- benchmark.py：默认10次、1次预热、每轮随机配置顺序、IQR与bootstrap区间、原始数据和父阶段计时。

## 真实历史与结果

原机器CPU：Intel Core i5-11300H，4物理核、8逻辑处理器。Python 3.13.12。其他配置读results中的environment.json，不能把新机器规格写成原机器规格。

|实验|数据/实现|用途|
|---|---|---|
|results/benchmark|初版v1，Newsgroups，5次重复|早期通信开销问题|
|results/optimized_benchmark|v2完整向量，5次重复|保留词频、减少重复分发后的结果|
|results/keyword_benchmark|v2关键词，5次重复|不同输出任务的独立实验|
|results/study_v3|v3，IMDb；250次正式测量|当前主要规模、进程数、长短文本研究|

v3 top-10任务的代表中位数：10000篇串行1.223s/4进程1.261s；50000篇6.203s/3.281s；90000篇11.301s/5.230s（2.16倍），8进程5.061s（2.23倍）。只能说明本次条件，不能承诺其他硬件同样快。

初版与新版同时改变代码和语料，不把全部改善归因于数据量。研究数据规模影响时，主要比较同一v3实现的嵌套子集。

## 新机器第一次接手

1. clone仓库，在项目根目录安装自己的.venv，不复制旧机器虚拟环境。
2. 运行doctor.py（本地报告默认忽略，不把真实主机/IP/密钥提交）。
3. 跑通sample串/并行、unittest；需要大语料时运行两个准备脚本。
4. 记录git rev-parse HEAD、Python/依赖版本、数据SHA256。
5. 先做MPI单机连通性探针，再确认两个不同主机；先做GPU数组探针，再写GPU TF-IDF。
6. 开新分支分别开发，不覆盖现有公开结果。

## 后续开发顺序

### A. MPI
读MPI_TWO_NODES.md。先解决runtime/解释器路径/节点连通性，保存rank和hostname证据。然后新增mpi_main.py、MPI后端和验证/重复实验工具。默认采用rank0集中读文本并scatter的方案；各rank保留本地词频，汇总DF，广播IDF，gather紧凑结果。

### B. GPU
读GPU_BACKEND.md。先确认NVIDIA显卡、驱动与CuPy适配。建议从CPU分词+CSR构造、GPU数值权重/归一化的混合流水线开始，明确CPU阶段。不要分配N×V稠密矩阵。必须同时提供包含预处理/传输/回传的总时间和独立数值阶段时间；为后者建立相同CSR的CPU向量化对照。

### C. 统一实验与报告
读EXPERIMENT_PROTOCOL.md。四种方案使用相同文件、子集、公式和输出任务；额外记录分布式节点、GPU与计时边界。两个不同硬件主机导致的性能差异需要说明。不要直接拿GPU新机器时间除以旧笔记本历史时间当作纯GPU加速。

## 验收交付

- CPU旧命令与测试继续通过。
- MPI单机与双机验证明确区分，双机有实际hostname列表和每机rank数。
- GPU实际运行证明、dtype、设备型号和同步计时证据。
- 小样例、边界输入、真实子集逐词权重核对；全量top-k核对。
- 原始重复实验CSV、JSON配置和阶段记录、图、使用命令、失败/跳过状态。
- README状态表只有在对应后端实现并验证后才能修改为已完成。

