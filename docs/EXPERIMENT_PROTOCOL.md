# 跨后端实验协议

## 必须一致

- 输入文件SHA256、固定顺序的同一个limit子集、文章总数N（含空文章）。
- `[a-z]+`小写分词，raw TF，平滑全局IDF，全向量L2归一化。
- 相同输出任务：完整向量或相同top-k。不能拿串行full对比MPI/GPU top-10。
- 返回原始文档顺序，词表编号映射正确，相同权重按词字母排序。
- 标准输出JSONL继续用id和keywords；可选完整vector。性能元数据单独JSON。

## 参考基线

1. 在每台新机器重跑CPU串行和multiprocessing，记录实际规格。
2. MPI单机与双机使用相同总rank数的对照，同时另列增加总资源的实验。
3. GPU全流程对照用同一GPU电脑上的CPU结果；数值内核对照用相同CSR的向量化CPU实现。
4. 旧笔记本的公开结果是历史记录，不是新平台的公平分母。

## 计时口径

|指标|包含|可用于什么结论|
|---|---|---|
|CPU现有流水线|从内存文本到结果字典、进程启动/IPC/重建/关闭|本机串行/多进程对比|
|MPI内部|Barrier后文本分发、计算、collective、网络、rank0恢复；全局完成|单列作业内对比，不冒充含远程启动时间|
|全进程冷启动|外层包住各模式完整命令，含Python/MPI启动、文件I/O、输出|额外的统一用户体验比较|
|GPU端到端流水线|原文CPU处理、CSR、H2D、设备运算、D2H、top-k/恢复，正确同步|主要CPU/GPU比较|
|GPU数值阶段/内核|按预先声明边界计时，CUDA events及同步|辅助解释，不直接等同全流程加速|

预热次数一致且明确：建议1次预热、10次正式测量。GPU初始化/JIT、MPI启动和常驻进程的区别要单列，不能悄悄删除不利开销。串行化、等待和计算可能重叠；父阶段墙钟区间不等于纯通信时间。

## 原始记录的最少字段

```text
git_commit, implementation, backend, task_kind, top_k, precision
dataset_sha256, documents, tokens, text_utf8_bytes, batch_size
trial, warmup, timing_scope, seconds
configured_workers, active_workers
node_count, rank_count, rank_host_mapping (MPI)
cpu_model, logical_cpus, ram_bytes, os, python, package_versions
gpu_model, gpu_memory_bytes, driver, cuda_runtime, cupy (GPU)
stage_durations, transfer_or_payload_definition
status: passed / failed / skipped / unavailable
```

新结果先放results/local/<实验名>，检查后再整理到可公开目录。机器真实IP、账户、token、私钥不公开；host映射可以在公开副本中替换为node-a/node-b，原始证据由小组本地保存。

## 统计与图

- 保留全部原始轮次，不删除慢值来改善结论。
- 时间给中位数、均值、样本标准差、IQR；图明确误差条含义。
- Speedup = 相同口径CPU串行中位时间 / 目标配置中位时间。
- CPU/MPI效率按相应进程/rank数计算，必须定义rank0是否参与工作。
- 不把GPU线程数代入CPU进程效率公式；GPU报告整体加速、设备参数与传输/内核分解。
- 区间跨1时不把微弱优势称为稳定加速；不承诺大数据或多进程总是更快。

## 发布门槛

CPU兼容测试通过；后端在真实硬件上跑通；参考核对通过；数据和配置可重现；原始测量与图齐全。只有设备探针通过，不满足TF-IDF后端完成条件。

