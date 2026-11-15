# 可以直接复制给另一台电脑Codex的提示词

先在新电脑clone仓库，并让Codex打开435文件夹。以下提示词要求它读取仓库，不依赖当前聊天记忆。

## 第一次接手

```text
请接手这个CST435W TF-IDF项目。先阅读根目录AGENTS.md、README.md、docs/HANDOFF.md和docs/EXPERIMENT_PROTOCOL.md，检查git状态，然后运行doctor.py及无需大语料的CPU测试。已完成的是CPU串行/多进程v3；MPI/GPU只是准备方案，不能假定已有后端。不要修改历史实测结果，也不要上传原始大数据、虚拟环境或凭据。先报告本机硬件与可用环境、当前功能和缺失项，然后在用户选择的扩展范围内继续实现。
```

## 双机MPI（两台电脑到位后）

```text
我要实现真实双机MPI TF-IDF。阅读AGENTS.md、docs/HANDOFF.md、docs/MPI_TWO_NODES.md和docs/EXPERIMENT_PROTOCOL.md。先检测两端OS和可用MPI运行时，确认项目/解释器路径、同一提交以及节点连通性；运行已有mpi_probe.py，必须验证两个不同实体主机，不能把单机多rank称为双机。然后在新分支实现独立mpi_main.py、共用核心函数的MPI后端和相应验证/重复实验。rank0读数据、scatter文档，各rank保留词频，gather局部DF，广播全局IDF，计算并gather紧凑结果，恢复输入顺序。保留CPU旧接口，处理collective错误避免挂死。使用相同数据/公式/top-10，区分MPI作业内时间与含mpiexec启动的冷启动时间。缺少对端访问或系统授权时明确指出具体缺失条件，不要虚构部署成功或性能结果。
```

## GPU（兼容显卡到位后）

```text
我要实现GPU TF-IDF。阅读AGENTS.md、docs/HANDOFF.md、docs/GPU_BACKEND.md和docs/EXPERIMENT_PROTOCOL.md。先检测实际GPU/驱动/显存，选择兼容的单一CuPy发行包，运行已有gpu_probe.py。然后在新分支实现可选GPU后端和gpu_main.py：复用CPU文本规则，构造稀疏CSR，在GPU计算全局DF/IDF/权重及L2归一化，回传后按原始顺序输出top-10。禁止构造巨大N×V稠密矩阵。先用float64对齐CPU，CPU预处理和数据传输必须计入整体时间。GPU计时要正确同步，数值内核时间独立报告，并建立相同CSR的CPU向量化对照。重测本机CPU基线，不能拿旧笔记本历史时间作为纯GPU分母。验证通过并保存真实重复测量前，不修改后端状态为已完成；无GPU时标明不可用，不使用CPU fallback冒充。
```

## 合并到报告

```text
请基于仓库中的真实CSV/JSON结果整理四种后端比较。先核实硬件、版本、输入SHA256、任务种类和计时口径是否可比。CPU v1/v2/v3历史实验分开标注；新增MPI/GPU必须有真实硬件证据和参考核对。给出中位数、误差区间、加速比；CPU/MPI按已定义进程数给效率，GPU不套用CPU线程效率公式。报告不超过老师规定页数，引用原始数据与工具官方文档，不编造缺失结果。
```

