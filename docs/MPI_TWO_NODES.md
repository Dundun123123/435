# 双机MPI对接与实施指南

**状态：本指南、主机配置示例和diagnostics/mpi_probe.py已准备；mpi_main.py及MPI TF-IDF后端尚未实现。下文明确区分现有探针与待实现命令。**

## 1. 硬件和网络

- 两台实体电脑，同一可信局域网；优先有线连接，Wi-Fi可用但记录带宽/链路情况。
- 记录两台机器CPU、RAM、OS、Python和MPI版本；不要把不同规格隐藏掉。
- 两端checkout同一个Git提交，同一MPI实现/兼容版本，同一Python主版本，建议同为Python3.13。
- 代码、Python解释器在两个节点都存在且可被远程启动；不要把Windows .venv复制到另一台机器。
- 先做2个rank（每机1个），再做4个rank（每机2个）。总rank数包含rank0，不能把它写成另外多出的工作进程。
- Windows与Linux原生MPI运行时通常不能直接拼成一个作业。优先两端同为Windows MS-MPI，或同为Linux Open MPI；混合系统先由接手Codex重新选择兼容路线。

## 2. Windows + MS-MPI（与当前项目最接近）

两台机器各自：
1. 安装Git和64位Python，clone代码。
2. 从Microsoft官方发布页安装匹配的MS-MPI Runtime；使用预编译mpi4py轮子通常不需要SDK，源码编译才可能需要开发工具。
3. 安装项目环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-mpi.txt
.\.venv\Scripts\python.exe -c "from mpi4py import MPI; print(MPI.Get_library_version())"
```

4. 检查 `Get-Command mpiexec`，避免误用Intel MPI或其他MPI发行版的mpiexec搭配MS-MPI。
5. 两端通过MS-MPI Launch Service或对应进程管理器支持远程启动。以当前安装版本和官方RunningMSMPI文档为准；由接手Codex先检测实际服务名、状态和本机帮助，不照抄网络上含拼写错误的服务/进程名。
6. 只为可信私有网络配置必要的MPI远程启动和应用通信规则，遵守用户对系统变更的授权；不要关闭整台机器的防火墙，也不要把密码写进Git配置文件。两端账户和远程启动认证需要按实际环境解决。

### 现有命令：单机探针

在仓库根目录：

```powershell
mpiexec -n 2 .\.venv\Scripts\python.exe diagnostics/mpi_probe.py --output results/local/mpi_probe_local.json
```

必须成功显示2个rank，但这只是本机验证，不是双机证明。

### 现有命令：双机探针模板

假设两端代码都位于`C:\CST435\435`，解释器都在其.venv下。该路径只是示例，接手Codex应使用用户可写且两端一致的实际路径。

```powershell
mpiexec -hosts 2 NODE_A 1 NODE_B 1 -wdir C:\CST435\435 C:\CST435\435\.venv\Scripts\python.exe diagnostics/mpi_probe.py --require-hosts 2 --output results/local/mpi_probe_two_hosts.json
```

把NODE_A/NODE_B换成实际可解析主机名。先确认两端能解析/访问，再运行；是否能ping通不是MPI通信一定成功的证明。输出必须有两个不同实际hostname和每个rank的解释器路径，否则探针失败。

MS-MPI主机文件示例见configs/hosts.msmpi.example；与Open MPI格式不同，不可混用。

## 3. Linux + Open MPI（替代路线）

两端安装相同兼容Open MPI、Python环境和mpi4py；可使用系统包管理器提供的openmpi-bin/libopenmpi-dev等对应包，具体命令由接手Codex按发行版确认。不要在已配置MS-MPI的环境盲装其他运行时。

Open MPI通常通过SSH远程启动，两端之间应能非交互登录，远程shell能找到MPI可执行程序和动态库。私钥留本机，不提交。代码和解释器使用两端一致路径，或明确设置远程路径，不能假设两端用户名相同。

把configs/hosts.openmpi.example复制为忽略的hosts.local.openmpi，替换实际主机名和slots。假设两端同一路径`/srv/cst435/435`：

```bash
mpirun --hostfile hosts.local.openmpi -np 4 --wdir /srv/cst435/435 /srv/cst435/435/.venv/bin/python diagnostics/mpi_probe.py --require-hosts 2 --output results/local/mpi_probe_two_hosts.json
```

不要通过允许root、关闭防火墙或屏蔽运行时错误来假装部署成功。WSL2双机网络与主机名/SSH可达性需要额外验证，不能把Windows主机IP自动当成WSL可达节点IP。

## 4. 接手Codex需实现的算法

计划接口（**当前不存在，不能直接运行**）：

```text
mpiexec ... python mpi_main.py --input data/imdb.jsonl --limit 90000 --top-k 10 --output results/local/mpi_keywords.jsonl
```

建议保留独立mpi_main.py入口，避免普通main.py启动多个MPI rank时重复输出/写文件。

1. rank0读取JSONL，保留原始索引；读取不纳入计算主计时，另报I/O。
2. 所有rank Barrier后开始MPI.Wtime。rank0分批/分区，scatter到各rank。rank0也参与计算并记录这个事实。
3. 各rank复用count_batch，保留词频；仅gather局部DF到rank0。
4. rank0合并整数DF、计算全局IDF，再bcast一次。不能每个节点独立算局部IDF。
5. 各rank复用weight_batch，输出相同top-k或完整向量。可以复用v3紧凑数组格式；优先逐步实现正确，再优化大对象传输。
6. gather结果到rank0，按索引重建。rank0写文件；其他rank不得争写同一文件。
7. 计时终点须在全局工作完成、rank0重建完成之后。报告各rank耗时的MAX及rank0最终结果就绪时间，说明主指标选择。
8. 错误要广播/全局处理，避免rank0在读输入失败后退出、其他rank永远卡在collective。任何非零退出均不能被记成成功实验。

## 5. 计时与实验设计

两类时间分开：
- 作业内部：Barrier到完整结果恢复，包含分发、collective、网络交换与计算；通常不包含mpiexec启动Python和MPI_Init。
- 冷启动：外层计时器包住整个mpiexec命令，包含远程启动和文件输出；串行/multiprocessing也采用相同外层边界作为额外对照。

不要拿不含启动的MPI内部时间，与含启动的multiprocessing历史总时间直接宣称MPI更快。可在新实验中同时保留各自完整冷启动时间，并单列内部计算流水线。

建议矩阵：单机MPI 1/2/4 ranks；双机2 ranks=1+1、4 ranks=2+2。采用同一输入、同一top-10任务、每配置1次预热+10次正式运行。双机与单机若总核心数不同，报告资源增加；另做相同总rank数对比以观察网络影响。

## 6. 验收

- sample与真实100/1000篇逐词权重对照CPU，容差rtol=1e-10、atol=1e-12。
- 空文档、空词表、重复内容、失效路径、不均匀分区、不足rank数的文档数量。
- 至少两台实体主机证据，保存rank→host映射（公开前去除敏感IP）。
- 同一提交、同一输入SHA256；每次实际rank数/每机rank数、两机规格、网络说明。
- 明确跳过/未部署/失败，不写虚构双机耗时。

## 官方参考
- https://mpi4py.readthedocs.io/en/stable/install.html
- https://github.com/microsoft/Microsoft-MPI/releases
- https://github.com/microsoft/Microsoft-MPI/blob/master/docs/RunningMSMPI.md
- https://github.com/microsoft/Microsoft-MPI/blob/master/src/mpi/mpiexec/mp_parse_command_line.cpp
- https://docs.open-mpi.org/en/main/launching-apps/ssh.html

