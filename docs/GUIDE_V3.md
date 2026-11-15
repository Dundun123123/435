# v3：真实大数据、阶段计时和重复实验

> GitHub仓库版本说明：本文保留最初本地交付的CPU说明。新电脑clone后，环境和完整数据尚不存在，必须先按根目录README安装环境并执行prepare_data.py/prepare_large_data.py；不能把下面“当前电脑已配置”的描述理解为clone会自动带来环境或大语料。公开结果在results/study_v3，新测量请改为results/local/以免覆盖历史结果。

## 当前电脑直接运行

在本项目目录打开PowerShell。环境已配置，数据已下载，不需要重新安装或联网。

```powershell
# 同一份大数据，两个执行模式，输出任务相同（top-10关键词）
powershell -ExecutionPolicy Bypass -File .\run.ps1 main --input data/imdb.jsonl --mode sequential --output results/imdb_seq.jsonl
powershell -ExecutionPolicy Bypass -File .\run.ps1 main --input data/imdb.jsonl --mode parallel --workers 4 --output results/imdb_par.jsonl

# 重做整套实验：三个规模、四种进程数、长短文档，每配置预热1次+正式10次
powershell -ExecutionPolicy Bypass -File .\run.ps1 study

# 只重做某个规模（建议先用这个熟悉输出）
powershell -ExecutionPolicy Bypass -File .\run.ps1 benchmark --input data/imdb.jsonl --sizes 90000 --workers 1 2 4 8 --repeats 10 --warmups 1 --top-k 10 --output-dir results/my_imdb_90000

# 正确性检查
powershell -ExecutionPolicy Bypass -File .\run.ps1 verify

# 全部98463篇的top-10结果核对，以及三类文本的scikit-learn核对
powershell -ExecutionPolicy Bypass -File .\run.ps1 validate_large
```

整套实验会反复处理真实大数据，运行时间明显长于单次演示。不要同时开另一份性能实验。换电脑后先运行setup.ps1。run.ps1只为本次进程放开脚本策略，不修改系统设置。

## 新增数据

来源：Stanford Large Movie Review Dataset v1.0。
原始站点：https://ai.stanford.edu/~amaas/data/sentiment/
论文：https://aclanthology.org/P11-1015/
下载使用D2L镜像：https://d2l-data.s3-accelerate.amazonaws.com/aclImdb_v1.tar.gz
镜像校验值出处：https://d2l.ai/_modules/d2l/torch.html

下载包先通过D2L公布的SHA-1核对，再记录SHA-256。直接流式读取压缩包中的文本，不解压10万个小文件，不执行包内代码。原始README保留在data/IMDB_ORIGINAL_README.txt。原文权利不由本项目重新授权，提交时引用原始来源。

清理HTML、转义字符和多余空白后，按清理后的完整正文精确去重。原始100000篇，去掉1537篇重复内容，保留98463篇，23257053个英文词。情感标签仅为元数据，不训练分类器。

|文件|篇数|词数|JSONL文件字节数|
|---|---|---|---|
|data/imdb.jsonl|98463|23257053|138001468|
|data/imdb_short.jsonl|5000|473642|3099404|
|data/imdb_long.jsonl|5000|2406105|13871669|

short和long分别从全语料词数的底部/顶部四分位中固定种子随机抽样5000篇，两组不重叠。不是复制扩展，也不是严格仅改变长度的因果实验：文章内容和词汇量也会变化。所有来源、清理方法、数量和SHA256见data/imdb_manifest.json。

## 结构改进

1. 复用count_batch、weight_batch；相同TF、IDF、分词和归一化。
2. 每个工作进程保留本地词频，只回传局部DF，避免词频往返。
3. 全局IDF每个工作进程只接收一次。
4. 用文档索引、偏移量、词编号和双精度权重四组紧凑数组传输结果，主进程恢复相同的稀疏字典接口。
5. 分批按字符量分配，结果按原始索引复原。
6. 默认top-k模式只传所需关键词；--full仍可计算并返回所有词权重。

不是所有输入都适合并行。小语料、过多进程、大词表和内存压力都可能导致变慢。这个项目不保证增加进程就加速。

## 阶段计时的含义

main.py终端和输出旁的summary.json包含profile：

主字段seconds仍表示TF-IDF计算流水线时间。另列input_read_seconds（读取）、diagnostics_seconds（语料统计）、output_write_seconds（结果整理和写出），以及through_output_seconds（main入口到关键词文件写完，包含参数解析，尚不包含最后的summary写出和终端打印）。这样可区分计算性能与实际运行体验。正式benchmark仍以范围一致的流水线时间比较，不混用两种时间口径。

- 串行：词频统计、DF合并及IDF计算、权重计算与收集。
- 并行：分区、进程启动及输入分配、等待词频与DF汇总、全局IDF计算及分发、等待权重与结果解码合并、进程关闭。
- 工作进程诊断：本地计数及DF发送、等待IDF、权重计算及打包、传输的数字数组字节数（不是精确的网络/管道总字节数）。

主进程的阶段是连续墙钟区间，每个区间可能同时包含通信、等待和其他进程的计算。工作进程诊断耗时互相重叠，也与主进程区间重叠，不能相加当总时间。父阶段和总时间间的小差额包括清理和函数调用开销。我们没有把阶段区间冒充纯计算或纯通信的精确测量。

## 实验与统计

默认repeats=10、warmups=1；每个配置获得相同次数的预热。每次调用均重新创建进程并计入启动/关闭时间，预热不意味着复用常驻进程。每轮配置顺序固定随机种子打乱，减轻顺序影响。

每次主计时包含从内存原文到输出字典的全部计算、IPC、进程启动关闭、打包解码。文件读取、计时外数据诊断、写结果、绘图、正确性验证不计入主时间，串行并行范围一致。top-k选择在两种模式中均计时。

输出：raw.csv（每轮时间）、summary.csv（中位数/均值/样本标准差/IQR/加速比/效率）、profiles.jsonl（包含预热的完整诊断）、phases.csv（正式轮次父阶段）、environment.json（环境、参数、输入SHA256）。

耗时图误差条是25%~75%四分位区间IQR，不是置信区间。加速比是中位数之比；按相同轮次索引做1000次配对bootstrap，给出95%百分位区间，作为本次重复测量的探索性不确定性描述。区间跨1时不把微弱优势称为稳定加速。效率按配置进程数计算，另记录active_workers供核查。

study.py按顺序完成scale、short、long三组，避免实验间并发干扰。规模实验使用已固定打乱语料的前10000/50000/90000篇，形成嵌套子集。结果路径results/study_v3，汇总为REPORT.md；每组包含耗时、加速比、效率误差图和父阶段均值堆叠图。

历史v1/v2结果保留，不与v3混算。重复实验可能因后台负载、温度、内存和解释器状态发生变化；不要只选择最快一次。
