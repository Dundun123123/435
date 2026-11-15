# 数据获取与复现

Git中只提供sample.jsonl、自编示例、公开数据来源清单及原始IMDb README。完整数据通过脚本下载，不存入Git。

|文件|来源/用途|获取方式|
|---|---|---|
|sample.jsonl|项目自编的5篇示例，含空文章|已提交|
|newsgroups.jsonl|20 Newsgroups，18311篇有效帖子|`python prepare_data.py`|
|imdb.jsonl|Stanford IMDb，98463篇清理去重影评|`python prepare_large_data.py`|
|imdb_short.jsonl / imdb_long.jsonl|分别5000篇、由同一IMDb语料取样|同上|

在项目虚拟环境运行。20 Newsgroups加载器依赖scikit-learn。IMDb脚本使用标准库，下载D2L镜像，校验发布的SHA-1，再记录SHA-256；不展开10万个小文件。两台机器可各自下载，也可由用户通过局域网复制处理后的JSONL。MPI rank0集中读取方案不要求每个节点都存大语料，但代码和解释器必须都可用。

必须核对：

```powershell
Get-FileHash data/imdb.jsonl -Algorithm SHA256
```

Linux对应 `sha256sum data/imdb.jsonl`。本次已发表CPU结果的数据校验值在imdb_manifest.json。若下载脚本或清理过程发生变化，保留新清单并说明，不能假装与原结果使用相同字节输入。

复现清理方法：去HTML、解码实体、合并空白、对清理后的完整文本精确去重、固定随机种子42。短长样本分别来自词数底部和顶部四分位，不重叠。标签只作元数据。

来源：
- https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset
- https://ai.stanford.edu/~amaas/data/sentiment/
- https://aclanthology.org/P11-1015/
- https://d2l.ai/_modules/d2l/torch.html

公开下载不构成本项目对原文重新授权。原始IMDb说明见IMDB_ORIGINAL_README.txt。

