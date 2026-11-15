# 已公开的CPU结果

study_v3/为主要实测：IMDb同一实现、10次正式运行+1次预热，包括三个规模和各5000篇的长短样本；共250次正式测量。REPORT.md描述方法和限制。

benchmark/是初版Newsgroups实验（5次）；optimized_benchmark/和keyword_benchmark/是v2不同任务实验（5次）。不能把这些历史结果混合成同一组，或把版本/数据集同时改变后的收益全部归因于语料增大。

verification_v3.txt、full_corpus_verification_v3.json、large_validation_v3.json为原机器CPU验证记录；它们不是新机器的自动通过证明。

目前没有MPI TF-IDF或GPU TF-IDF性能结果。未来开发在local/运行，验证后再发布到明确命名的新目录。

