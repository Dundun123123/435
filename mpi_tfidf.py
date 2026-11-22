"""MPI TF-IDF with the same tokenizer, IDF, and top-k contract as tfidf.py.

Rank 0 reads the corpus before this function. All ranks call compute_tfidf_mpi.
The reported wall time includes scatter, both global reductions, and gather.
"""
import math
from collections import Counter
from time import perf_counter

from tfidf import count_batch, weight_batch


def compute_tfidf_mpi(texts_on_root, comm, top_k=10):
    rank = comm.Get_rank()
    size = comm.Get_size()
    comm.Barrier()
    start = perf_counter() if rank == 0 else None

    if rank == 0:
        if not texts_on_root:
            raise ValueError('Nonempty input required')
        n = len(texts_on_root)
        shards = [texts_on_root[(n * r) // size:(n * (r + 1)) // size]
                  for r in range(size)]
    else:
        shards = None
    local_texts = comm.scatter(shards, root=0)
    local_counts, local_df = count_batch(local_texts)
    dfs = comm.gather(local_df, root=0)
    if rank == 0:
        global_df = Counter()
        for df in dfs:
            global_df.update(df)
        if not global_df:
            raise ValueError('The corpus contains no English tokens matching [a-z]+.')
        idf = {word: math.log((n + 1) / (frequency + 1)) + 1
               for word, frequency in global_df.items()}
    else:
        idf = None
    idf = comm.bcast(idf, root=0)
    local_vectors = weight_batch(local_counts, idf, top_k)
    vectors_by_rank = comm.gather(local_vectors, root=0)
    if rank != 0:
        return None
    vectors = [vector for shard in vectors_by_rank for vector in shard]
    return vectors, idf, perf_counter() - start
