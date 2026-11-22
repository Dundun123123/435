"""Experimental multiple-process, single-GPU TF-IDF pipeline.

Each spawned process owns a CUDA context and weights a CSR batch on the same
physical GPU. CPU tokenization and result reconstruction are counted in the
full-pipeline timer. This is process-level scheduling, not multiple GPUs.
"""
import os
import multiprocessing as mp
import traceback
from time import perf_counter

import numpy as np

from gpu_tfidf import _chunks, _restore, prepare_csr


_WORKER_IDF = None


def _init_gpu_worker(host_idf):
    import cupy as cp
    global _WORKER_IDF
    cp.cuda.Device(0).use()
    _WORKER_IDF = cp.asarray(host_idf)
    cp.cuda.Stream.null.synchronize()


def _weight_shard(task):
    import cupy as cp
    row, end, lo, hi, local_indptr, indices, tf = task
    if lo == hi:
        return lo, hi, np.empty(0, dtype=np.float64), os.getpid()
    gpu_indices = cp.asarray(indices)
    gpu_tf = cp.asarray(tf)
    gpu_lengths = cp.asarray(np.diff(local_indptr))
    row_ids = cp.repeat(cp.arange(end - row), gpu_lengths)
    weighted = gpu_tf * _WORKER_IDF[gpu_indices]
    norms = cp.sqrt(cp.bincount(row_ids, weights=weighted * weighted,
                               minlength=end - row))
    values = (weighted / norms[row_ids]).get()
    cp.cuda.Stream.null.synchronize()
    return lo, hi, values, os.getpid()


def _worker_group(connection, host_idf, tasks):
    try:
        _init_gpu_worker(host_idf)
        connection.send(('ok', [_weight_shard(task) for task in tasks]))
    except BaseException:
        connection.send(('error', traceback.format_exc()))
    finally:
        connection.close()


def compute_tfidf_gpu_multiprocess(texts, workers=2, batch_size=500,
                                   top_k=10, profile=None):
    """Return vectors, IDF, elapsed seconds using several CUDA processes."""
    if workers < 2 or batch_size < 1 or (top_k is not None and top_k < 1):
        raise ValueError('workers >= 2, batch_size >= 1, positive top_k required')
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError('No CUDA device available')
    start = perf_counter()
    csr = prepare_csr(texts, batch_size)
    indptr, indices, tf, words = csr
    prepared_at = perf_counter()
    n = len(texts)
    df = cp.zeros(len(words), dtype=cp.int64)
    for _, _, lo, hi in _chunks(indptr, batch_size):
        if lo < hi:
            df += cp.bincount(cp.asarray(indices[lo:hi]), minlength=len(words))
    gpu_idf = cp.log((n + 1) / (df + 1)) + 1.0
    host_idf = cp.asnumpy(gpu_idf)
    cp.cuda.Stream.null.synchronize()
    idf_at = perf_counter()
    output = np.empty(len(indices), dtype=np.float64)
    tasks = []
    for row, end, lo, hi in _chunks(indptr, batch_size):
        tasks.append((row, end, lo, hi, (indptr[row:end + 1] - lo).copy(),
                      indices[lo:hi].copy(), tf[lo:hi].copy()))
    pids = set()
    context = mp.get_context('spawn')
    count = min(workers, len(tasks))
    groups = [tasks[i::count] for i in range(count)]
    connections, processes = [], []
    try:
        for group in groups:
            parent, child = context.Pipe(duplex=False)
            process = context.Process(target=_worker_group, args=(child, host_idf, group))
            process.start()
            child.close()
            connections.append(parent)
            processes.append(process)
        for connection in connections:
            status, payload = connection.recv()
            if status != 'ok':
                raise RuntimeError(payload)
            for lo, hi, values, pid in payload:
                output[lo:hi] = values
                pids.add(pid)
    finally:
        for connection in connections:
            connection.close()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join()
    weighted_at = perf_counter()
    vectors = _restore(indptr, indices, output, words, top_k)
    seconds = perf_counter() - start
    if profile is not None:
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name']
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode('utf-8', errors='replace')
        profile.clear()
        profile.update(preprocess_and_csr_seconds=prepared_at - start,
                       global_df_idf_seconds=idf_at - prepared_at,
                       spawn_transfer_gpu_weight_return_seconds=weighted_at - idf_at,
                       reconstruction_seconds=seconds - (weighted_at - start),
                       total_seconds=seconds, requested_workers=workers,
                       active_gpu_processes=len(pids), worker_pids=sorted(pids),
                       gpu_model=gpu_name, precision='float64',
                       batch_size=batch_size, nnz=len(indices), vocabulary=len(words))
    return vectors, dict(zip(words, host_idf.tolist())), seconds
