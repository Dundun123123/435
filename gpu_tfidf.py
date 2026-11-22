"""Optional CPU-tokenization / CUDA-numeric TF-IDF backend.

The sparse CSR arrays are shared by the CPU numeric comparator. CuPy is only
imported when the GPU backend is called, so the existing CPU program stays usable.
"""
import heapq
from array import array
from time import perf_counter

import numpy as np

from tfidf import count_batch


def prepare_csr(texts, batch_size=500):
    if not texts or batch_size < 1:
        raise ValueError('Nonempty input and positive batch size required.')
    words, word_ids = [], {}
    indptr, indices, values = array('q', [0]), array('i'), array('d')
    for start in range(0, len(texts), batch_size):
        counts, _ = count_batch(texts[start:start + batch_size])
        for count in counts:
            for word, frequency in count.items():
                if word not in word_ids:
                    word_ids[word] = len(words)
                    words.append(word)
                indices.append(word_ids[word])
                values.append(float(frequency))
            indptr.append(len(indices))
    if not words:
        raise ValueError('The corpus contains no English tokens matching [a-z]+.')
    return (np.frombuffer(indptr, dtype=np.int64),
            np.frombuffer(indices, dtype=np.int32),
            np.frombuffer(values, dtype=np.float64), words)


def _chunks(indptr, batch_size):
    for row in range(0, len(indptr) - 1, batch_size):
        end = min(row + batch_size, len(indptr) - 1)
        yield row, end, int(indptr[row]), int(indptr[end])


def _restore(indptr, indices, values, words, top_k):
    vectors = []
    for row in range(len(indptr) - 1):
        pairs = [(words[int(indices[j])], float(values[j]))
                 for j in range(int(indptr[row]), int(indptr[row + 1]))]
        if top_k is not None:
            pairs = heapq.nsmallest(top_k, pairs, key=lambda item: (-item[1], item[0]))
        vectors.append(dict(pairs))
    return vectors


def cpu_numeric(csr, batch_size=500, top_k=None, profile=None, restore=True):
    """Vectorized CPU comparator on the exact CSR used by the GPU."""
    start = perf_counter()
    indptr, indices, tf, words = csr
    n = len(indptr) - 1
    df = np.bincount(indices, minlength=len(words))
    idf = np.log((n + 1) / (df + 1)) + 1.0
    output = np.empty(len(indices), dtype=np.float64)
    for row, end, lo, hi in _chunks(indptr, batch_size):
        if lo == hi:
            continue
        row_ids = np.repeat(np.arange(end - row), np.diff(indptr[row:end + 1]))
        weighted = tf[lo:hi] * idf[indices[lo:hi]]
        norms = np.sqrt(np.bincount(row_ids, weights=weighted * weighted,
                                   minlength=end - row))
        output[lo:hi] = weighted / norms[row_ids]
    computed_at = perf_counter()
    vectors = _restore(indptr, indices, output, words, top_k) if restore else output
    if profile is not None:
        profile.update(calculation_seconds=computed_at - start,
                       reconstruction_seconds=perf_counter() - computed_at,
                       total_seconds=perf_counter() - start)
    return vectors, dict(zip(words, idf.tolist())) if restore else idf


def gpu_numeric(csr, batch_size=500, top_k=None, profile=None, restore=True):
    """GPU DF/IDF/weights/norms with bounded batches and synchronized timings."""
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError('CuPy is required for the GPU backend; see docs/GPU_BACKEND.md') from exc
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError('No CUDA device available')
    indptr, indices, tf, words = csr
    n = len(indptr) - 1
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    sync = cp.cuda.Stream.null.synchronize
    sync()
    start = perf_counter()
    kernel_ms = 0.0
    df = cp.zeros(len(words), dtype=cp.int64)
    for _, _, lo, hi in _chunks(indptr, batch_size):
        if lo < hi:
            gpu_indices = cp.asarray(indices[lo:hi])
            event_start, event_end = cp.cuda.Event(), cp.cuda.Event()
            event_start.record()
            df += cp.bincount(gpu_indices, minlength=len(words))
            event_end.record()
            event_end.synchronize()
            kernel_ms += cp.cuda.get_elapsed_time(event_start, event_end)
    sync()
    df_at = perf_counter()
    event_start, event_end = cp.cuda.Event(), cp.cuda.Event()
    event_start.record()
    gpu_idf = cp.log((n + 1) / (df + 1)) + 1.0
    event_end.record()
    event_end.synchronize()
    kernel_ms += cp.cuda.get_elapsed_time(event_start, event_end)
    sync()
    idf_at = perf_counter()
    output = np.empty(len(indices), dtype=np.float64)
    for row, end, lo, hi in _chunks(indptr, batch_size):
        if lo == hi:
            continue
        gpu_indices = cp.asarray(indices[lo:hi])
        gpu_tf = cp.asarray(tf[lo:hi])
        gpu_lengths = cp.asarray(np.diff(indptr[row:end + 1]))
        event_start, event_end = cp.cuda.Event(), cp.cuda.Event()
        event_start.record()
        row_ids = cp.repeat(cp.arange(end - row), gpu_lengths)
        weighted = gpu_tf * gpu_idf[gpu_indices]
        norms = cp.sqrt(cp.bincount(row_ids, weights=weighted * weighted,
                                   minlength=end - row))
        normalized = weighted / norms[row_ids]
        event_end.record()
        event_end.synchronize()
        kernel_ms += cp.cuda.get_elapsed_time(event_start, event_end)
        output[lo:hi] = normalized.get()
    host_idf = cp.asnumpy(gpu_idf)
    sync()
    numeric_at = perf_counter()
    vectors = _restore(indptr, indices, output, words, top_k) if restore else output
    if profile is not None:
        profile.update(df_and_transfer_seconds=df_at - start,
                       idf_seconds=idf_at - df_at,
                       weight_transfer_and_return_seconds=numeric_at - idf_at,
                       reconstruction_seconds=perf_counter() - numeric_at,
                       numeric_total_seconds=perf_counter() - start,
                       device_kernel_seconds=kernel_ms / 1000.0,
                       precision='float64', nnz=len(indices), vocabulary=len(words),
                       batch_size=batch_size)
    return vectors, dict(zip(words, host_idf.tolist())) if restore else host_idf


def compute_tfidf_gpu(texts, batch_size=500, top_k=None, profile=None):
    if top_k is not None and top_k < 1:
        raise ValueError('top_k must be positive or None')
    start = perf_counter()
    if profile is not None:
        profile.clear()
    csr = prepare_csr(texts, batch_size)
    prepared_at = perf_counter()
    vectors, idf = gpu_numeric(csr, batch_size, top_k, profile)
    elapsed = perf_counter() - start
    if profile is not None:
        profile.update(preprocess_and_csr_seconds=prepared_at - start,
                       total_seconds=elapsed,
                       implementation='cpu-tokenization-gpu-sparse-numeric-v1')
    return vectors, idf, elapsed
