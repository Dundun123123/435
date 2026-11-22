"""Repeat same-machine CPU/GPU TF-IDF experiments with raw measurements."""
import argparse
import csv
import ctypes
import hashlib
import json
import os
import platform
import random
import statistics
from pathlib import Path
from time import perf_counter

import numpy as np

from gpu_tfidf import compute_tfidf_gpu, cpu_numeric, gpu_numeric, prepare_csr
from tfidf import TOKEN, compute_tfidf, read_documents


def total_ram_bytes():
    if os.name != 'nt':
        return None
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong) for name in
            ('total_physical', 'available_physical', 'total_page',
             'available_page', 'total_virtual', 'available_virtual',
             'available_extended')]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    return int(status.total_physical) if ctypes.windll.kernel32.GlobalMemoryStatusEx(
        ctypes.byref(status)) else None


def check_vectors(reference, candidate):
    if len(reference) != len(candidate):
        raise AssertionError('Document count mismatch')
    for row, (a, b) in enumerate(zip(reference, candidate)):
        if a.keys() != b.keys():
            raise AssertionError(f'Document {row}: term mismatch')
        for word in a:
            if not np.isclose(a[word], b[word], rtol=1e-10, atol=1e-12):
                raise AssertionError(f'Document {row}, term {word}: weight mismatch')


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/newsgroups.jsonl')
    parser.add_argument('--limit', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--output-dir', default='results/local/gpu_benchmark')
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 1:
        parser.error('repeats and warmups must be positive')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    docs = read_documents(args.input, args.limit)
    texts = [doc['text'] for doc in docs]
    csr = prepare_csr(texts, args.batch_size)
    reference, reference_idf, _ = compute_tfidf(texts, top_k=args.top_k)
    cpu_check = cpu_numeric(csr, args.batch_size, args.top_k)
    cold_start = perf_counter()
    gpu_check = gpu_numeric(csr, args.batch_size, args.top_k)
    cold_gpu_numeric_seconds = perf_counter() - cold_start
    for name, pair in [('cpu_numeric', cpu_check), ('gpu_numeric', gpu_check)]:
        vectors, idf = pair
        check_vectors(reference, vectors)
        for word in reference_idf:
            if not np.isclose(reference_idf[word], idf[word], rtol=1e-10, atol=1e-12):
                raise AssertionError(f'{name}: IDF mismatch for {word}')

    def trial(backend):
        profile = {}
        if backend == 'cpu_sequential':
            vectors, _, seconds = compute_tfidf(texts, top_k=args.top_k, profile=profile)
        elif backend == 'cpu_parallel':
            vectors, _, seconds = compute_tfidf(texts, mode='parallel', workers=args.workers,
                                               batch_size=args.batch_size, top_k=args.top_k,
                                               profile=profile)
        elif backend == 'gpu_hybrid':
            vectors, _, seconds = compute_tfidf_gpu(texts, args.batch_size, args.top_k, profile)
        elif backend == 'cpu_numeric':
            start = perf_counter()
            vectors, _ = cpu_numeric(csr, args.batch_size, args.top_k, profile)
            seconds = perf_counter() - start
        else:
            start = perf_counter()
            vectors, _ = gpu_numeric(csr, args.batch_size, args.top_k, profile)
            seconds = perf_counter() - start
        check_vectors(reference, vectors)
        return {'backend': backend, 'scope': 'full_pipeline' if backend in
                ('cpu_sequential', 'cpu_parallel', 'gpu_hybrid') else 'numeric_shared_csr',
                'seconds': seconds, 'profile': profile}

    modes = ['cpu_sequential', 'cpu_parallel', 'gpu_hybrid', 'cpu_numeric', 'gpu_numeric']
    rows = []
    for iteration in range(-args.warmups, args.repeats):
        order = modes.copy()
        random.Random(42 + iteration).shuffle(order)
        for backend in order:
            row = trial(backend)
            row.update(trial=iteration + 1 if iteration >= 0 else iteration,
                       warmup=iteration < 0)
            rows.append(row)
            print(f'{row["trial"]:>2} {backend:>15} {row["seconds"]:.6f}s', flush=True)
    fields = ['trial', 'warmup', 'backend', 'scope', 'seconds', 'profile']
    with (output / 'raw.csv').open('w', newline='', encoding='utf-8') as dest:
        writer = csv.DictWriter(dest, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, 'profile': json.dumps(row['profile'], sort_keys=True)})
    summary = {}
    for backend in modes:
        values = [row['seconds'] for row in rows if row['backend'] == backend and not row['warmup']]
        sorted_values = sorted(values)
        summary[backend] = {'median_seconds': statistics.median(values),
                            'mean_seconds': statistics.mean(values),
                            'sample_std_seconds': statistics.stdev(values) if len(values) > 1 else None,
                            'iqr_seconds': (np.percentile(sorted_values, 75) -
                                            np.percentile(sorted_values, 25)).item()}
    baseline = summary['cpu_sequential']['median_seconds']
    summary['gpu_hybrid']['speedup_vs_cpu_sequential'] = baseline / summary['gpu_hybrid']['median_seconds']
    summary['cpu_parallel']['speedup_vs_cpu_sequential'] = baseline / summary['cpu_parallel']['median_seconds']
    numeric_base = summary['cpu_numeric']['median_seconds']
    summary['gpu_numeric']['speedup_vs_cpu_numeric'] = numeric_base / summary['gpu_numeric']['median_seconds']
    import cupy as cp
    gpu = cp.cuda.runtime.getDeviceProperties(0)
    name = gpu['name'].decode() if isinstance(gpu['name'], bytes) else gpu['name']
    metadata = {'status': 'passed', 'dataset_sha256': hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
                'input': args.input, 'documents': len(texts),
                'tokens': sum(len(TOKEN.findall(t.lower())) for t in texts),
                'task_kind': 'top_k', 'top_k': args.top_k, 'precision': 'float64',
                'batch_size': args.batch_size, 'workers': args.workers,
                'warmups': args.warmups, 'repeats': args.repeats,
                'cold_gpu_numeric_seconds': cold_gpu_numeric_seconds,
                'cpu_model': platform.processor(), 'logical_cpus': os.cpu_count(),
                'ram_bytes': total_ram_bytes(),
                'os': platform.platform(), 'python': platform.python_version(),
                'numpy': np.__version__, 'cupy': cp.__version__,
                'gpu_model': name, 'gpu_memory_bytes': int(gpu['totalGlobalMem']),
                'driver_version': cp.cuda.runtime.driverGetVersion(),
                'cuda_runtime': cp.cuda.runtime.runtimeGetVersion(),
                'text_utf8_bytes': sum(len(text.encode('utf-8')) for text in texts),
                'csr_nnz': len(csr[1]),
                'csr_host_bytes': sum(part.nbytes for part in csr[:3]),
                'max_batch_nnz': max(int(csr[0][end] - csr[0][row]) for row in
                                     range(0, len(texts), args.batch_size)
                                     for end in [min(row + args.batch_size, len(texts))]),
                'summary': summary}
    (output / 'summary.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    run()
