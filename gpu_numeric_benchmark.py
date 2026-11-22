"""Compare CPU and GPU calculations on one prepared CSR, excluding restoration."""
import argparse
import csv
import gc
import hashlib
import json
import random
import statistics
from pathlib import Path

from gpu_tfidf import cpu_numeric, gpu_numeric, prepare_csr
from tfidf import read_documents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/imdb.jsonl')
    parser.add_argument('--limit', type=int, default=90000)
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--output-dir', default='results/local/gpu_numeric_90000')
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    docs = read_documents(args.input, args.limit)
    csr = prepare_csr([doc['text'] for doc in docs], args.batch_size)
    cpu_vectors, cpu_idf = cpu_numeric(csr, args.batch_size, args.top_k)
    gpu_vectors, gpu_idf = gpu_numeric(csr, args.batch_size, args.top_k)
    if cpu_vectors != gpu_vectors or cpu_idf != gpu_idf:
        # Exact equality is stricter than needed for float64, so inspect with tolerance.
        import math
        if len(cpu_vectors) != len(gpu_vectors):
            raise AssertionError('Document count mismatch')
        for a, b in zip(cpu_vectors, gpu_vectors):
            if a.keys() != b.keys() or any(not math.isclose(a[w], b[w], rel_tol=1e-10,
                                                            abs_tol=1e-12) for w in a):
                raise AssertionError('CPU/GPU numeric vectors differ')
        if any(not math.isclose(cpu_idf[w], gpu_idf[w], rel_tol=1e-10, abs_tol=1e-12)
               for w in cpu_idf):
            raise AssertionError('CPU/GPU IDF differs')
    del cpu_vectors, gpu_vectors, cpu_idf, gpu_idf
    gc.collect()
    rows = []
    for iteration in range(-args.warmups, args.repeats):
        order = ['cpu_numpy', 'gpu_cupy']
        random.Random(91 + iteration).shuffle(order)
        for backend in order:
            profile = {}
            if backend == 'cpu_numpy':
                cpu_numeric(csr, args.batch_size, args.top_k, profile, restore=False)
                calculation = profile['calculation_seconds']
                total = profile['total_seconds']
                kernel = None
            else:
                gpu_numeric(csr, args.batch_size, args.top_k, profile, restore=False)
                total = profile['numeric_total_seconds']
                calculation = total - profile['reconstruction_seconds']
                kernel = profile['device_kernel_seconds']
            row = {'backend': backend, 'trial': iteration + 1 if iteration >= 0 else iteration,
                   'warmup': iteration < 0, 'calculation_seconds': calculation,
                   'reconstruction_seconds': profile['reconstruction_seconds'],
                   'total_seconds': total, 'device_kernel_seconds': kernel,
                   'profile': json.dumps(profile, sort_keys=True)}
            rows.append(row)
            print(f'{row["trial"]:>2} {backend:>10} calculation={calculation:.6f}s '
                  f'total={total:.6f}s', flush=True)
    with (output / 'raw.csv').open('w', newline='', encoding='utf-8') as dest:
        writer = csv.DictWriter(dest, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for backend in ('cpu_numpy', 'gpu_cupy'):
        own = [r for r in rows if r['backend'] == backend and not r['warmup']]
        summary[backend] = {
            f'{metric}_median_seconds': statistics.median(r[f'{metric}_seconds'] for r in own)
            for metric in ('calculation', 'reconstruction', 'total')
        }
    report = {'status': 'passed', 'input': args.input,
              'dataset_sha256': hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
              'documents': len(docs), 'batch_size': args.batch_size,
              'top_k': args.top_k, 'warmups': args.warmups, 'repeats': args.repeats,
              'scope': 'same prepared CSR; calculation and timing exclude keyword reconstruction',
              'summary': summary}
    (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
