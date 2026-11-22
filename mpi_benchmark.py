"""Repeat the same 4-rank MPI experiment on one host and on two hosts.

Run with mpiexec. Only rank 0 reads the input and writes benchmark results.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import socket
import statistics
from collections import Counter
from pathlib import Path

from mpi4py import MPI

from mpi_tfidf import compute_tfidf_mpi
from tfidf import compute_tfidf, read_documents


def same_output(expected, actual):
    if len(expected) != len(actual):
        return False
    return all(a.keys() == b.keys() and all(
        math.isclose(a[word], b[word], rel_tol=1e-10, abs_tol=1e-12) for word in a)
        for a, b in zip(expected, actual))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/imdb.jsonl')
    parser.add_argument('--limit', type=int, default=10000)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--require-hosts', type=int, choices=(1, 2), required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    if args.limit < 1 or args.top_k < 1 or args.warmups < 1 or args.repeats < 2:
        parser.error('Positive limit/top-k/warmups and at least two repeats required')

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    hosts = comm.gather(socket.gethostname(), root=0)
    if rank == 0:
        distinct_hosts = sorted(set(hosts))
        placement = Counter(hosts)
        placement_ok = (comm.Get_size() == 4 and len(distinct_hosts) == args.require_hosts and
                        (args.require_hosts != 2 or sorted(placement.values()) == [2, 2]))
    else:
        placement_ok = None
    placement_ok = comm.bcast(placement_ok, root=0)
    if not placement_ok:
        if rank == 0:
            print(f'Invalid rank placement: {list(enumerate(hosts))}', flush=True)
        raise SystemExit('Expected four ranks on one host or two ranks on each of two hosts')
    if rank == 0:
        docs = read_documents(args.input, args.limit)
        texts = [doc['text'] for doc in docs]
        reference, reference_idf, _ = compute_tfidf(texts, top_k=args.top_k)
        input_sha256 = hashlib.sha256(Path(args.input).read_bytes()).hexdigest()
        print(f'Rank placement: {list(enumerate(hosts))}', flush=True)
        print(f'Input: {len(texts):,} documents | SHA256: {input_sha256}', flush=True)
    else:
        texts = reference = reference_idf = input_sha256 = None

    rows = []
    for trial in range(-args.warmups, args.repeats):
        result = compute_tfidf_mpi(texts, comm, top_k=args.top_k)
        if rank == 0:
            vectors, idf, seconds = result
            if not same_output(reference, vectors):
                raise AssertionError(f'TF-IDF output differs in trial {trial}')
            if idf.keys() != reference_idf.keys() or any(
                not math.isclose(idf[word], reference_idf[word], rel_tol=1e-10, abs_tol=1e-12)
                for word in reference_idf):
                raise AssertionError(f'IDF differs in trial {trial}')
            row = {'trial': trial, 'warmup': trial < 0, 'seconds': seconds,
                   'rank_count': comm.Get_size(), 'host_count': len(distinct_hosts),
                   'rank_hosts': json.dumps(hosts)}
            rows.append(row)
            print(f"Trial {trial:>2} | {seconds:.6f} s | hosts={len(distinct_hosts)} "
                  f"| correct=yes", flush=True)

    if rank == 0:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with (output / 'raw.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        times = [row['seconds'] for row in rows if not row['warmup']]
        summary = {'status': 'passed', 'documents': len(texts), 'dataset_sha256': input_sha256,
                   'top_k': args.top_k, 'rank_count': comm.Get_size(),
                   'host_count': len(distinct_hosts), 'rank_hosts': hosts,
                   'rank_placement': list(enumerate(hosts)),
                   'warmups': args.warmups, 'repeats': args.repeats,
                   'median_seconds': statistics.median(times),
                   'mean_seconds': statistics.mean(times),
                   'min_seconds': min(times), 'max_seconds': max(times),
                   'python': platform.python_version(), 'os': platform.platform(),
                   'timing_scope': 'root in-memory text to gathered top-k dictionaries, '
                                   'including scatter, global DF/IDF, broadcast and gather; excludes file I/O',
                   'correctness': 'each trial matched root CPU sequential top-k and IDF'}
        (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(f"Median: {summary['median_seconds']:.6f} s", flush=True)
        print(f'Output: {output}', flush=True)


if __name__ == '__main__':
    main()
