"""Measure one-host 4 processes or two-host 2+2 processes using same worker code."""
import argparse
import csv
import hashlib
import json
import math
import multiprocessing
import socket
import statistics
from multiprocessing.connection import Listener
from pathlib import Path
from time import perf_counter

from socket_tfidf import finish_workers, merge_df, receive, send_idf, start_workers
from tfidf import compute_tfidf, read_documents


def same_output(expected, actual):
    return len(expected) == len(actual) and all(
        a.keys() == b.keys() and all(
            math.isclose(a[word], b[word], rel_tol=1e-10, abs_tol=1e-12) for word in a)
        for a, b in zip(expected, actual))


def run_trial(texts, top_k, remote):
    start = perf_counter()
    shards = [texts[len(texts) * i // 4:len(texts) * (i + 1) // 4] for i in range(4)]
    if remote is not None:
        remote.send({'kind': 'task', 'shards': shards[2:], 'top_k': top_k})
        local_shards = shards[:2]
    else:
        local_shards = shards
    workers = start_workers(local_shards, top_k)
    try:
        local_dfs = receive(workers, 'df')
        if remote is not None:
            kind, remote_df = remote.recv()
            if kind != 'df':
                raise RuntimeError(f'Expected remote df, got {kind}: {remote_df}')
            local_dfs.append(remote_df)
        df = merge_df(local_dfs)
        if not df:
            raise ValueError('Corpus contains no [a-z]+ terms')
        n = len(texts)
        idf = {word: math.log((n + 1) / (frequency + 1)) + 1
               for word, frequency in df.items()}
        send_idf(workers, idf)
        if remote is not None:
            remote.send(('idf', idf))
        vectors_by_worker = receive(workers, 'vectors')
        if remote is not None:
            kind, remote_vectors = remote.recv()
            if kind != 'vectors':
                raise RuntimeError(f'Expected remote vectors, got {kind}: {remote_vectors}')
            vectors_by_worker.extend(remote_vectors)
        vectors = [row for shard in vectors_by_worker for row in shard]
        return vectors, idf, perf_counter() - start
    finally:
        finish_workers(workers)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/imdb.jsonl')
    parser.add_argument('--limit', type=int, default=10000)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--mode', choices=('one-host', 'two-host'), required=True)
    parser.add_argument('--listen', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--expected-peer')
    parser.add_argument('--key-file')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--allow-same-host-smoke', action='store_true')
    args = parser.parse_args()
    if args.limit < 1 or args.top_k < 1 or args.warmups < 1 or args.repeats < 2:
        parser.error('Positive limit/top-k/warmups and at least two repeats required')
    if args.mode == 'two-host' and not args.key_file:
        parser.error('--key-file is required for two-host mode')
    if args.mode == 'two-host' and not args.expected_peer:
        parser.error('--expected-peer is required for two-host mode')

    docs = read_documents(args.input, args.limit)
    texts = [doc['text'] for doc in docs]
    reference, reference_idf, _ = compute_tfidf(texts, top_k=args.top_k)
    digest = hashlib.sha256(Path(args.input).read_bytes()).hexdigest()
    local_host = socket.gethostname()
    listener = remote = None
    peer_host = None
    try:
        if args.mode == 'two-host':
            key = bytes.fromhex(Path(args.key_file).read_text(encoding='ascii').strip())
            if len(key) < 16:
                parser.error('Key file must contain at least 16 random bytes in hex')
            listener = Listener((args.listen, args.port), authkey=key)
            print(f'Waiting for laptop B on {args.listen}:{args.port} ...', flush=True)
            remote = listener.accept()
            peer_ip = listener.last_accepted[0]
            peer_info = remote.recv()
            peer_host = peer_info['hostname']
            if (peer_ip != args.expected_peer or peer_info['worker_processes'] != 2 or
                (peer_host == local_host and not args.allow_same_host_smoke)):
                raise RuntimeError(f'Invalid remote placement: IP={peer_ip}, host={peer_host}')
            print(f'Connected: {peer_host} ({peer_ip}), 2 remote CPU workers', flush=True)
        rows = []
        for trial in range(-args.warmups, args.repeats):
            vectors, idf, seconds = run_trial(texts, args.top_k, remote)
            if not same_output(reference, vectors):
                raise AssertionError(f'Top-k mismatch in trial {trial}')
            if idf.keys() != reference_idf.keys() or any(
                not math.isclose(idf[word], reference_idf[word], rel_tol=1e-10, abs_tol=1e-12)
                for word in reference_idf):
                raise AssertionError(f'IDF mismatch in trial {trial}')
            row = {'trial': trial, 'warmup': trial < 0, 'seconds': seconds,
                   'mode': args.mode, 'worker_processes': 4,
                   'hosts': json.dumps([local_host] * (2 if remote else 4) +
                                       ([peer_host] * 2 if remote else []))}
            rows.append(row)
            print(f'Trial {trial:>2} | {seconds:.6f} s | correct=yes', flush=True)
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with (output / 'raw.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        times = [row['seconds'] for row in rows if not row['warmup']]
        summary = {'status': 'passed', 'backend': 'multiprocessing plus authenticated TCP',
                   'mode': args.mode, 'host_count': 2 if remote else 1,
                   'worker_processes': 4, 'rank_hosts': json.loads(rows[0]['hosts']),
                   'documents': len(texts), 'dataset_sha256': digest,
                   'warmups': args.warmups, 'repeats': args.repeats,
                   'median_seconds': statistics.median(times),
                   'mean_seconds': statistics.mean(times),
                   'timing_scope': 'in-memory text to gathered top-k dictionaries; includes worker spawn, '
                                   'IPC and data transfer; excludes file I/O and initial TCP connection',
                   'correctness': 'each trial matched CPU sequential top-k and global IDF'}
        (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        print(f"Median: {summary['median_seconds']:.6f} s", flush=True)
        print(f'Output: {output}', flush=True)
    finally:
        if remote is not None:
            try:
                remote.send({'kind': 'stop'})
            except (OSError, EOFError):
                pass
            remote.close()
        if listener is not None:
            listener.close()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
