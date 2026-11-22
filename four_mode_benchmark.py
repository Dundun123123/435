"""Same-input four-mode benchmark with raw trials and correctness checks."""
import argparse
import csv
import hashlib
import json
import math
import platform
import random
import statistics
from pathlib import Path

from four_mode_main import run_mode
from tfidf import read_documents


MODES = ('cpu-single', 'cpu-multi', 'gpu-single', 'gpu-multi')


def assert_same(expected, actual):
    if len(expected) != len(actual):
        raise AssertionError('Document count differs')
    for row, (a, b) in enumerate(zip(expected, actual)):
        if a.keys() != b.keys():
            raise AssertionError(f'Keywords differ in row {row}')
        for word in a:
            if not math.isclose(a[word], b[word], rel_tol=1e-10, abs_tol=1e-12):
                raise AssertionError(f'Weight differs in row {row}, term {word}')


def quartiles(values):
    q1, _, q3 = statistics.quantiles(values, n=4, method='inclusive')
    return q1, q3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/imdb.jsonl')
    parser.add_argument('--limit', type=int, default=10000)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--output-dir', default='results/local/four_mode_10000')
    parser.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES),
                        help='Modes to measure; cpu-single is required as correctness baseline')
    args = parser.parse_args()
    if args.warmups < 1 or args.repeats < 2:
        parser.error('At least one warmup and two repeats are required')
    if 'cpu-single' not in args.modes:
        parser.error('cpu-single must be included')
    selected_modes = tuple(dict.fromkeys(args.modes))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    docs = read_documents(args.input, args.limit)
    texts = [d['text'] for d in docs]
    reference, reference_idf, _ = run_mode(texts, 'cpu-single', args.workers,
                                           args.batch_size, args.top_k, {})
    for mode in selected_modes:
        if mode == 'cpu-single':
            continue
        candidate, idf, _ = run_mode(texts, mode, args.workers,
                                     args.batch_size, args.top_k, {})
        assert_same(reference, candidate)
        if idf.keys() != reference_idf.keys() or any(
                not math.isclose(idf[w], reference_idf[w], rel_tol=1e-10, abs_tol=1e-12)
                for w in reference_idf):
            raise AssertionError(f'{mode} IDF mismatch')
    del reference_idf
    rows = []
    for iteration in range(-args.warmups, args.repeats):
        order = list(selected_modes)
        random.Random(142 + iteration).shuffle(order)
        for mode in order:
            profile = {}
            vectors, _, seconds = run_mode(texts, mode, args.workers,
                                            args.batch_size, args.top_k, profile)
            assert_same(reference, vectors)
            row = {'trial': iteration + 1 if iteration >= 0 else iteration,
                   'warmup': iteration < 0, 'mode': mode, 'seconds': seconds,
                   'active_gpu_processes': profile.get('active_gpu_processes', ''),
                   'profile': json.dumps(profile, sort_keys=True)}
            rows.append(row)
            print(f"{row['trial']:>2} {mode:>10} {seconds:.6f} s"
                  + (f" GPU processes={row['active_gpu_processes']}" if mode == 'gpu-multi' else ''),
                  flush=True)
    with (output / 'raw.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for mode in selected_modes:
        values = [r['seconds'] for r in rows if r['mode'] == mode and not r['warmup']]
        q1, q3 = quartiles(values)
        summary[mode] = {'median_seconds': statistics.median(values),
                         'mean_seconds': statistics.mean(values),
                         'sample_std_seconds': statistics.stdev(values),
                         'q1_seconds': q1, 'q3_seconds': q3,
                         'iqr_seconds': q3 - q1}
    serial = summary['cpu-single']['median_seconds']
    for mode in selected_modes:
        if mode == 'cpu-single':
            continue
        summary[mode]['speedup_vs_cpu_single'] = serial / summary[mode]['median_seconds']
    report = {'status': 'passed', 'input': args.input,
              'dataset_sha256': hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
              'documents': len(docs), 'workers': args.workers, 'batch_size': args.batch_size,
              'top_k': args.top_k, 'warmups': args.warmups, 'repeats': args.repeats,
              'modes': selected_modes,
              'gpu_process_design': 'multiple CUDA contexts on one physical RTX 4060',
              'scope': 'in-memory text to keyword dictionaries; excludes file I/O',
              'correctness': 'all documents top-k and global IDF matched CPU single',
              'python': platform.python_version(), 'os': platform.platform(),
              'summary': summary}
    (output / 'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
