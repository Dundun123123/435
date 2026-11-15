"""Warmed, randomized repeated experiments with transparent uncertainty."""
import argparse
import csv
import hashlib
import json
import multiprocessing
import platform
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from tfidf import TOKEN, compute_tfidf, read_documents


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower = int(position)
    return values[lower] + (values[min(lower + 1, len(values) - 1)] - values[lower]) * (position - lower)


def speedup_interval(base, parallel):
    rng = random.Random(2026)
    samples = []
    for _ in range(1000):
        indices = [rng.randrange(len(base)) for _ in base]
        samples.append(statistics.median(base[i] for i in indices) /
                       statistics.median(parallel[i] for i in indices))
    return percentile(samples, .025), percentile(samples, .975)


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(summary, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for metric, ylabel in [('median_seconds', 'Execution time (s): median and IQR'),
                           ('speedup', 'Speedup: median ratio and bootstrap 95% interval'),
                           ('efficiency', 'Efficiency (fraction)')]:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for size in sorted({r['documents'] for r in summary}):
            rows = sorted([r for r in summary if r['documents'] == size and r['mode'] == 'parallel'], key=lambda r: r['workers'])
            ys = [r[metric] for r in rows]
            if metric == 'median_seconds':
                low = [r['median_seconds'] - r['q25_seconds'] for r in rows]
                high = [r['q75_seconds'] - r['median_seconds'] for r in rows]
            else:
                divisor = [r['workers'] if metric == 'efficiency' else 1 for r in rows]
                low = [max(0, y - r['speedup_ci_low'] / d) for y, r, d in zip(ys, rows, divisor)]
                high = [max(0, r['speedup_ci_high'] / d - y) for y, r, d in zip(ys, rows, divisor)]
            ax.errorbar([r['workers'] for r in rows], ys, yerr=[low, high], marker='o', capsize=4, label=f'{size:,} docs')
            if metric == 'median_seconds':
                seq = next(r for r in summary if r['documents'] == size and r['mode'] == 'sequential')
                ax.axhline(seq['median_seconds'], linestyle=':', alpha=.6, label=f'{size:,} seq median')
        if metric in ('speedup', 'efficiency'):
            ax.axhline(1, color='gray', linestyle='--', label='1.0 reference')
        ax.set(xlabel='Configured worker processes', ylabel=ylabel)
        ax.set_xticks(sorted({r['workers'] for r in summary if r['mode'] == 'parallel'}))
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(destination / f'{metric}.png', dpi=170)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/newsgroups.jsonl')
    parser.add_argument('--sizes', nargs='+', type=int, default=[1000, 5000, 10000])
    parser.add_argument('--workers', nargs='+', type=int, default=[1, 2, 4, 8])
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--top-k', type=int, default=0, help='0: full vectors; positive: keyword extraction')
    parser.add_argument('--output-dir', default='results/benchmark_v3')
    args = parser.parse_args()
    if min(args.sizes + args.workers + [args.repeats, args.batch_size]) < 1 or min(args.warmups, args.top_k) < 0:
        parser.error('Sizes/workers/repeats/batch-size must be positive; warmups/top-k nonnegative')
    docs = read_documents(args.input)
    if max(args.sizes) > len(docs):
        parser.error(f'Requested size exceeds available documents ({len(docs)})')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input)
    with input_path.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    environment = {'python': platform.python_version(), 'platform': platform.platform(),
        'processor': platform.processor(), 'logical_cpus': multiprocessing.cpu_count(),
        'arguments': vars(args), 'input_sha256': digest, 'input_file_bytes': input_path.stat().st_size,
        'started_utc': datetime.now(timezone.utc).isoformat(), 'implementation': 'packed-shard-processes-v3',
        'timing': 'In-memory raw text through output dictionaries; includes process startup, IPC, result decoding and shutdown. Excludes input/output files, token-count diagnostics, plotting and validation.',
        'uncertainty': 'Time error bars: sample IQR (not CI). Speedup: ratio of medians; paired-trial bootstrap percentile 95% interval, 1000 resamples, seed2026; exploratory, not a guarantee.',
        'phase_note': 'Parent stages are consecutive wall intervals, not pure compute/communication attribution. Worker timings overlap each other and parent stages.'}
    (out / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    rng = random.Random(42)
    raw, summary, phases = [], [], []
    with (out / 'profiles.jsonl').open('w', encoding='utf-8') as details:
        for size in sorted(set(args.sizes)):
            texts = [d['text'] for d in docs[:size]]
            lengths = [len(TOKEN.findall(t.lower())) for t in texts]
            tokens = sum(lengths)
            text_bytes = sum(len(t.encode('utf-8')) for t in texts)
            configurations = [('sequential', 1)] + [('parallel', w) for w in sorted(set(args.workers))]
            timings = {c: [] for c in configurations}
            for trial in range(1 - args.warmups, args.repeats + 1):
                order = configurations.copy()
                rng.shuffle(order)
                for mode, workers in order:
                    profile = {}
                    vectors, idf, seconds = compute_tfidf(texts, mode, workers, args.batch_size,
                        top_k=args.top_k or None, profile=profile)
                    del vectors, idf
                    row = dict(documents=size, tokens=tokens, text_utf8_bytes=text_bytes,
                        mean_document_tokens=statistics.mean(lengths), median_document_tokens=statistics.median(lengths),
                        mode=mode, workers=workers, active_workers=profile['active_workers'],
                        result_kind='keywords' if args.top_k else 'full_vectors', top_k=args.top_k,
                        trial=trial, warmup=trial <= 0, seconds=seconds)
                    details.write(json.dumps(dict(**row, profile=profile)) + '\n')
                    details.flush()
                    label = 'warmup' if trial <= 0 else f'trial={trial}'
                    print(f'{size} docs {mode} workers={workers} {label}: {seconds:.4f}s', flush=True)
                    if trial <= 0:
                        continue
                    timings[(mode, workers)].append(seconds)
                    raw.append(row)
                    for stage, duration in profile['stages'].items():
                        phases.append(dict(documents=size, mode=mode, workers=workers, trial=trial,
                                           stage=stage, seconds=duration))
                    write_csv(out / 'raw.csv', raw)
                    write_csv(out / 'phases.csv', phases)
            base = timings[('sequential', 1)]
            for (mode, workers), times in timings.items():
                median = statistics.median(times)
                low, high = speedup_interval(base, times) if mode == 'parallel' else (1, 1)
                summary.append(dict(documents=size, tokens=tokens, text_utf8_bytes=text_bytes,
                    mean_document_tokens=statistics.mean(lengths), median_document_tokens=statistics.median(lengths),
                    mode=mode, workers=workers, active_workers=min(workers, (size + args.batch_size - 1) // args.batch_size) if mode == 'parallel' else 1,
                    result_kind='keywords' if args.top_k else 'full_vectors', top_k=args.top_k,
                    repeats=args.repeats, warmups=args.warmups, median_seconds=median,
                    mean_seconds=statistics.mean(times), stdev_seconds=statistics.stdev(times) if len(times) > 1 else 0,
                    q25_seconds=percentile(times, .25), q75_seconds=percentile(times, .75),
                    min_seconds=min(times), max_seconds=max(times),
                    speedup=statistics.median(base) / median, speedup_ci_low=low, speedup_ci_high=high,
                    efficiency=statistics.median(base) / median / workers,
                    observed_parallel_advantage=(low > 1)))
            write_csv(out / 'summary.csv', summary)
    plot_results(summary, out)
    print(f'Finished: {out.resolve()}', flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
