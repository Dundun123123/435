"""Run the complete study serially; never overlap measured experiments."""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--scales', nargs='+', type=int, default=[10000, 50000, 90000])
    parser.add_argument('--workers', nargs='+', type=int, default=[1, 2, 4, 8])
    parser.add_argument('--output-dir', default='results/study_v3')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    os.chdir(root)
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    cases = [('scale', 'data/imdb.jsonl', args.scales),
             ('short', 'data/imdb_short.jsonl', [5000]),
             ('long', 'data/imdb_long.jsonl', [5000])]
    for name, source, sizes in cases:
        if not Path(source).exists():
            parser.error('Real data missing; run prepare_large_data.py first.')
    for name, source, sizes in cases:
        command = [sys.executable, 'benchmark.py', '--input', source, '--sizes', *map(str, sizes),
            '--workers', *map(str, args.workers), '--repeats', str(args.repeats), '--warmups', str(args.warmups),
            '--top-k', '10', '--output-dir', str(destination / name)]
        print('Running:', ' '.join(command), flush=True)
        subprocess.run(command, check=True)
    subprocess.run([sys.executable, 'summarize_study.py', '--input-dir', str(destination)], check=True)


if __name__ == '__main__':
    main()
