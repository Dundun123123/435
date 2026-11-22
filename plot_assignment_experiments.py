"""Plot the three assignment comparisons from measured summary.json files."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', default='results/local')
    parser.add_argument('--output', default='results/local/assignment_experiments.png')
    args = parser.parse_args()
    import matplotlib.pyplot as plt

    root = Path(args.results)
    sizes = (10000, 50000, 90000)
    base = {n: read(root / f'four_mode_imdb_{n}' / 'summary.json') for n in sizes}
    hashes = {item['dataset_sha256'] for item in base.values()}
    if len(hashes) != 1:
        raise ValueError('Different input datasets across experiments')
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=170)
    comparisons = [
        (axes[0], [('cpu-single', 'CPU serial'), ('cpu-multi', 'CPU 4 processes')],
         '1. Serial vs parallel CPU'),
        (axes[1], [('cpu-single', 'CPU serial'), ('gpu-single', 'One-process GPU hybrid')],
         '2. CPU serial vs GPU hybrid'),
    ]
    for ax, modes, title in comparisons:
        for mode, label in modes:
            ax.plot([n / 1000 for n in sizes],
                    [base[n]['summary'][mode]['median_seconds'] for n in sizes],
                    marker='o', label=label)
        ax.set(title=title, xlabel='IMDb reviews (thousands)', ylabel='Median elapsed seconds')
        ax.set_xticks([10, 50, 90])
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)

    distributed = {}
    backend = 'socket' if all((root / f'socket_one_host_{n}' / 'summary.json').exists()
                              for n in sizes) else 'mpi'
    for placement in ('one_host', 'two_hosts'):
        stem = f'{backend}_{placement}'
        if backend == 'socket' and placement == 'two_hosts':
            stem = 'socket_two_host'
        paths = [root / f'{stem}_{n}' / 'summary.json' for n in sizes]
        if all(path.exists() for path in paths):
            distributed[placement] = [read(path) for path in paths]
    ax = axes[2]
    if distributed:
        for placement, label, expected_hosts in (
                ('one_host', 'One PC: 4 processes', 1),
                ('two_hosts', 'Two PCs: 2 + 2 processes', 2)):
            if placement not in distributed:
                continue
            rows = distributed[placement]
            if any(row['status'] != 'passed' or row['host_count'] != expected_hosts or
                   row.get('rank_count', row.get('worker_processes')) != 4 or
                   row['dataset_sha256'] not in hashes or
                   row['documents'] != n for row, n in zip(rows, sizes)):
                raise ValueError(f'Invalid {placement} {backend} result set')
            ax.plot([n / 1000 for n in sizes],
                    [row['median_seconds'] for row in rows], marker='o', label=label)
        ax.legend(fontsize=8)
    if 'two_hosts' not in distributed:
        ax.text(.5, .08, 'Two-host timing pending',
                ha='center', va='center', transform=ax.transAxes,
                bbox={'facecolor': 'white', 'edgecolor': '#CCCCCC'})
    ax.set(title='3. One host vs two hosts', xlabel='IMDb reviews (thousands)',
           ylabel='Median elapsed seconds')
    ax.set_xticks([10, 50, 90])
    ax.grid(alpha=.25)
    figure.tight_layout()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination)
    print(destination.resolve())


if __name__ == '__main__':
    main()
