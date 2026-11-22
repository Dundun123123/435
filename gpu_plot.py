"""Draw a dependency-light SVG timing figure from the raw benchmark CSV."""
import argparse
import csv
import json
import statistics
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir')
    args = parser.parse_args()
    root = Path(args.result_dir)
    metadata = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    with (root / 'raw.csv').open(newline='', encoding='utf-8') as source:
        rows = list(csv.DictReader(source))
    panels = [('Full pipeline', ['cpu_sequential', 'cpu_parallel', 'gpu_hybrid']),
              ('Numeric on the same CSR', ['cpu_numeric', 'gpu_numeric'])]
    colors = {'cpu_sequential': '#477CA8', 'cpu_parallel': '#4F9D91',
              'gpu_hybrid': '#E88937', 'cpu_numeric': '#477CA8',
              'gpu_numeric': '#E88937'}
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="470" '
           'viewBox="0 0 1200 470">',
           '<rect width="1200" height="470" fill="white"/>',
           '<style>text{font-family:Arial,sans-serif;fill:#202735} '
           '.title{font-size:18px;font-weight:bold}.label{font-size:13px} '
           '.small{font-size:11px;fill:#56616e}</style>',
           f'<text x="600" y="28" text-anchor="middle" class="title">'
           f'{escape(Path(metadata["input"]).name)} · {metadata["documents"]:,} documents · '
           'TF-IDF top-k timing</text>']
    for panel_id, (title, modes) in enumerate(panels):
        x0 = 45 + 600 * panel_id
        y0, plot_h, plot_w = 385, 290, 495
        stats = []
        for mode in modes:
            data = [float(row['seconds']) for row in rows
                    if row['backend'] == mode and row['warmup'] == 'False']
            median = statistics.median(data)
            q1, q3 = np.percentile(data, [25, 75])
            stats.append((mode, median, q1, q3))
        scale = plot_h / (max(s[3] for s in stats) * 1.2)
        svg.append(f'<text x="{x0 + plot_w/2:.1f}" y="67" text-anchor="middle" '
                   f'class="title">{escape(title)}</text>')
        for tick in range(5):
            value = tick * max(s[3] for s in stats) * 1.2 / 4
            y = y0 - tick * plot_h / 4
            svg.append(f'<line x1="{x0}" x2="{x0 + plot_w}" y1="{y:.1f}" y2="{y:.1f}" '
                       'stroke="#E6EBEF"/>')
            svg.append(f'<text x="{x0 - 7}" y="{y + 4:.1f}" text-anchor="end" '
                       f'class="small">{value:.2f}s</text>')
        for idx, (mode, median, q1, q3) in enumerate(stats):
            center = x0 + (idx + 0.5) * plot_w / len(stats)
            bar_w = min(90, plot_w / len(stats) * 0.55)
            y = y0 - median * scale
            svg.append(f'<rect x="{center-bar_w/2:.1f}" y="{y:.1f}" '
                       f'width="{bar_w:.1f}" height="{median*scale:.1f}" '
                       f'fill="{colors[mode]}"/>')
            yq1, yq3 = y0 - q1 * scale, y0 - q3 * scale
            svg.append(f'<line x1="{center:.1f}" x2="{center:.1f}" '
                       f'y1="{yq3:.1f}" y2="{yq1:.1f}" stroke="#17202D" stroke-width="2"/>')
            for yy in (yq1, yq3):
                svg.append(f'<line x1="{center-8:.1f}" x2="{center+8:.1f}" '
                           f'y1="{yy:.1f}" y2="{yy:.1f}" stroke="#17202D" stroke-width="2"/>')
            svg.append(f'<text x="{center:.1f}" y="{yq3-9:.1f}" text-anchor="middle" '
                       f'class="label">{median:.3f}s</text>')
            svg.append(f'<text x="{center:.1f}" y="407" text-anchor="middle" '
                       f'class="label">{escape(mode)}</text>')
    svg.append('<text x="600" y="447" text-anchor="middle" class="small">'
               f'Source SHA-256: {metadata["dataset_sha256"][:16]}… · '
               f'{metadata["repeats"]} measured runs after {metadata["warmups"]} warm-up; '
               'whiskers: interquartile range.</text>')
    svg.append('</svg>')
    (root / 'time_comparison.svg').write_text('\n'.join(svg), encoding='utf-8')


if __name__ == '__main__':
    main()
