"""Create readable results and phase plots without rerunning benchmarks."""
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', default='results/study_v3')
    args = parser.parse_args()
    root = Path(args.input_dir)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lines = ['# TF-IDF v3 实测研究', '',
        '同一套TF-IDF公式、相同输入与相同输出任务（每篇top-10关键词）。串行和并行均在计时内完成关键词选择。',
        '每个配置的预热次数及正式重复次数见各子目录environment.json。预热不纳入统计，但每次调用仍重新创建进程，启动成本始终计时。',
        '时间误差条为样本四分位区间IQR，并非置信区间；加速比区间为1000次按轮次配对bootstrap的95%百分位区间。',
        '这些区间只描述本次重复实验的不确定性，不保证其他电脑或其他运行条件下的结果。', '']
    for case in ('scale', 'short', 'long'):
        with (root / case / 'summary.csv').open(encoding='utf-8') as source:
            rows = list(csv.DictReader(source))
        lines += [f'## {case}', '', '|文档数|总词数|正文MiB|平均词数/篇|模式|进程数|中位秒数|IQR秒|加速比|95%区间|效率|',
                  '|---|---|---|---|---|---|---|---|---|---|---|']
        for r in rows:
            lines.append(f"|{r['documents']}|{r['tokens']}|{float(r['text_utf8_bytes'])/1024**2:.1f}|{float(r['mean_document_tokens']):.1f}|{r['mode']}|{r['workers']}|{float(r['median_seconds']):.3f}|{float(r['q25_seconds']):.3f}–{float(r['q75_seconds']):.3f}|{float(r['speedup']):.2f}|{float(r['speedup_ci_low']):.2f}–{float(r['speedup_ci_high']):.2f}|{float(r['efficiency'])*100:.1f}%|")
        # Parent stage means add up to mean pipeline time (apart from outer cleanup),
        # unlike sums of medians or sums of overlapping worker durations.
        stages = defaultdict(list)
        size = max(int(r['documents']) for r in rows)
        with (root / case / 'phases.csv').open(encoding='utf-8') as source:
            for r in csv.DictReader(source):
                if int(r['documents']) == size:
                    stages[(r['mode'], int(r['workers']), r['stage'])].append(float(r['seconds']))
        configs = [('sequential', 1)] + [('parallel', int(r['workers'])) for r in rows if int(r['documents']) == size and r['mode'] == 'parallel']
        names = list(dict.fromkeys(key[2] for key in stages))
        fig, ax = plt.subplots(figsize=(11, 6))
        bottom = [0.] * len(configs)
        for stage in names:
            values = [statistics.mean(stages[(mode, workers, stage)]) if (mode, workers, stage) in stages else 0
                      for mode, workers in configs]
            ax.bar(range(len(configs)), values, bottom=bottom, label=stage.removesuffix('_seconds'))
            bottom = [a + b for a, b in zip(bottom, values)]
        ax.set_xticks(range(len(configs)), ['seq' if m == 'sequential' else f'p{w}' for m, w in configs])
        ax.set(ylabel='Mean parent wall intervals (s)', title=f'{case}: {size:,} documents; overlapping worker times excluded')
        ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1, 1))
        fig.tight_layout()
        fig.savefig(root / case / 'stage_breakdown.png', dpi=160)
        plt.close(fig)
    lines += ['', '## 如何解释', '',
        '加速比小于1表示并行更慢；区间跨过1时，不能把微小的中位数优势描述为稳定加速。不要挑某一次最快结果代替10次的分布。',
        'short/long来自同一真实数据集的不同长度四分位，各随机抽取5000篇，无重复扩展；内容和词汇也随长度变化，不能解释成只改变长度的因果实验。',
        'profiles.jsonl保存主进程阶段及各工作进程观察值。主进程阶段包含等待与通信，工作进程计算与这些阶段重叠，不能直接相加或称为精确的纯通信耗时。',
        '输入文件大小包含JSON元数据；表内正文MiB仅为该次选中正文的UTF-8字节数。使用同一已打乱语料的前缀做规模实验，固定种子42。']
    (root / 'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    with (root / 'scale' / 'summary.csv').open(encoding='utf-8') as source:
        scale = list(csv.DictReader(source))
    configs = [('sequential', 1)] + [('parallel', int(r['workers'])) for r in scale
        if int(r['documents']) == min(int(x['documents']) for x in scale) and r['mode'] == 'parallel']
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for mode, workers in configs:
        selected = sorted([r for r in scale if r['mode'] == mode and int(r['workers']) == workers], key=lambda r: int(r['documents']))
        medians = [float(r['median_seconds']) for r in selected]
        ax.errorbar([int(r['documents']) for r in selected], medians,
            yerr=[[v - float(r['q25_seconds']) for r, v in zip(selected, medians)],
                  [float(r['q75_seconds']) - v for r, v in zip(selected, medians)]],
            marker='o', capsize=4, label='sequential' if mode == 'sequential' else f'parallel {workers}')
    ax.set(xlabel='Documents', ylabel='Execution time (s): median and IQR', ylim=(0, None))
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / 'time_vs_size.png', dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for shift, case, color in [(-.18, 'short', '#3274a1'), (.18, 'long', '#e1812c')]:
        with (root / case / 'summary.csv').open(encoding='utf-8') as source:
            by_config = {(r['mode'], int(r['workers'])): r for r in csv.DictReader(source)}
        selected = [by_config[c] for c in configs]
        medians = [float(r['median_seconds']) for r in selected]
        ax.bar([i + shift for i in range(len(configs))], medians, width=.36, color=color,
            yerr=[[v - float(r['q25_seconds']) for r, v in zip(selected, medians)],
                  [float(r['q75_seconds']) - v for r, v in zip(selected, medians)]],
            capsize=3, label=f"{case}: {float(selected[0]['mean_document_tokens']):.0f} mean tokens/doc")
    ax.set_xticks(range(len(configs)), ['seq' if m == 'sequential' else f'p{w}' for m, w in configs])
    ax.set(ylabel='Execution time (s): median and IQR', title='Equal document counts: 5,000 real reviews per cohort')
    ax.legend()
    fig.tight_layout()
    fig.savefig(root / 'length_comparison.png', dpi=170)
    plt.close(fig)
    print(f'Report: {(root / "REPORT.md").resolve()}')


if __name__ == '__main__':
    main()
