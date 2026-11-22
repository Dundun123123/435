"""One CLI and output format for CPU/GPU single and multiple process runs."""
import argparse
import json
from pathlib import Path

from gpu_multiprocess import compute_tfidf_gpu_multiprocess
from gpu_tfidf import compute_tfidf_gpu
from tfidf import compute_tfidf, read_documents


def run_mode(texts, mode, workers, batch_size, top_k, profile):
    if mode == 'cpu-single':
        return compute_tfidf(texts, top_k=top_k, profile=profile)
    if mode == 'cpu-multi':
        return compute_tfidf(texts, mode='parallel', workers=workers,
                             batch_size=batch_size, top_k=top_k, profile=profile)
    if mode == 'gpu-single':
        return compute_tfidf_gpu(texts, batch_size, top_k, profile)
    return compute_tfidf_gpu_multiprocess(texts, workers, batch_size, top_k, profile)


def main():
    parser = argparse.ArgumentParser(description='Compare the same TF-IDF task in four modes')
    parser.add_argument('--mode', choices=('cpu-single', 'cpu-multi', 'gpu-single', 'gpu-multi'), required=True)
    parser.add_argument('--input', default='data/sample.jsonl')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--output', default='results/local/four_mode_output.jsonl')
    args = parser.parse_args()
    docs = read_documents(args.input, args.limit)
    profile = {}
    vectors, idf, seconds = run_mode([d['text'] for d in docs], args.mode,
                                     args.workers, args.batch_size, args.top_k, profile)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as file:
        for doc, vector in zip(docs, vectors):
            words = sorted(vector.items(), key=lambda item: (-item[1], item[0]))
            file.write(json.dumps({'id': doc['id'], 'keywords': [
                {'term': word, 'weight': weight} for word, weight in words]},
                ensure_ascii=False) + '\n')
    summary = {'mode': args.mode, 'documents': len(docs), 'vocabulary': len(idf),
               'workers': args.workers if args.mode.endswith('multi') else 1,
               'seconds': seconds, 'profile': profile,
               'preview': {'id': docs[0]['id'], 'keywords': list(sorted(vectors[0].items(),
                           key=lambda item: (-item[1], item[0])))[:3]},
               'output': str(output)}
    output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"Mode: {args.mode} | workers: {summary['workers']} | documents: {len(docs):,}")
    print(f"Vocabulary: {len(idf):,} | elapsed: {seconds:.6f} s")
    print(f"{docs[0]['id']}: " + ', '.join(f'{w}={v:.6f}' for w, v in summary['preview']['keywords']))
    if args.mode == 'gpu-multi':
        print(f"Active GPU processes: {profile['active_gpu_processes']}")
    print(f'Output: {output}')


if __name__ == '__main__':
    main()
