"""Run the optional hybrid CPU/CUDA TF-IDF pipeline."""
import argparse
import hashlib
import json
from pathlib import Path

from gpu_tfidf import compute_tfidf_gpu
from tfidf import read_documents


def main():
    parser = argparse.ArgumentParser(description='CPU text preprocessing + GPU sparse TF-IDF')
    parser.add_argument('--input', default='data/sample.jsonl')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--output', default='results/local/gpu_keywords.jsonl')
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error('--top-k must be positive')
    docs = read_documents(args.input, args.limit)
    profile = {}
    vectors, idf, seconds = compute_tfidf_gpu(
        [doc['text'] for doc in docs], args.batch_size,
        None if args.full else args.top_k, profile)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as dest:
        for doc, vector in zip(docs, vectors):
            top = sorted(vector.items(), key=lambda pair: (-pair[1], pair[0]))[:args.top_k]
            row = {'id': doc['id'], 'keywords': [{'term': term, 'weight': weight}
                                                  for term, weight in top]}
            if args.full:
                row['vector'] = vector
            dest.write(json.dumps(row, ensure_ascii=False) + '\n')
    import cupy as cp
    props = cp.cuda.runtime.getDeviceProperties(0)
    name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
    summary = {'status': 'passed', 'implementation': 'cpu-tokenization-gpu-sparse-numeric-v1',
               'input': args.input, 'input_sha256': hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
               'documents': len(docs), 'vocabulary': len(idf), 'top_k': None if args.full else args.top_k,
               'seconds': seconds, 'profile': profile, 'gpu_name': name,
               'gpu_memory_bytes': int(props['totalGlobalMem']), 'cupy': cp.__version__,
               'cuda_runtime': cp.cuda.runtime.runtimeGetVersion(),
               'driver_version': cp.cuda.runtime.driverGetVersion(), 'output': str(output)}
    output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
