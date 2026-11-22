"""Check real-corpus full vectors and IDF against the CPU implementation."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from gpu_tfidf import compute_tfidf_gpu
from tfidf import compute_tfidf, read_documents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/newsgroups.jsonl')
    parser.add_argument('--sizes', type=int, nargs='+', default=[100, 1000])
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--output', default='results/local/gpu_correctness.json')
    args = parser.parse_args()
    docs = read_documents(args.input, max(args.sizes))
    checks = []
    for size in args.sizes:
        texts = [doc['text'] for doc in docs[:size]]
        cpu, cpu_idf, _ = compute_tfidf(texts)
        gpu, gpu_idf, _ = compute_tfidf_gpu(texts, args.batch_size)
        max_weight_error = max_idf_error = max_norm_error = 0.0
        for a, b in zip(cpu, gpu):
            if a.keys() != b.keys():
                raise AssertionError(f'{size} documents: vector terms differ')
            for word in a:
                max_weight_error = max(max_weight_error, abs(a[word] - b[word]))
                if not math.isclose(a[word], b[word], rel_tol=1e-10, abs_tol=1e-12):
                    raise AssertionError(f'{size} documents: weight differs for {word}')
            if b:
                max_norm_error = max(max_norm_error,
                                     abs(math.sqrt(sum(v * v for v in b.values())) - 1))
        for word in cpu_idf:
            max_idf_error = max(max_idf_error, abs(cpu_idf[word] - gpu_idf[word]))
            if not math.isclose(cpu_idf[word], gpu_idf[word], rel_tol=1e-10, abs_tol=1e-12):
                raise AssertionError(f'{size} documents: IDF differs for {word}')
        checks.append({'documents': size, 'status': 'passed', 'vocabulary': len(cpu_idf),
                       'max_abs_weight_error': max_weight_error,
                       'max_abs_idf_error': max_idf_error,
                       'max_l2_norm_error': max_norm_error})
    report = {'status': 'passed', 'input': args.input,
              'dataset_sha256': hashlib.sha256(Path(args.input).read_bytes()).hexdigest(),
              'rtol': 1e-10, 'atol': 1e-12, 'checks': checks}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
