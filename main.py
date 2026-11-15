import argparse
import json
import multiprocessing
from pathlib import Path
from time import perf_counter
from tfidf import TOKEN, compute_tfidf, read_documents


def main():
    run_start = perf_counter()
    parser = argparse.ArgumentParser(description='Sequential or parallel document TF-IDF')
    parser.add_argument('--input', default='data/sample.jsonl')
    parser.add_argument('--mode', choices=['sequential', 'parallel'], default='sequential')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--output', default='results/keywords.jsonl')
    parser.add_argument('--full', action='store_true', help='Include full sparse vectors in output')
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error('--top-k must be positive')
    read_start = perf_counter()
    docs = read_documents(args.input, args.limit)
    read_seconds = perf_counter() - read_start
    diagnostics_start = perf_counter()
    # Corpus diagnostics are deliberately outside the computation timer.
    token_count = sum(len(TOKEN.findall(d['text'].lower())) for d in docs)
    text_bytes = sum(len(d['text'].encode('utf-8')) for d in docs)
    diagnostics_seconds = perf_counter() - diagnostics_start
    profile = {}
    vectors, idf, seconds = compute_tfidf([d['text'] for d in docs], args.mode, args.workers, args.batch_size,
                                        top_k=None if args.full else args.top_k, profile=profile)
    output = Path(args.output)
    output_start = perf_counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    preview = []
    with output.open('w', encoding='utf-8') as dest:
        for doc, vector in zip(docs, vectors):
            top = sorted(vector.items(), key=lambda pair: (-pair[1], pair[0]))[:args.top_k]
            row = {'id': doc['id'], 'keywords': [{'term': w, 'weight': v} for w, v in top]}
            if args.full:
                row['vector'] = vector
            dest.write(json.dumps(row, ensure_ascii=False) + '\n')
            if len(preview) < 3:
                preview.append(row)
    output_seconds = perf_counter() - output_start
    summary = {'mode': args.mode, 'workers': args.workers if args.mode == 'parallel' else 1,
               'implementation': 'packed-shard-processes-v3',
               'result_kind': 'full_vectors' if args.full else 'top_k_keywords',
               'top_k': None if args.full else args.top_k,
               'active_workers': min(args.workers, (len(docs) + args.batch_size - 1) // args.batch_size) if args.mode == 'parallel' else 1,
               'batch_size': args.batch_size, 'documents': len(docs), 'vocabulary': len(idf),
               'tokens': token_count, 'input_file_bytes': Path(args.input).stat().st_size,
               'selected_text_utf8_bytes': text_bytes,
               'seconds': seconds, 'input_read_seconds': read_seconds,
               'diagnostics_seconds': diagnostics_seconds, 'output_write_seconds': output_seconds,
               'through_output_seconds': perf_counter() - run_start,
               'output': str(output)}
    summary['profile'] = profile
    output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    console_summary = dict(summary)
    console_summary['profile'] = {key: value for key, value in profile.items() if key != 'workers'}
    print(json.dumps(console_summary, indent=2))
    for row in preview:
        print(row['id'], ', '.join(f"{x['term']}={x['weight']:.6f}" for x in row['keywords']))


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
