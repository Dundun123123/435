"""Run after benchmarks, so correctness checks cannot disturb timed trials."""
import json
import math
import multiprocessing
from pathlib import Path
from tfidf import compute_tfidf, read_documents


def equal(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        assert a.keys() == b.keys()
        assert all(math.isclose(a[w], b[w], rel_tol=1e-10, abs_tol=1e-12) for w in a)


def main():
    import verify
    # Use the reference library on independent subsets, including length cohorts.
    for name in ('imdb', 'imdb_short', 'imdb_long'):
        verify.check([d['text'] for d in read_documents(f'data/{name}.jsonl', 100)])
    texts = [d['text'] for d in read_documents('data/imdb.jsonl')]
    seq, si, _ = compute_tfidf(texts, top_k=10)
    parallel, pi, _ = compute_tfidf(texts, 'parallel', workers=4, top_k=10)
    assert si == pi
    equal(seq, parallel)
    report = {'implementation': 'packed-shard-processes-v3', 'documents': len(texts),
        'all_top10_weights_match': True, 'reference_library_checks': '100 docs each: full IMDb, short, long',
        'rel_tol': 1e-10, 'abs_tol': 1e-12}
    Path('results/large_validation_v3.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
