import math
import multiprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from tfidf import compute_tfidf, read_documents


def assert_vectors(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        assert a.keys() == b.keys()
        for word in a:
            assert math.isclose(a[word], b[word], rel_tol=1e-10, abs_tol=1e-12), word


def check(texts):
    seq, seq_idf, _ = compute_tfidf(texts, batch_size=2)
    par, par_idf, _ = compute_tfidf(texts, 'parallel', workers=2, batch_size=2)
    assert seq_idf == par_idf
    assert_vectors(seq, par)
    model = TfidfVectorizer(token_pattern=r'[a-z]+', lowercase=True,
                           smooth_idf=True, sublinear_tf=False, norm='l2')
    matrix = model.fit_transform(texts)
    names = model.get_feature_names_out()
    expected = []
    for row in matrix:
        expected.append({str(names[i]): float(v) for i, v in zip(row.indices, row.data)})
    assert_vectors(seq, expected)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    from benchmark import percentile, speedup_interval
    assert percentile([1, 2, 3, 4], .25) == 1.75
    assert speedup_interval([2.] * 10, [1.] * 10) == (2., 2.)
    check(['Apple apple banana', 'banana carrot', '', '123 !!!', 'A I', 'Apple apple banana'])
    check([d['text'] for d in read_documents('data/newsgroups.jsonl', 100)])
    # Batch count may be less than requested workers; preserve original order
    # despite length-based assignment and verify the one-worker backend too.
    texts = ['z ' * 1000, '', 'apple banana', 'a', 'banana ' * 20]
    reference, _, _ = compute_tfidf(texts, batch_size=1)
    for workers in (1, 8):
        result, _, _ = compute_tfidf(texts, 'parallel', workers=workers, batch_size=1)
        assert_vectors(reference, result)
    expected_top = [dict(sorted(v.items(), key=lambda p: (-p[1], p[0]))[:2]) for v in reference]
    for mode in ('sequential', 'parallel'):
        actual, _, _ = compute_tfidf(texts, mode, workers=2, batch_size=1, top_k=2)
        assert_vectors(expected_top, actual)
        diagnostic = {}
        compute_tfidf(texts, mode, workers=2, batch_size=1, profile=diagnostic)
        assert diagnostic['total_seconds'] >= sum(diagnostic['stages'].values())
        assert diagnostic['active_workers'] == (1 if mode == 'sequential' else 2)
    for mode in ('sequential', 'parallel'):
        try:
            compute_tfidf(['', '123 !!!'], mode, workers=2)
        except ValueError:
            pass
        else:
            raise AssertionError('Empty vocabulary should raise ValueError')
    print('PASS: complete sequential/parallel vectors agree with scikit-learn; edge cases pass.')
