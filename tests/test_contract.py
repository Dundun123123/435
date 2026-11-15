"""Small, deterministic tests: no dataset download, sklearn, MPI or GPU needed."""
import math
import unittest
from tfidf import compute_tfidf


class CPUContract(unittest.TestCase):
    def assert_vectors(self, left, right):
        self.assertEqual(len(left), len(right))
        for a, b in zip(left, right):
            self.assertEqual(a.keys(), b.keys())
            for word in a:
                self.assertTrue(math.isclose(a[word], b[word], rel_tol=1e-10, abs_tol=1e-12))

    def test_known_weights_and_empty_document_count(self):
        vectors, idf, _ = compute_tfidf(['A a b', 'b c', ''])
        expected_a = math.log(4 / 2) + 1
        expected_b = math.log(4 / 3) + 1
        self.assertAlmostEqual(idf['a'], expected_a)
        self.assertAlmostEqual(idf['b'], expected_b)
        norm = math.sqrt((2 * expected_a) ** 2 + expected_b ** 2)
        self.assertAlmostEqual(vectors[0]['a'], 2 * expected_a / norm)
        self.assertEqual(vectors[2], {})

    def test_parallel_order_topk_and_profile(self):
        texts = ['z ' * 1000, '', 'apple banana apple', 'a b c', 'banana ' * 20]
        full, _, _ = compute_tfidf(texts)
        expected = [dict(sorted(v.items(), key=lambda x: (-x[1], x[0]))[:2]) for v in full]
        for workers in (1, 2, 8):
            profile = {}
            actual, _, elapsed = compute_tfidf(texts, 'parallel', workers, batch_size=1, top_k=2, profile=profile)
            self.assert_vectors(expected, actual)
            self.assertEqual(profile['active_workers'], min(workers, len(texts)))
            self.assertGreaterEqual(elapsed, sum(profile['stages'].values()))
        self.assertEqual(set(expected[3]), {'a', 'b'})  # Equal weights: alphabetic tie break.

    def test_all_empty_vocabulary_and_invalid_parameters(self):
        for mode in ('sequential', 'parallel'):
            with self.assertRaises(ValueError):
                compute_tfidf(['', '123 !!!'], mode, workers=2)
        with self.assertRaises(ValueError):
            compute_tfidf(['ok'], mode='gpu')  # Not implemented: must not silently fall back.
        with self.assertRaises(ValueError):
            compute_tfidf(['ok'], batch_size=0)


if __name__ == '__main__':
    unittest.main()
