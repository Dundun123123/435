"""Cross-check the optional GPU backend against the published CPU contract."""
import math
import unittest

from gpu_tfidf import cpu_numeric, prepare_csr, compute_tfidf_gpu
from tfidf import compute_tfidf


TEXTS = ['A a b', 'b c', '', 'z z a', 'a b c', '123 !!!', 'APPLE apple banana']


def assert_same(test, left, right):
    test.assertEqual(len(left), len(right))
    for a, b in zip(left, right):
        test.assertEqual(a.keys(), b.keys())
        for word in a:
            test.assertTrue(math.isclose(a[word], b[word], rel_tol=1e-10, abs_tol=1e-12),
                            (word, a[word], b[word]))


class CSRContract(unittest.TestCase):
    def test_cpu_numeric_on_shared_csr(self):
        expected, idf, _ = compute_tfidf(TEXTS)
        for batch_size in (1, 2, 500):
            csr = prepare_csr(TEXTS, batch_size)
            actual, numeric_idf = cpu_numeric(csr, batch_size)
            assert_same(self, expected, actual)
            self.assertEqual(idf, numeric_idf)
            top_expected, _, _ = compute_tfidf(TEXTS, top_k=2)
            top_actual, _ = cpu_numeric(csr, batch_size, top_k=2)
            assert_same(self, top_expected, top_actual)

    def test_empty_vocabulary(self):
        with self.assertRaisesRegex(ValueError, 'no English tokens'):
            prepare_csr(['', '123 !!!'])


class GPUContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() == 0:
                raise RuntimeError('No CUDA device')
        except Exception as exc:
            raise unittest.SkipTest(f'GPU unavailable: {exc}') from exc

    def test_full_vectors_and_topk(self):
        for batch_size in (1, 2, 500):
            for top_k in (None, 2):
                expected, idf, _ = compute_tfidf(TEXTS, top_k=top_k)
                actual, gpu_idf, _ = compute_tfidf_gpu(TEXTS, batch_size, top_k)
                assert_same(self, expected, actual)
                for word in idf:
                    self.assertTrue(math.isclose(idf[word], gpu_idf[word],
                                                  rel_tol=1e-10, abs_tol=1e-12))


if __name__ == '__main__':
    unittest.main()
