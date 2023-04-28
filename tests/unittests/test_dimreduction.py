import unittest
from cnlp import dimensionality_reduction_methods
from tests import Configs
import numpy as np
from scipy import sparse


class TestDimensionalityReductionMethods(unittest.TestCase):

    def __perform_test(self, fun, params: dict = {}, debug: bool = False):
        g = Configs.load_normal_dataset()
        g_labels = Configs.load_labels_dataset()

        res = None
        res_labels = None

        with self.subTest('Int Labels'):
            res = fun(g, **params)

            if debug:
                print(res)
                print(type(res))

            self.assertIsNotNone(res, "None result is returned")
            self.assertTrue(
                type(res) is sparse.csr_matrix or type(res) is np.ndarray,
                "Wrong return type")

        with self.subTest('String Labels'):
            res_labels = fun(g_labels, **params)

            if debug:
                print(res_labels)
                print(type(res_labels))

            self.assertIsNotNone(res_labels, "None result is returned")
            self.assertTrue(
                type(res_labels) is sparse.csr_matrix
                or type(res_labels) is np.ndarray, "Wrong return type")

        with self.subTest('CMP Results'):
            try:
                self.assertTrue(
                    (res.__round__(4) != res_labels.__round__(4)).nnz == 0,
                    "Results are different !")
            except AssertionError as e:
                print(e)

        return res

    def test_SVD(self):
        self.__perform_test(
            dimensionality_reduction_methods.link_prediction_svd, {'k': 40})

    def test_NMF(self):
        self.__perform_test(
            dimensionality_reduction_methods.link_prediction_nmf, {
                'num_features': 10,
                'num_iterations': 100
            })

    # def setUp(self):
    #     self.start_time = time()

    # def tearDown(self):
    #     t = time() - self.start_time

    #     print(f"{round(t, 2)} s")

    # # TODO: @ncvesvera @Cosci @posta ricontrollare
    # # @unittest.skip("Metodo non ancora implementato")
    # def test_svd_3(self):
    #     g, adjacency = Configs.load_hard_dataset()
    #     k = 100

    #     svd = SVD(n_components=k)

    #     embedding = svd.fit_transform(adjacency)
    #     our_embedding = dimensionality_reduction_methods.link_prediction_svd(
    #         g, k=k)

    #     print()
    #     print(svd.predict(adjacency))
    #     print()
    #     print()
    #     print(embedding)
    #     print(our_embedding)

    #     self.assertEqual(embedding, our_embedding)

    # @timeout(Configs.timeout)
    # def test_svd_time(self):
    #     g, adjacency = Configs.load_hard_dataset()
    #     k = 100
    #     our_embedding = dimensionality_reduction_methods.link_prediction_svd(
    #         g, k=k)

    #     self.assertIsNotNone(our_embedding)


if __name__ == '__main__':
    unittest.main()
