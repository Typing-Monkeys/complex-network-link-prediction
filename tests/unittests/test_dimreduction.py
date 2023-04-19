import unittest
from social_network_link_prediction import dimensionality_reduction_methods
from sknetwork.embedding import SVD
from tests import Configs
import numpy as np
from scipy import sparse


class TestDimensionalityReductionMethods(unittest.TestCase):

    def __perform_test(self, g, fun, params: dict = {}, debug = False):
        res = fun(g, **params)

        if debug:
            print(res)
            print(type(res))
        
        self.assertIsNotNone(res)
        self.assertTrue(type(res) is sparse.csr_matrix or type(res) is np.ndarray)
        
        return res

    def test_SVD_nolabels(self):
        g = Configs.load_normal_dataset()

        self.__perform_test(g, dimensionality_reduction_methods.link_prediction_svd, {'k': 40})

    def test_SVD_labels(self):
        g = Configs.load_labels_dataset()
        
        self.__perform_test(g, dimensionality_reduction_methods.link_prediction_svd, {'k': 40})

    def test_NMF_nolabels(self):
        g = Configs.load_normal_dataset()

        self.__perform_test(g, dimensionality_reduction_methods.link_prediction_nmf, {'num_features': 10, 'num_iterations': 100})

    def test_NMF_labels(self):
        g = Configs.load_labels_dataset()
        
        self.__perform_test(g, dimensionality_reduction_methods.link_prediction_nmf, {'num_features': 10, 'num_iterations': 100})

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
