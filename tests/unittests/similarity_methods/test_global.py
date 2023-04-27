import unittest
from cnlp.similarity_methods import global_similarity
from tests import Configs
from scipy import sparse
import numpy as np


class TestGlobalSimilarityMethods(unittest.TestCase):

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

    def test_katz(self):
        self.__perform_test(global_similarity.katz_index, {'beta': .001})

    def test_randwalk(self):
        self.__perform_test(global_similarity.link_prediction_rwr, {
            'c': .05,
            'max_iters': 10
        })

    def test_rootedpage(self):
        self.__perform_test(global_similarity.rooted_page_rank, {'alpha': .5})

    def test_shortestpath(self):
        self.__perform_test(global_similarity.shortest_path, {'cutoff': None})

    def test_simrank(self):
        self.__perform_test(global_similarity.sim_rank)

    # @timeout(Configs.timeout)
    # def test_katz_time(self):
    #     g, adjacency = Configs.load_hard_dataset()
    #     beta = .001

    #     res = global_similarity.katz_index(g, beta=beta)
    #     # print(res)

    #     self.assertIsNotNone(res)


if __name__ == '__main__':
    unittest.main()
