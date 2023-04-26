import unittest
from social_network_link_prediction.similarity_methods import global_similarity
from tests import Configs
from scipy import sparse
import numpy as np


class TestGlobalSimilarityMethods(unittest.TestCase):

    def __perform_test(self, g, fun, params: dict = {}, debug=False):
        res = fun(g, **params)

        if debug:
            print(res)
            print(type(res))

        self.assertIsNotNone(res, "None result is returned")
        self.assertTrue(
            type(res) is sparse.csr_matrix or type(res) is np.ndarray,
            "Wrong return type")

        return res

    def test_katz_nolabels(self):
        g = Configs.load_normal_dataset()

        self.__perform_test(g, global_similarity.katz_index, {'beta': .001})

    def test_katz_labels(self):
        g = Configs.load_labels_dataset()

        self.__perform_test(g, global_similarity.katz_index, {'beta': .001})

    def test_randwalk_nolabels(self):
        g = Configs.load_normal_dataset()

        self.__perform_test(g, global_similarity.link_prediction_rwr, {
            'c': .05,
            'max_iters': 10
        })

    def test_randwalk_labels(self):
        g = Configs.load_labels_dataset()

        self.__perform_test(g, global_similarity.link_prediction_rwr, {
            'c': .05,
            'max_iters': 10
        })

    def test_rootedpage_nolabels(self):
        g = Configs.load_normal_dataset()

        self.__perform_test(g, global_similarity.rooted_page_rank,
                            {'alpha': .5})

    def test_rootedpage_labels(self):
        g = Configs.load_labels_dataset()

        self.__perform_test(g, global_similarity.rooted_page_rank,
                            {'alpha': .5})

    def test_shortestpath_nolabels(self):
        g = Configs.load_normal_dataset()

        self.__perform_test(g, global_similarity.shortest_path,
                            {'cutoff': None})

    def test_shortestpath_labels(self):
        g = Configs.load_labels_dataset()

        self.__perform_test(g, global_similarity.shortest_path,
                            {'cutoff': None})

    def test_simrank_nolabels(self):
        g = Configs.load_normal_dataset()

        self.__perform_test(g, global_similarity.sim_rank)

    def test_simrank_labels(self):
        g = Configs.load_labels_dataset()

        self.__perform_test(g, global_similarity.sim_rank)

    # @timeout(Configs.timeout)
    # def test_katz_time(self):
    #     g, adjacency = Configs.load_hard_dataset()
    #     beta = .001

    #     res = global_similarity.katz_index(g, beta=beta)
    #     # print(res)

    #     self.assertIsNotNone(res)


if __name__ == '__main__':
    unittest.main()
