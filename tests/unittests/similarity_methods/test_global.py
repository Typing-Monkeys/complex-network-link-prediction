import unittest
from social_network_link_prediction.similarity_methods import global_similarity
from tests import Configs
from scipy import sparse
import numpy as np


class TestGlobalSimilarityMethods(unittest.TestCase):

    def test_katz_nolabels(self):
        g = Configs.load_normal_dataset()
        beta = .001

        res = global_similarity.katz_index(g, beta=beta)
        
        with self.subTest():
            self.assertIsNotNone(res)
        with self.subTest():
            self.assertTrue(type(res) is sparse.csr_matrix or type(res) is np.ndarray)

    def test_katz_labels(self):
        g = Configs.load_labels_dataset()
        beta = .001

        res = global_similarity.katz_index(g, beta=beta)
        
        with self.subTest():
            self.assertIsNotNone(res)
        with self.subTest():
            self.assertTrue(type(res) is sparse.csr_matrix or type(res) is np.ndarray)

    def test_randwalk_nolabels(self):
        g = Configs.load_normal_dataset()
        c = .05
        max_iters = 10

        res = global_similarity.link_prediction_rwr(g, c = c, max_iters=max_iters)

        with self.subTest():
            self.assertIsNotNone(res)
        with self.subTest():
            self.assertTrue(type(res) is sparse.csr_matrix or type(res) is np.ndarray)
            
    def test_randwalk_labels(self):
        g = Configs.load_labels_dataset()
        c = .05
        max_iters = 10

        res = global_similarity.link_prediction_rwr(g, c = c, max_iters=max_iters)
        
        with self.subTest():
            self.assertIsNotNone(res)
        with self.subTest():
            self.assertTrue(type(res) is sparse.csr_matrix or type(res) is np.ndarray)

    def test_rootedpage_nolabels(self):
        g = Configs.load_normal_dataset()
        alpha = .5

        res = global_similarity.rooted_page_rank(g, alpha=alpha)
        
        with self.subTest():
            self.assertIsNotNone(res)
        with self.subTest():
            self.assertTrue(type(res) is sparse.csr_matrix or type(res) is np.ndarray)

    def test_rootedpage_labels(self):
        g = Configs.load_labels_dataset()
        alpha = .5

        res = global_similarity.rooted_page_rank(g, alpha=alpha)
        
        with self.subTest():
            self.assertIsNotNone(res)
        with self.subTest():
            self.assertTrue(type(res) is sparse.csr_matrix or type(res) is np.ndarray)
    
    def test_shortestpath_nolabels(self):
        g = Configs.load_normal_dataset()
        cutoff = None

        res = global_similarity.shortest_path(g, cutoff=cutoff)
        
        with self.subTest():
            self.assertIsNotNone(res)
        with self.subTest():
            self.assertTrue(type(res) is sparse.csr_matrix or type(res) is np.ndarray)

    def test_shortestpath_labels(self):
        g = Configs.load_labels_dataset()
        cutoff = None

        res = global_similarity.shortest_path(g, cutoff=cutoff)
        
        with self.subTest():
            self.assertIsNotNone(res)
        with self.subTest():
            self.assertTrue(type(res) is sparse.csr_matrix or type(res) is np.ndarray)
    # @timeout(Configs.timeout)
    # def test_katz_time(self):
    #     g, adjacency = Configs.load_hard_dataset()
    #     beta = .001

    #     res = global_similarity.katz_index(g, beta=beta)
    #     # print(res)

    #     self.assertIsNotNone(res)


if __name__ == '__main__':
    unittest.main()
