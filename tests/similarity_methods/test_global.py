import unittest
from social_network_link_prediction.similarity_methods import global_similarity
from timeout_decorator import timeout
from tests import Configs
from time import time


class TestGlobalSimilarityMethods(unittest.TestCase):

    def setUp(self):
        self.start_time = time()

    def tearDown(self):
        t = time() - self.start_time

        print(f"{round(t, 2)} s")

    @unittest.skip(
        "Il metodo di Networkit differisce nell'implementazione dal nostro")
    def test_katz_1(self):
        pass

    @unittest.skip(
        "Il metodo di Networkit differisce nell'implementazione dal nostro")
    def test_katz_2(self):
        pass

    @unittest.skip(
        "Il metodo di Networkit differisce nell'implementazione dal nostro")
    def test_katz_3(self):
        pass

    @timeout(Configs.timeout)
    def test_katz_time(self):
        g, adjacency = Configs.load_hard_dataset()
        beta = .001

        res = global_similarity.katz_index(g, beta=beta)
        # print(res)

        self.assertIsNotNone(res)


if __name__ == '__main__':
    unittest.main()
