import unittest
from social_network_link_prediction.similarity_methods import quasi_local_similarity
from timeout_decorator import timeout
from tests import Configs
from time import time


class TestQuasiGlobalSimilarityMethods(unittest.TestCase):

    def setUp(self):
        self.start_time = time()

    def tearDown(self):
        t = time() - self.start_time

        print(f"{round(t, 2)} s")

    @unittest.skip("Non è presente in scikit-network")
    def test_lpi_1(self):
        pass

    @unittest.skip("Non è presente in scikit-network")
    def test_lpi_2(self):
        pass

    @unittest.skip("Non è presente in scikit-network")
    def test_lpi_3(self):
        pass

    # TODO: @ncvesvera @fagiolo ricontrollare
    @timeout(Configs.timeout)
    def test_lpi_time(self):
        g, adjacency = Configs.load_hard_dataset()
        epsilon = .1
        n = 4

        res = quasi_local_similarity.local_path_index(g, epsilon, n)
        # print(res)

        self.assertIsNotNone(res)

    @unittest.skip("Non è presente in scikit-network")
    def test_pol3_1(self):
        pass

    @unittest.skip("Non è presente in scikit-network")
    def test_pol3_2(self):
        pass

    @unittest.skip("Non è presente in scikit-network")
    def test_pol3_3(self):
        pass

    @timeout(Configs.timeout)
    def test_pol3_time(self):
        g, adjacency = Configs.load_hard_dataset()

        res = quasi_local_similarity.path_of_length_three(g)
        # print(res)

        self.assertIsNotNone(res)


if __name__ == '__main__':
    unittest.main()
