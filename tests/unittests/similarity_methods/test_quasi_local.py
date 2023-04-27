import unittest
from social_network_link_prediction.similarity_methods import quasi_local_similarity
from tests import Configs
from scipy import sparse
import numpy as np


class TestQuasiGlobalSimilarityMethods(unittest.TestCase):

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

    def test_LPI(self):
        self.__perform_test(quasi_local_similarity.local_path_index, {
            'epsilon': .1,
            'n': 10
        })

    def test_PL3(self):
        self.__perform_test(quasi_local_similarity.path_of_length_three)

    # def setUp(self):
    #     self.start_time = time()

    # def tearDown(self):
    #     t = time() - self.start_time

    #     print(f"{round(t, 2)} s")

    # @unittest.skip("Non è presente in scikit-network")
    # def test_lpi_1(self):
    #     pass

    # @unittest.skip("Non è presente in scikit-network")
    # def test_lpi_2(self):
    #     pass

    # @unittest.skip("Non è presente in scikit-network")
    # def test_lpi_3(self):
    #     pass

    # # TODO: @ncvesvera @fagiolo ricontrollare
    # @timeout(Configs.timeout)
    # def test_lpi_time(self):
    #     g, adjacency = Configs.load_hard_dataset()
    #     epsilon = .1
    #     n = 10

    #     res = quasi_local_similarity.local_path_index(g, epsilon, n)
    #     # print(res)

    #     self.assertIsNotNone(res)

    # @unittest.skip("Non è presente in scikit-network")
    # def test_pol3_1(self):
    #     pass

    # @unittest.skip("Non è presente in scikit-network")
    # def test_pol3_2(self):
    #     pass

    # @unittest.skip("Non è presente in scikit-network")
    # def test_pol3_3(self):
    #     pass

    # @timeout(Configs.timeout)
    # def test_pol3_time(self):
    #     g, adjacency = Configs.load_hard_dataset()

    #     res = quasi_local_similarity.path_of_length_three(g)
    #     # print(res)

    #     self.assertIsNotNone(res)


if __name__ == '__main__':
    unittest.main()
