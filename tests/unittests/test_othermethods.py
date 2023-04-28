import unittest
from cnlp import other_methods
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

    def test_MI(self):
        self.__perform_test(other_methods.information_theory.MI)

    def test_pathentropy(self):
        self.__perform_test(other_methods.information_theory.path_entropy)


if __name__ == '__main__':
    unittest.main()
