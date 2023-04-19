import unittest
from social_network_link_prediction import other_methods
from tests import Configs
import numpy as np
from scipy import sparse


class TestDimensionalityReductionMethods(unittest.TestCase):

    def __perform_test(self, g, fun, params: dict = {}, debug = False):
        res = fun(g, **params)

        if debug:
            print(res)
            print(type(res))
        
        self.assertIsNotNone(res, "None result is returned")
        self.assertTrue(type(res) is sparse.csr_matrix or type(res) is np.ndarray, "Wrong return type")
        
        return res

    def test_MI_nolabels(self):
        g = Configs.load_normal_dataset()

        self.__perform_test(g, other_methods.MI, debug=True)

    def test_MI_labels(self):
        g = Configs.load_labels_dataset()
        
        self.__perform_test(g, other_methods.MI)


if __name__ == '__main__':
    unittest.main()
