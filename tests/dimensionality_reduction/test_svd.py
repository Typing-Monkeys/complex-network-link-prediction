import unittest
from social_network_link_prediction import dimensionality_reduction_methods
from sknetwork.embedding import SVD
from tests import Configs
from time import time
from timeout_decorator import timeout


class TestDimensionalityReductionMethods(unittest.TestCase):

    def setUp(self):
        self.start_time = time()

    def tearDown(self):
        t = time() - self.start_time

        print(f"{round(t, 2)} s")

    # TODO: @ncvesvera @Cosci @posta ricontrollare
    # @unittest.skip("Metodo non ancora implementato")
    def test_svd_3(self):
        g, adjacency = Configs.load_hard_dataset()
        k = 100

        svd = SVD(n_components=k)

        embedding = svd.fit_transform(adjacency)
        our_embedding = dimensionality_reduction_methods.link_prediction_svd(
            g, k=k)

        print()
        print(svd.predict(adjacency))
        print()
        print()
        print(embedding)
        print(our_embedding)

        self.assertEqual(embedding, our_embedding)

    @timeout(Configs.timeout)
    def test_svd_time(self):
        g, adjacency = Configs.load_hard_dataset()
        k = 100
        our_embedding = dimensionality_reduction_methods.link_prediction_svd(
            g, k=k)

        self.assertIsNotNone(our_embedding)


if __name__ == '__main__':
    unittest.main()
