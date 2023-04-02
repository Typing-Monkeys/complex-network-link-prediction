import unittest
from social_network_link_prediction.similarity_methods import dimensionality_reduction
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

    # TODO: ricontrollare
    @unittest.skip("Metodo non ancora implementato")
    def test_svd_3(self):
        g, adjacency = Configs.load_hard_dataset()

        svd = SVD()
        embedding = svd.fit_transform(adjacency)
        our_embedding = dimensionality_reduction.svd(g)

        self.assertEqual(embedding, our_embedding)

    @unittest.skip("Metodo non ancora implementato")
    @timeout(Configs.timeout)
    def test_svd_time(self):
        g, adjacency = Configs.load_hard_dataset()

        our_embedding = dimensionality_reduction.svd(g)

        self.assertIsNotNone(our_embedding)


if __name__ == '__main__':
    unittest.main()
