import unittest
from social_network_link_prediction.similarity_methods import dimensionality_reduction
import networkx as nx
from sknetwork.data import karate_club
import numpy as np
import networkit as nk


class TestDimensionalityReductionMethods(unittest.TestCase):

    # TODO: ricontrollare
    @unittest.skip("Metodo non ancora implementato")
    def test_svd(self):
        from socketserver.embedding import SVD

        adjacency = karate_club()
        G = nx.from_numpy_array(adjacency)

        svd = SVD()
        embedding = svd.fit_transform(adjacency)
        our_embedding = dimensionality_reduction.svd(G)

        self.assertEqual(embedding, our_embedding)


if __name__ == '__main__':
    unittest.main()
