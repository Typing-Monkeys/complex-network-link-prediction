import unittest
from social_network_link_prediction.similarity_methods import local_similarity
import networkx as nx
from sknetwork.data import house
import random


class TestLocalSimilarityMethods(unittest.TestCase):

    def test_common_neighbors(self):
        from sknetwork.linkpred import CommonNeighbors

        adjacency = house()
        cn = CommonNeighbors()

        cn.fit(adjacency)
        our_sim = local_similarity.common_neighbors(
            nx.from_numpy_array(adjacency))

        indxes = [(random.randrange(adjacency.shape[0]),
                   random.randrange(adjacency.shape[0])) for a in range(20)]

        for i, j in indxes:
            expected = cn.predict((i, j))
            our_res = our_sim[i, j]

            self.assertEqual(expected, our_res)

    def test_jaccard(self):
        from sknetwork.linkpred import JaccardIndex

        adjacency = house()
        cn = JaccardIndex()

        cn.fit(adjacency)
        our_sim = local_similarity.jaccard(nx.from_numpy_array(adjacency))

        indxes = [(random.randrange(adjacency.shape[0]),
                   random.randrange(adjacency.shape[0])) for a in range(20)]

        for i, j in indxes:
            expected = cn.predict((i, j))
            our_res = our_sim[i, j]

            self.assertEqual(expected.round(2), our_res.round(2))

    def test_sorensen(self):
        from sknetwork.linkpred import SorensenIndex

        adjacency = house()
        cn = SorensenIndex()

        cn.fit(adjacency)
        our_sim = local_similarity.sorensen(nx.from_numpy_array(adjacency))

        indxes = [(random.randrange(adjacency.shape[0]),
                   random.randrange(adjacency.shape[0])) for a in range(20)]

        for i, j in indxes:
            expected = cn.predict((i, j))
            our_res = our_sim[i, j]

            self.assertEqual(expected.round(2), our_res.round(2))

    def test_hubpromoted(self):
        from sknetwork.linkpred import HubPromotedIndex

        adjacency = house()
        cn = HubPromotedIndex()

        cn.fit(adjacency)
        our_sim = local_similarity.hub_promoted(nx.from_numpy_array(adjacency))

        indxes = [(random.randrange(adjacency.shape[0]),
                   random.randrange(adjacency.shape[0])) for a in range(20)]

        for i, j in indxes:
            expected = cn.predict((i, j))
            our_res = our_sim[i, j]

            self.assertEqual(expected.round(2), our_res.round(2))

    def test_hubdepressed(self):
        from sknetwork.linkpred import HubDepressedIndex

        adjacency = house()
        cn = HubDepressedIndex()

        cn.fit(adjacency)
        our_sim = local_similarity.hub_depressed(
            nx.from_numpy_array(adjacency))

        indxes = [(random.randrange(adjacency.shape[0]),
                   random.randrange(adjacency.shape[0])) for a in range(20)]

        for i, j in indxes:
            expected = cn.predict((i, j))
            our_res = our_sim[i, j]

            self.assertEqual(expected.round(2), our_res.round(2))

    def test_adamicadar(self):
        from sknetwork.linkpred import AdamicAdar

        adjacency = house()
        cn = AdamicAdar()

        cn.fit(adjacency)
        our_sim = local_similarity.adamic_adar(nx.from_numpy_array(adjacency))

        indxes = [(random.randrange(adjacency.shape[0]),
                   random.randrange(adjacency.shape[0])) for a in range(20)]

        for i, j in indxes:
            expected = cn.predict((i, j))
            our_res = our_sim[i, j]

            self.assertEqual(expected.round(2), our_res.round(2))

    def test_resourceallocation(self):
        from sknetwork.linkpred import ResourceAllocation

        adjacency = house()
        cn = ResourceAllocation()

        cn.fit(adjacency)
        our_sim = local_similarity.resource_allocation(
            nx.from_numpy_array(adjacency))

        indxes = [(random.randrange(adjacency.shape[0]),
                   random.randrange(adjacency.shape[0])) for a in range(20)]

        for i, j in indxes:
            expected = cn.predict((i, j))
            our_res = our_sim[i, j]

            self.assertEqual(expected.round(2), our_res.round(2))

    def test_preferentialattachment(self):
        from sknetwork.linkpred import PreferentialAttachment

        adjacency = house()
        cn = PreferentialAttachment()

        cn.fit(adjacency)
        our_sim = local_similarity.preferential_attachment(
            nx.from_numpy_array(adjacency))

        indxes = [(random.randrange(adjacency.shape[0]),
                   random.randrange(adjacency.shape[0])) for a in range(20)]

        for i, j in indxes:
            expected = cn.predict((i, j))
            our_res = our_sim[i, j]

            self.assertEqual(expected.round(2), our_res.round(2))

    @unittest.skip("Non è presente in scikit-network")
    def test_cosine(self):
        pass

    @unittest.skip("Non è presente in scikit-network")
    def test_nodeclustering(self):
        pass


if __name__ == '__main__':
    unittest.main()
