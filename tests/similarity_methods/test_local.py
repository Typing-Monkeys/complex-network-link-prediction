import networkx as nx
import unittest
import random
from social_network_link_prediction.similarity_methods import local_similarity
from sknetwork.data import house, karate_club, load_konect
from sknetwork.linkpred import CommonNeighbors
from sknetwork.linkpred import JaccardIndex


class TestLocalSimilarityMethods(unittest.TestCase):
    __dataset_easy = house()
    __dataset_medium = karate_club()
    __dataset_hard = load_konect('ego-facebook', verbose=False)["adjacency"]

    def __load_easy_dataset(self):
        adj = self.__dataset_easy
        g = nx.from_numpy_array(adj)

        return g, adj

    def __load_medium_dataset(self):
        adj = self.__dataset_medium
        g = nx.from_numpy_array(adj)

        return g, adj

    def __load_hard_dataset(self):
        adj = self.__dataset_hard
        g = nx.from_numpy_array(adj)

        return g, adj

    def __perform_sktest(self,
                         our_values,
                         skfunction,
                         samples_range,
                         samples_number=20):
        indxes = [(random.randrange(samples_range),
                   random.randrange(samples_range))
                  for a in range(samples_number)]

        for i, j in indxes:
            expected = skfunction.predict((i, j))
            our_res = our_values[i, j]

            self.assertEqual(expected.round(2), our_res.round(2))

    def test_common_neighbors_1(self):
        g, adjacency = self.__load_easy_dataset()
        cn = CommonNeighbors()

        cn.fit(adjacency)
        our_sim = local_similarity.common_neighbors(g)

        self.__perform_sktest(our_sim, cn, adjacency.shape[0], 20)

    def test_common_neighbors_2(self):
        g, adjacency = self.__load_medium_dataset()

        cn = CommonNeighbors()
        cn.fit(adjacency)

        our_sim = local_similarity.common_neighbors(g)

        self.__perform_sktest(our_sim, cn, adjacency.shape[0])

    def test_common_neighbors_3(self):
        g, adjacency = self.__load_hard_dataset()
        cn = CommonNeighbors()

        cn.fit(adjacency)
        our_sim = local_similarity.common_neighbors(g)

        self.__perform_sktest(our_sim, cn, adjacency.shape[0])

    def test_jaccard_1(self):
        g, adjacency = self.__load_easy_dataset()

        cn = JaccardIndex()
        cn.fit(adjacency)

        our_sim = local_similarity.jaccard(g)

        self.__perform_sktest(our_sim, cn, adjacency.shape[0])

    def test_jaccard_2(self):
        g, adjacency = self.__load_medium_dataset()

        cn = JaccardIndex()
        cn.fit(adjacency)

        our_sim = local_similarity.jaccard(g)

        self.__perform_sktest(our_sim, cn, adjacency.shape[0])

    def test_jaccard_3(self):
        g, adjacency = self.__load_hard_dataset()

        cn = JaccardIndex()
        cn.fit(adjacency)

        our_sim = local_similarity.jaccard(g)

        self.__perform_sktest(our_sim, cn, adjacency.shape[0])

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
