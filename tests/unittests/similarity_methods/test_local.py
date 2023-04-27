import unittest
from cnlp.similarity_methods import local_similarity
from tests import Configs
from scipy import sparse
import numpy as np


class TestLocalSimilarityMethods(unittest.TestCase):

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

    def test_common_neighbors(self):
        self.__perform_test(local_similarity.common_neighbors)

    def test_adamicadar(self):
        self.__perform_test(local_similarity.adamic_adar)

    def test_jaccard(self):
        self.__perform_test(local_similarity.jaccard)

    def test_sorensen(self):
        self.__perform_test(local_similarity.sorensen)

    def test_hubpromoted(self):
        self.__perform_test(local_similarity.hub_promoted)

    def test_hubdepressed(self):
        self.__perform_test(local_similarity.hub_depressed)

    def test_resourceallocation(self):
        self.__perform_test(local_similarity.resource_allocation)

    def test_prefattachment(self):
        self.__perform_test(local_similarity.preferential_attachment)

    def test_cosine(self):
        self.__perform_test(local_similarity.cosine_similarity)

    def test_nodeclustering(self):
        self.__perform_test(local_similarity.node_clustering)

    # def setUp(self):
    #     self.start_time = time()

    # def tearDown(self):
    #     t = time() - self.start_time
    #     print(f"{round(t, 2)} s")

    # def __perform_sktest(self,
    #                      our_values,
    #                      skfunction,
    #                      samples_range,
    #                      samples_number=20,
    #                      decimal_precision=3):
    #     indxes = [(random.randrange(samples_range),
    #                random.randrange(samples_range))
    #               for a in range(samples_number)]

    #     for i, j in indxes:
    #         expected = skfunction.predict((i, j))
    #         our_res = our_values[i, j]

    #         self.assertAlmostEqual(expected, our_res, decimal_precision)

    # def test_common_neighbors_1(self):
    #     g, adjacency = Configs.load_easy_dataset()
    #     cn = CommonNeighbors()

    #     cn.fit(adjacency)
    #     our_sim = local_similarity.common_neighbors(g)

    #     self.__perform_sktest(our_sim, cn, adjacency.shape[0], 20)

    # def test_common_neighbors_2(self):
    #     g, adjacency = Configs.load_medium_dataset()

    #     cn = CommonNeighbors()
    #     cn.fit(adjacency)

    #     our_sim = local_similarity.common_neighbors(g)

    #     self.__perform_sktest(our_sim, cn, adjacency.shape[0])

    # @timeout(Configs.timeout)
    # def test_common_neighbors_3(self):
    #     g, adjacency = Configs.load_hard_dataset()
    #     cn = CommonNeighbors()

    #     cn.fit(adjacency)
    #     our_sim = local_similarity.common_neighbors(g)

    #     self.__perform_sktest(our_sim, cn, adjacency.shape[0])

    # def test_jaccard_1(self):
    #     g, adjacency = Configs.load_easy_dataset()

    #     cn = JaccardIndex()
    #     cn.fit(adjacency)

    #     our_sim = local_similarity.jaccard(g)

    #     self.__perform_sktest(our_sim, cn, adjacency.shape[0])

    # def test_jaccard_2(self):
    #     g, adjacency = Configs.load_medium_dataset()

    #     cn = JaccardIndex()
    #     cn.fit(adjacency)

    #     our_sim = local_similarity.jaccard(g)

    #     self.__perform_sktest(our_sim, cn, adjacency.shape[0])

    # @timeout(Configs.timeout)
    # def test_jaccard_3(self):
    #     g, adjacency = Configs.load_hard_dataset()

    #     cn = JaccardIndex()
    #     cn.fit(adjacency)

    #     our_sim = local_similarity.jaccard(g)

    #     self.__perform_sktest(our_sim, cn, adjacency.shape[0])

    # def test_sorensen_1(self):
    #     g, adjacency = Configs.load_easy_dataset()

    #     cn = SorensenIndex()
    #     cn.fit(adjacency)

    #     our_sim = local_similarity.sorensen(g)

    #     self.__perform_sktest(our_sim, cn, adjacency.shape[0])

    # def test_sorensen_2(self):
    #     g, adjacency = Configs.load_medium_dataset()

    #     cn = SorensenIndex()
    #     cn.fit(adjacency)

    #     our_sim = local_similarity.sorensen(g)

    #     self.__perform_sktest(our_sim, cn, adjacency.shape[0])

    # @timeout(Configs.timeout)
    # def test_sorensen_3(self):
    #     g, adjacency = Configs.load_hard_dataset()

    #     cn = SorensenIndex()
    #     cn.fit(adjacency)

    #     our_sim = local_similarity.sorensen(g)

    #     self.__perform_sktest(our_sim, cn, adjacency.shape[0])

    # def test_hubpromoted_1(self):
    #     g, adjacency = Configs.load_easy_dataset()

    #     cn = HubPromotedIndex()
    #     cn.fit(adjacency)

    #     our_sim = local_similarity.hub_promoted(g)

    #     self.__perform_sktest(our_sim, cn, adjacency.shape[0])

    # def test_hubpromoted_2(self):
    #     g, adjacency = Configs.load_medium_dataset()

    #     cn = HubPromotedIndex()
    #     cn.fit(adjacency)

    #     our_sim = local_similarity.hub_promoted(g)

    #     self.__perform_sktest(our_sim, cn, adjacency.shape[0])

    # @timeout(Configs.timeout)
    # def test_hubpromoted_3(self):
    #     g, adjacency = Configs.load_hard_dataset()

    #     cn = HubPromotedIndex()
    #     cn.fit(adjacency)

    #     our_sim = local_similarity.hub_promoted(g)

    #     self.__perform_sktest(our_sim, cn, adjacency.shape[0])

    # def test_hubdepressed_1(self):
    #     g, adjacency = Configs.load_easy_dataset()

    #     cn = HubDepressedIndex()
    #     cn.fit(adjacency)

    #     our_sim = local_similarity.hub_depressed(g)

    #     self.__perform_sktest(our_sim, cn, adjacency.shape[0])

    # def test_hubdepressed_2(self):
    #     g, adjacency = Configs.load_medium_dataset()

    #     cn = HubDepressedIndex()
    #     cn.fit(adjacency)

    #     our_sim = local_similarity.hub_depressed(g)

    #     self.__perform_sktest(our_sim, cn, adjacency.shape[0])

    # @timeout(Configs.timeout)
    # def test_hubdepressede_3(self):
    #     g, adjacency = Configs.load_hard_dataset()

    #     cn = HubDepressedIndex()
    #     cn.fit(adjacency)

    #     our_sim = local_similarity.hub_depressed(g)

    #     self.__perform_sktest(our_sim, cn, adjacency.shape[0])

    # def test_adamicadar_1(self):
    #     g, adjacency = Configs.load_easy_dataset()

    #     cn = AdamicAdar()
    #     cn.fit(adjacency)

    #     our_sim = local_similarity.adamic_adar(g)

    #     self.__perform_sktest(our_sim, cn, adjacency.shape[0])

    # def test_adamicadar_2(self):
    #     g, adjacency = Configs.load_medium_dataset()

    #     cn = AdamicAdar()
    #     cn.fit(adjacency)

    #     our_sim = local_similarity.adamic_adar(g)

    #     self.__perform_sktest(our_sim, cn, adjacency.shape[0])

    # @timeout(Configs.timeout)
    # def test_adamicadar_3(self):
    #     g, adjacency = Configs.load_hard_dataset()

    #     cn = AdamicAdar()
    #     cn.fit(adjacency)

    #     our_sim = local_similarity.adamic_adar(g)

    #     self.__perform_sktest(our_sim, cn, adjacency.shape[0])

    # def test_resourceallocation_1(self):
    #     g, adjacency = Configs.load_easy_dataset()

    #     cn = ResourceAllocation()
    #     cn.fit(adjacency)

    #     our_sim = local_similarity.resource_allocation(g)

    #     self.__perform_sktest(our_sim, cn, adjacency.shape[0])

    # def test_resourceallocation_2(self):
    #     g, adjacency = Configs.load_medium_dataset()

    #     cn = ResourceAllocation()
    #     cn.fit(adjacency)

    #     our_sim = local_similarity.resource_allocation(g)

    #     self.__perform_sktest(our_sim, cn, adjacency.shape[0])

    # @timeout(Configs.timeout)
    # def test_resourceallocation_3(self):
    #     g, adjacency = Configs.load_hard_dataset()

    #     cn = ResourceAllocation()
    #     cn.fit(adjacency)

    #     our_sim = local_similarity.resource_allocation(g)

    #     self.__perform_sktest(our_sim, cn, adjacency.shape[0])

    # def test_preferentialattachment_1(self):
    #     g, adjacency = Configs.load_easy_dataset()

    #     cn = PreferentialAttachment()
    #     cn.fit(adjacency)

    #     our_sim = local_similarity.preferential_attachment(g)

    #     self.__perform_sktest(our_sim, cn, adjacency.shape[0])

    # def test_preferentialattachment_2(self):
    #     g, adjacency = Configs.load_medium_dataset()

    #     cn = PreferentialAttachment()
    #     cn.fit(adjacency)

    #     our_sim = local_similarity.preferential_attachment(g)

    #     self.__perform_sktest(our_sim, cn, adjacency.shape[0])

    # @timeout(Configs.timeout)
    # def test_preferentialattachment_3(self):
    #     g, adjacency = Configs.load_hard_dataset()

    #     cn = PreferentialAttachment()
    #     cn.fit(adjacency)

    #     our_sim = local_similarity.preferential_attachment(g)

    #     self.__perform_sktest(our_sim, cn, adjacency.shape[0])

    # @unittest.skip("Non è presente in scikit-network")
    # def test_cosine_1(self):
    #     pass

    # @unittest.skip("Non è presente in scikit-network")
    # def test_cosine_2(self):
    #     pass

    # @unittest.skip("Non è presente in scikit-network")
    # def test_cosine_3(self):
    #     pass

    # @timeout(Configs.timeout)
    # def test_cosine_time(self):
    #     g, adjacency = Configs.load_hard_dataset()

    #     our_sim = local_similarity.cosine_similarity(g)

    #     self.assertIsNotNone(our_sim)

    # @unittest.skip("Non è presente in scikit-network")
    # def test_nodeclustering_1(self):
    #     pass

    # @unittest.skip("Non è presente in scikit-network")
    # def test_nodeclustering_2(self):
    #     pass

    # @unittest.skip("Non è presente in scikit-network")
    # def test_nodeclustering_3(self):
    #     pass

    # @timeout(Configs.timeout)
    # def test_nodeclustering_time(self):
    #     g, adjacency = Configs.load_hard_dataset()

    #     our_sim = local_similarity.node_clustering(g)

    #     self.assertIsNotNone(our_sim)


if __name__ == '__main__':
    unittest.main()
