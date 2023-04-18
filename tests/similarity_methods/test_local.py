import unittest
import random
from time import time
from social_network_link_prediction.similarity_methods import local_similarity
from sknetwork.linkpred import CommonNeighbors
from sknetwork.linkpred import JaccardIndex
from sknetwork.linkpred import SorensenIndex
from sknetwork.linkpred import HubPromotedIndex
from sknetwork.linkpred import HubDepressedIndex
from sknetwork.linkpred import AdamicAdar
from sknetwork.linkpred import ResourceAllocation
from sknetwork.linkpred import PreferentialAttachment
from tests import Configs


class TestLocalSimilarityMethods(unittest.TestCase):

    def __perform_test(self, g, func):
        sim = func(g)
        self.assertIsNotNone(sim)

    def test_common_neighbors_nolabels(self):
        g = Configs.load_normal_dataset()
        self.__perform_test(g, local_similarity.common_neighbors)

    def test_common_neighbors_labels(self):
        g = Configs.load_labels_dataset()
        self.__perform_test(g, local_similarity.common_neighbors)
   
    def test_adamicadar_nolabels(self):
        g = Configs.load_normal_dataset()
        self.__perform_test(g, local_similarity.adamic_adar)

    def test_adamicadar_labels(self):
        g = Configs.load_labels_dataset()
        self.__perform_test(g, local_similarity.adamic_adar)

    def test_jaccard_nolabels(self):
        g = Configs.load_normal_dataset()
        self.__perform_test(g, local_similarity.jaccard)

    def test_jaccard_labels(self):
        g = Configs.load_labels_dataset()
        self.__perform_test(g, local_similarity.jaccard)

    def test_sorensen_nolabels(self):
        g = Configs.load_normal_dataset()
        self.__perform_test(g, local_similarity.sorensen)

    def test_sorensen_labels(self):
        g = Configs.load_labels_dataset()
        self.__perform_test(g, local_similarity.sorensen)

    def test_hubpromoted_nolabels(self):
        g = Configs.load_normal_dataset()
        self.__perform_test(g, local_similarity.hub_promoted)

    def test_hubpromoted_labels(self):
        g = Configs.load_labels_dataset()
        self.__perform_test(g, local_similarity.hub_promoted)

    def test_hubdepressed_nolabels(self):
        g = Configs.load_normal_dataset()
        self.__perform_test(g, local_similarity.hub_depressed)

    def test_hubdepressed_labels(self):
        g = Configs.load_labels_dataset()
        self.__perform_test(g, local_similarity.hub_depressed)

    def test_resourceallocation_nolabels(self):
        g = Configs.load_normal_dataset()
        self.__perform_test(g, local_similarity.resource_allocation)

    def test_resourceallocation_labels(self):
        g = Configs.load_labels_dataset()
        self.__perform_test(g, local_similarity.resource_allocation)

    def test_prefattachment_nolabels(self):
        g = Configs.load_normal_dataset()
        self.__perform_test(g, local_similarity.preferential_attachment)

    def test_prefattachment_labels(self):
        g = Configs.load_labels_dataset()
        self.__perform_test(g, local_similarity.preferential_attachment)

    def test_cosine_nolabels(self):
        g = Configs.load_normal_dataset()
        self.__perform_test(g, local_similarity.cosine_similarity)

    def test_cosine_labels(self):
        g = Configs.load_labels_dataset()
        self.__perform_test(g, local_similarity.cosine_similarity)

    def test_nodeclustering_nolabels(self):
        g = Configs.load_normal_dataset()
        self.__perform_test(g, local_similarity.node_clustering)

    def test_nodeclustering_labels(self):
        g = Configs.load_labels_dataset()
        self.__perform_test(g, local_similarity.node_clustering)

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
