import unittest
from social_network_link_prediction.similarity_methods import global_similarity
import networkx as nx
from sknetwork.data import house
import networkit as nk


class TestGlobalSimilarityMethods(unittest.TestCase):

    # TODO: @Cosci, anche questa implementazione non va
    @unittest.skip(
        "Il metodo di Networkit differisce nell'implementazione dal nostro")
    def test_katz(self):
        from networkit.linkprediction import KatzIndex

        adjacency = house()

        beta = .3

        G = nx.from_numpy_array(adjacency)
        G_nk = nk.Graph(G.number_of_nodes())

        for edge in G.edges():
            G_nk.addEdge(*edge)

        print(nk.overview(G_nk))

        kaz = KatzIndex(G_nk, 10, beta)

        # ritorna la similarità solo tra nodi non connessi
        scores_nk = kaz.runAll()
        our_scores = global_similarity.katz_index(G, beta=beta).toarray()

        print()
        print(scores_nk)
        print()
        print(our_scores)

        self.assertEqual(scores_nk, our_scores)


if __name__ == '__main__':
    unittest.main()
