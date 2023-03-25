import unittest
from social_network_link_prediction.similarity_methods import global_similarity
import networkx as nx
from sknetwork.data import house
import numpy as np
import networkit as nk


class TestGlobalSimilarityMethods(unittest.TestCase):

    # TODO: @Cosci, potrebbe esserci qualche problema
    def test_katz(self):
        from sknetwork.ranking import Katz
        from networkit.linkprediction import KatzIndex

        beta = .1
        adjacency = house()
        G =nx.from_numpy_array(adjacency)

        G_nk = nk.Graph(G.number_of_nodes())
        
        for edge in G.edges():
            G_nk.addEdge(*edge)
        
        print(nk.overview(G_nk))

        cn = Katz(beta)
        kaz = KatzIndex(G_nk)

        scores_skn = cn.fit_predict(adjacency)
        scores_nx  = nx.katz_centrality(G, beta=beta)
        scores_nk = kaz.runAll() # ritorna la similarità solo tra nodi non connessi
        our_scores = global_similarity.katz_index(G, beta=beta).toarray()
        
        print()
        print(np.round(scores_skn,2))
        print(np.round(list(scores_nx.values()),2))
        print(scores_nk)
        print(np.round(our_scores.diagonal(),2))
        
        self.assertEqual(scores_nx.values(), our_scores.diagonal())
        self.assertEqual(scores_skn, our_scores.diagonal())


if __name__ == '__main__':
    unittest.main()
