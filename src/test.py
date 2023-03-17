import networkx as nx

import numpy as np
from social_network_link_prediction.similarity_methods.quasi_local_methods import LPI
from social_network_link_prediction.similarity_methods.quasi_local_methods import path_of_length_three
import matplotlib.pyplot as plt

def test_LPI():
    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2, 3])
    graph.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])

    nx.draw(graph, with_labels=True)

    print(LPI.local_path_index(graph, 0.1, 4))

    plt.show()


def test_path_of_length():
    G = nx.gnp_random_graph(1000, .01)

    print(path_of_length_three.path_of_length_three(G))

    nx.draw(G, with_labels=True)
    plt.show()

# --- COMMON NEIGHBORS

from social_network_link_prediction.similarity_methods.local_similarity import common_neighbors


def test_common_neighbors_easy():
    
    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2, 3])
    graph.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])
    
    nx.draw(graph, with_labels=True)

    print(common_neighbors(graph))

    plt.show()


def test_common_neighbors_hard():
    G = nx.gnp_random_graph(1000, .01)

    print(common_neighbors(G))

    nx.draw(G, with_labels=True)
    plt.show()


# test_path_of_length()
# test_common_neighbors_easy()
test_common_neighbors_hard()
