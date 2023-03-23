import networkx as nx
import matplotlib.pyplot as plt
from social_network_link_prediction.similarity_methods.local_similarity import common_neighbors, adamic_adar, preferential_attachment, resource_allocation, cosine_similarity, sorensen,hub_promoted, hub_depressed, node_clustering
from social_network_link_prediction.similarity_methods.quasi_local_similarity import local_path_index
from social_network_link_prediction.similarity_methods.quasi_local_similarity import path_of_length_three
from social_network_link_prediction.similarity_methods.local_similarity import jaccard


def test_LPI():
    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2, 3])
    graph.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])

    nx.draw(graph, with_labels=True)

    print(local_path_index(graph, 0.1, 4))

    plt.show()


def test_path_of_length():
    G = nx.gnp_random_graph(1000, .01)

    print(path_of_length_three(G))

    nx.draw(G, with_labels=True)
    plt.show()


# --- COMMON NEIGHBORS
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


# --- Jaccard Measure
def test_jaccard():
    G = nx.gnp_random_graph(1000, .01)

    print(jaccard(G))

    nx.draw(G, with_labels=True)
    plt.show()

# --- Adamic Adar Measure
def test_adamic_adar():
    G = nx.gnp_random_graph(1000, .01)

    print(adamic_adar(G))

    nx.draw(G, with_labels=True)
    plt.show()

# --- Preferentaial Attachment
def test_preferential():
    G = nx.gnp_random_graph(1000, .01)

    print(preferential_attachment(G))

    nx.draw(G, with_labels=True)
    plt.show()

# --- Resource Allocation    
def test_resourceallocation():
    G = nx.gnp_random_graph(1000, .01)

    print(resource_allocation(G))

    nx.draw(G, with_labels=True)
    plt.show()

# --- Cosine Similarity
def test_cosinesimilarity():
    G = nx.gnp_random_graph(1000, .01)

    print(cosine_similarity(G))

    nx.draw(G, with_labels=True)
    plt.show()

# --- Sorensen
def test_sorensen():
    G = nx.gnp_random_graph(1000, .01)

    print(sorensen(G))

    nx.draw(G, with_labels=True)
    plt.show()

# --- Hub Promoted
def test_hubpromoted():
    G = nx.gnp_random_graph(1000, .01)

    print(hub_promoted(G))

    nx.draw(G, with_labels=True)
    plt.show()

# --- Hub Depressed
def test_hubdepressed():
    G = nx.gnp_random_graph(1000, .01)

    print(hub_depressed(G))

    nx.draw(G, with_labels=True)
    plt.show()

# --- Node Clustering
def test_nodeclusterintg():
    G = nx.gnp_random_graph(1000, .01)

    print(node_clustering(G))

    nx.draw(G, with_labels=True)
    plt.show()

# test_path_of_length()
# test_common_neighbors_easy()
# test_common_neighbors_hard()
# test_jaccard()
# test_adamic_adar()
# test_preferential()
# test_resourceallocation()
# test_cosinesimilarity()
# test_sorensen()
# test_hubpromoted()
# test_hubdepressed()
test_nodeclusterintg()