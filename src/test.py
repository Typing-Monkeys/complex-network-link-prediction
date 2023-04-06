import networkx as nx
import matplotlib.pyplot as plt
from social_network_link_prediction.similarity_methods.local_similarity import common_neighbors, adamic_adar, preferential_attachment, resource_allocation, cosine_similarity, sorensen, hub_promoted, hub_depressed, node_clustering, jaccard
from social_network_link_prediction.similarity_methods.quasi_local_similarity import local_path_index, path_of_length_three
from social_network_link_prediction.similarity_methods.global_similarity import shortest_path, katz_index, rooted_page_rank
from social_network_link_prediction.dimensionality_reduction_methods import link_prediction_svd, link_prediction_nmf


# G =  nx.karate_club_graph()
G = nx.Graph() 
G.add_nodes_from([0, 1, 2, 3])
G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])


def test_LPI():
    nx.draw(G, with_labels=True)

    print(local_path_index(G, 0.1, 4))
    
    print('\n')

    plt.show()


def test_path_of_length():
    print(path_of_length_three(G))
    
    print('\n')

    nx.draw(G, with_labels=True)
    plt.show()


# --- COMMON NEIGHBORS
def test_common_neighbors_easy():

    nx.draw(G, with_labels=True)
    print(common_neighbors(G))
    print('\n')
    plt.show()

def test_common_neighbors_hard():
    print(common_neighbors(G))
    print('\n')

    nx.draw(G, with_labels=True)
    plt.show()


# --- Jaccard Measure
def test_jaccard():
    print(jaccard(G))
    print('\n')

    nx.draw(G, with_labels=True)
    plt.show()


# --- Adamic Adar Measure
def test_adamic_adar():
    print(adamic_adar(G))
    print('\n')

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
    print(cosine_similarity(G))
    print('\n')

    nx.draw(G, with_labels=True)
    plt.show()


# --- Sorensen
def test_sorensen():
    print(sorensen(G))
    print('\n')

    nx.draw(G, with_labels=True)
    plt.show()


# --- Hub Promoted
def test_hubpromoted():
    print(hub_promoted(G))
    print('\n')

    nx.draw(G, with_labels=True)
    plt.show()

# --- Hub Depressed
def test_hubdepressed():
    print(hub_depressed(G))
    print('\n')
    
    nx.draw(G, with_labels=True)
    plt.show()

# --- Node Clustering
def test_nodeclusterintg():
    G = nx.gnp_random_graph(1000, .01)

    print(node_clustering(G))

    nx.draw(G, with_labels=True)
    plt.show()

# --- Katz Index
def test_katz_index():
    print(katz_index(G, 10))
    print('\n')

    nx.draw(G, with_labels=True)
    plt.show()

# --- Shortest Path
def test_shortest_path():
    print(shortest_path(G, 10))
    print('\n')
    
    nx.draw(G, with_labels=True)
    plt.show()
    
# --- Rooted Page Rank
def test_rooted_page_rank():
    print(rooted_page_rank(G))
    print('\n')
    
    nx.draw(G, with_labels=True)
    plt.show()

# --- SVD
def test_svd():
    G = nx.gnp_random_graph(1000, .01)

    print(link_prediction_svd(G, normalize=True))

    nx.draw(G, with_labels=True)
    plt.show()

# --- NMF
def test_link_prediction_nmf():
    G = nx.gnp_random_graph(1000, .01)
    
    print(link_prediction_nmf(G))


#-----------------------------
'''
    Local Similarity
'''

#---OK
'''
test_common_neighbors_easy()
test_common_neighbors_hard()
test_jaccard()
'''
#---Result

#---OK
'''
test_common_neighbors_easy()
test_common_neighbors_hard()
test_sorensen()
test_jaccard()
'''
#---Result

#---OK
'''
test_common_neighbors_easy()
test_common_neighbors_hard()
test_sorensen()
test_jaccard()
test_hubdepressed()
test_hubpromoted()
'''
#---Result

#---OK
'''
test_common_neighbors_easy()
test_common_neighbors_hard()
test_sorensen()
test_jaccard()
test_hubdepressed()
test_hubpromoted()
test_adamic_adar()
test_cosinesimilarity()
'''
#---Result



#-----------------------------
'''
    Global Similarity
'''
#---OK
'''
test_katz_index()
test_shortest_path()
'''
#---Result

#---Ricontrollare 
'''
test_katz_index()
test_shortest_path()
test_rooted_page_rank()
'''
#-----------------------------



#-----------------------------
'''
    Quasi local index
'''
#---OK
'''
test_LPI()
test_path_of_length()
'''
#---Result


