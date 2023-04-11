import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from social_network_link_prediction.similarity_methods.local_similarity import common_neighbors, adamic_adar, preferential_attachment, resource_allocation, cosine_similarity, sorensen, hub_promoted, hub_depressed, node_clustering, jaccard
from social_network_link_prediction.similarity_methods.quasi_local_similarity import local_path_index, path_of_length_three
from social_network_link_prediction.similarity_methods.global_similarity import shortest_path, katz_index, rooted_page_rank
from social_network_link_prediction.dimensionality_reduction_methods import link_prediction_svd, link_prediction_nmf


G =  nx.karate_club_graph()
# G = nx.Graph() 
# G.add_nodes_from([0, 1, 2, 3, 4, 5])
# G.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (2, 5)])


def top_five_element(result):
    # Convert CSR matrix to 1D array
    flatten_matrix = result.data
    
    # Sort the values in descending order
    sorted_values = np.sort(flatten_matrix)[::-1]

    # Get the top 5 values
    top_5_values = sorted_values[:5]

    print("Top 5 elements with the highest values:\n", top_5_values)
    print('\n')
    

def test_LPI():
    nx.draw(G, with_labels=True)

    result = local_path_index(G, 0.1, 4)
    
    top_five_element(result)

    plt.show()


def test_path_of_length():
    result = path_of_length_three(G)
    top_five_element(result)

    nx.draw(G, with_labels=True)
    plt.show()


# --- COMMON NEIGHBORS
def test_common_neighbors_easy():

    nx.draw(G, with_labels=True)
    
    result = common_neighbors(G)
    
    top_five_element(result)
    
    plt.show()


def test_common_neighbors_hard():
    nx.draw(G, with_labels=True)
    
    result = common_neighbors(G)
    
    top_five_element(result)
    
    plt.show()


# --- Jaccard Measure
def test_jaccard():
    nx.draw(G, with_labels=True)
    result = jaccard(G)
    
    top_five_element(result)

    plt.show()


# --- Adamic Adar Measure
def test_adamic_adar():
    result = adamic_adar(G)
    top_five_element(result)
    
    nx.draw(G, with_labels=True)
    plt.show()


# --- Preferentaial Attachment
def test_preferential():
    G = nx.gnp_random_graph(1000, .01)

    result = preferential_attachment(G)

    top_five_element(result)
    
    nx.draw(G, with_labels=True)
    plt.show()


# --- Resource Allocation
def test_resourceallocation():
    result = resource_allocation(G)
    
    top_five_element(result)

    nx.draw(G, with_labels=True)
    plt.show()


# --- Cosine Similarity
def test_cosinesimilarity():
    result = cosine_similarity(G)
    
    top_five_element(result)

    nx.draw(G, with_labels=True)
    plt.show()


# --- Sorensen
def test_sorensen():
    result = sorensen(G)
    
    top_five_element(result)
    
    nx.draw(G, with_labels=True)
    plt.show()


# --- Hub Promoted
def test_hubpromoted():
    result = hub_promoted(G)
    
    top_five_element(result)

    nx.draw(G, with_labels=True)
    plt.show()

# --- Hub Depressed
def test_hubdepressed():
    result = hub_depressed(G)
    
    top_five_element(result)
    
    nx.draw(G, with_labels=True)
    plt.show()

# --- Node Clustering
def test_nodeclusterintg():
    result = node_clustering(G)
    top_five_element(result)

    nx.draw(G, with_labels=True)
    plt.show()

# --- Katz Index
def test_katz_index():
    result = katz_index(G, 10)
    
    top_five_element(result)

    nx.draw(G, with_labels=True)
    plt.show()

# --- Shortest Path
def test_shortest_path():
    result = shortest_path(G, 10)
    top_five_element(result)
    
    nx.draw(G, with_labels=True)
    plt.show()
    
# --- Rooted Page Rank
def test_rooted_page_rank():
    result = rooted_page_rank(G)
    top_five_element(result)
    
    nx.draw(G, with_labels=True)
    plt.show()

# --- SVD
def test_svd():
    result = link_prediction_svd(G, normalize=True)
    top_five_element(result)
    nx.draw(G, with_labels=True)
    plt.show()

# --- NMF
def test_link_prediction_nmf():
    result = link_prediction_nmf(G)
    top_five_element(result)


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
#---RIvedere Adamic e Nodeclustering
'''
test_common_neighbors_easy()
test_common_neighbors_hard()
test_sorensen()
test_jaccard()
test_hubdepressed()
test_hubpromoted()
test_adamic_adar()
test_cosinesimilarity()
test_nodeclusterintg()
'''
#---Result
#-----------------------------

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

#---OK 
'''
test_katz_index()
test_shortest_path()
test_rooted_page_rank()
'''
#---Result
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
#-----------------------------



#-----------------------------
'''
    Probabilistic index
'''
#---
'''
test_svd()
test_link_prediction_nmf()
'''
#---Result
#-----------------------------
