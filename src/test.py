import networkx as nx
from social_network_link_prediction.similarity_methods.quasi_local_methods import path_of_length_three
import matplotlib.pyplot as plt


if __name__ == "__main__":
    '''
        If we have a unweighted G the weight have value of 1 
    '''
    """
        LP3 -> G weight
    """
    G = nx.gnp_random_graph(1000, .01)

    print(path_of_length_three.path_of_length_three(G))

    nx.draw(G, with_labels=True)
    plt.show()