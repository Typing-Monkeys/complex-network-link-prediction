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
    G = nx.gnp_random_graph(1000, .02)

    print(path_of_length_three.path_of_length_three(G))

    nx.draw(G, with_labels=True)
    plt.show()
    exit(0)
    # create an empty G
    G = nx.Graph()

    # add nodes
    G.add_nodes_from([1, 2, 3, 4])

    # add weighted edges
    G.add_weighted_edges_from([(1, 2, 0.5), (2, 3, 1.0), (3, 4, 2.0),
                               (4, 1, 1.5)])

    # access edge weights
    '''
        Show weights of G

        print(G[1][2]['weight'])  # print 0.5
        print(G[3][4]['weight'])  # print 2.0
    '''

    result = path_of_length_three.path_of_length_three(G)

    print(result)
