import networkx as nx
import numpy as np
from social_network_link_prediction.utils import nodes_to_indexes
from scipy.sparse import lil_matrix, csr_matrix


def __resource_allocation(G: nx.Graph, x, y) -> float:
    return sum([1 / G.degree[z] for z in set(G[x]) & set(G[y])])


def resource_allocation(G: nx.Graph) -> csr_matrix:
    """TODO

    Parameters
    ----------
    G: nx.Graph :
        grafo da analizzare

    Returns
    -------
    S: csr_matrix : matrice di Similarità
    """
    size = G.number_of_nodes()
    S = lil_matrix((size, size))
    name_index_map = list(nodes_to_indexes(G).items())

    for x, y in zip(*np.triu_indices(size)):
        x_node = name_index_map[x][0]
        y_node = name_index_map[y][0]

        S[x, y] = __resource_allocation(G, x_node, y_node)
        S[y, x] = S[x, y]

    return S.tocsr()
