import networkx as nx
import numpy as np
from .common_neighbors import __common_neighbors
from scipy.sparse import lil_matrix, csr_matrix
from social_network_link_prediction.utils import nodes_to_indexes


def __cosine_similarity(G: nx.Graph, x, y) -> float:
    return __common_neighbors(G, x, y) / np.sqrt(G.degree[x] * G.degree[y])


def cosine_similarity(G: nx.Graph) -> csr_matrix:
    """

    Parameters
    ----------
    G: nx.Graph :
        

    Returns
    -------

    """
    size = G.number_of_nodes()
    S = lil_matrix((size, size))
    name_index_map = list(nodes_to_indexes(G).items())

    for x, y in zip(*np.triu_indices(size)):
        x_node = name_index_map[x][0]
        y_node = name_index_map[y][0]

        S[x, y] = __cosine_similarity(G, x_node, y_node)
        S[y, x] = S[x, y]

    return S.tocsr()
