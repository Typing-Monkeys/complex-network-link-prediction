import networkx as nx
import numpy as np
from social_network_link_prediction.utils import to_adjacency_matrix, nodes_to_indexes
from scipy.sparse import lil_matrix, csr_matrix


def __t(G: nx.Graph, z) -> int:
    """
        Numero di Triangoli passanti per il Nodo Z
    """

    return nx.triangles(G, z)


def __C(G: nx.Graph, z):
    z_degree = G.degree[z]
    return __t(G, z) / (z_degree * (z_degree - 1))


def __node_clustering(G: nx.Graph, x, y):
    return sum([__C(G, z) for z in (set(G[x]) & set(G[y]))])


def node_clustering(G: nx.Graph) -> csr_matrix:
    size = G.number_of_nodes()
    S = lil_matrix((size, size))
    name_index_map = list(nodes_to_indexes(G).items())

    for x, y in zip(*np.triu_indices(size)):
        x_node = name_index_map[x][0]
        y_node = name_index_map[y][0]

        S[x, y] = __node_clustering(G, x_node, y_node)
        S[y, x] = S[x, y]

    return S.tocsr()
