import networkx as nx
import numpy as np
from social_network_link_prediction.utils import nodes_to_indexes
from scipy.sparse import lil_matrix, csr_matrix


def __t(G: nx.Graph, z) -> int:
    """Number of triangles passing through the node z"""
    return nx.triangles(G, z)


def __C(G: nx.Graph, z) -> float:
    """Clustering Coefficient"""
    z_degree = G.degree[z]

    # avoiding 0 divition error
    if z_degree == 1:
        return 0

    return __t(G, z) / (z_degree * (z_degree - 1))


def __node_clustering(G: nx.Graph, x, y) -> float:
    """Compute the Node Clustering Coefficient for 2 given nodes."""
    return sum([__C(G, z) for z in (set(G[x]) & set(G[y]))])


def node_clustering(G: nx.Graph) -> csr_matrix:
    """Compute the Hub Depressed Index for all nodes in the Graph.
    Each similarity value is defined as:

    .. math::
        S(x, y) = \\sum_{z \\in \\Gamma(x) \\cap \\Gamma(y)} C(z)

    where

    .. math::
        C(z) = \\frac{t(z)}{k_z(k_z - 1)}

    is the clustering coefficient of node \\(z\\), \\(t(z)\\)
    is the total triangles passing through the node \\(z\\),
    \\(\\Gamma(x)\\) are the neighbors of node \\(x\\)
    and \\(k_x\\) is the degree of the node \\(x\\).

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)

    Returns
    -------
    S: csr_matrix : the Similarity Matrix (in sparse format)

    Notes
    -----
    This index is also based on the clustering coefficient property
    of the network in which the clustering coefficients of all
    the common neighbors of a seed node pair are computed
    and summed to find the final similarity score of the pair.
    """
    size = G.number_of_nodes()
    S = lil_matrix((size, size))
    name_index_map = list(nodes_to_indexes(G).items())

    for x, y in zip(*np.triu_indices(size)):
        x_node = name_index_map[x][0]
        y_node = name_index_map[y][0]

        S[x, y] = __node_clustering(G, x_node, y_node)
        S[y, x] = S[x, y]

    return S.tocsr()
