import networkx as nx
import numpy as np
from .common_neighbors import __common_neighbors
from scipy.sparse import lil_matrix, csr_matrix
from social_network_link_prediction.utils import nodes_to_indexes


def __hub_promoted(G: nx.Graph, x, y) -> float:
    """Compute the Hub Promoted Index for 2 given nodes."""
    return __common_neighbors(G, x, y) / min(G.degree[x], G.degree[y])


def hub_promoted(G: nx.Graph) -> csr_matrix:
    """Compute the Hub Promoted Index for all nodes in the Graph.
    Each similarity value is defined as:

    .. math::
        S(x, y) = \\frac{2 |\Gamma(x) \cap \Gamma(y)|}{\min(k_x, k_y)}

    where \\(\Gamma(x)\\) are the neighbors of node \\(x\\) and \\(k_x\) is the degree of the node \\(x\\).

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)

    Returns
    -------
    S: csr_matrix : the Similarity Matrix (in sparse format)

    Notes
    -----
    This similarity index promotes the formation of links between the sparsely connected nodes and hubs.
    It also tries to prevent links formation between the hub nodes.
    """
    size = G.number_of_nodes()
    S = lil_matrix((size, size))
    name_index_map = list(nodes_to_indexes(G).items())

    for x, y in zip(*np.triu_indices(size)):
        x_node = name_index_map[x][0]
        y_node = name_index_map[y][0]

        S[x, y] = __hub_promoted(G, x_node, y_node)
        S[y, x] = S[x, y]

    return S.tocsr()
