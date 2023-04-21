import networkx as nx
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from .common_neighbors import __common_neighbors
from social_network_link_prediction.utils import nodes_to_indexes


def __sorensen(G: nx.Graph, x, y) -> float:
    """Compute the Sorensen Index for 2 given nodes."""
    return (2 * __common_neighbors(G, x, y)) / (G.degree[x] + G.degree[y])


def sorensen(G: nx.Graph) -> csr_matrix:
    """Compute the Sorensen Index for all nodes in the Graph.
    Each similarity value is defined as:

    .. math::
        S(x, y) = \\frac{2 |\\Gamma(x) \\cap \\Gamma(y)|}{k_x + k_y}

    where \\(\\Gamma(x)\\) are the neighbors of node \\(x\\)
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
    It is very similar to the Jaccard index. **McCune et al.** show
    that it is more robust than Jaccard against the outliers.
    """
    size = G.number_of_nodes()
    S = lil_matrix((size, size))
    node_index_map = nodes_to_indexes(G)

    for x, y in nx.complement(G).edges():
        _x = node_index_map[x]
        _y = node_index_map[y]

        S[_x, _y] = __sorensen(G, x, y)
        # S[y, x] = S[x, y]

    return S.tocsr()
