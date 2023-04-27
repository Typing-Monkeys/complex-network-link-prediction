import networkx as nx
import numpy as np
from .common_neighbors import __common_neighbors
from scipy.sparse import lil_matrix, csr_matrix
from cnlp.utils import nodes_to_indexes


def __cosine_similarity(G: nx.Graph, x, y) -> float:
    """Compute the Cosine Similarity Index for 2 given nodes."""
    return __common_neighbors(G, x, y) / np.sqrt(G.degree[x] * G.degree[y])


def cosine_similarity(G: nx.Graph) -> csr_matrix:
    """Compute the Cosine Similarity Index
    (a.k.a. Salton Index) for all nodes in the Graph.
    Each similarity value is defined as:

    .. math::
        S(x, y) = \\frac{|\\Gamma(x) \\cap \\Gamma(y)|}{\\sqrt{k_x k_y}}

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
    This similarity index between two nodes is measured by
    calculating the Cosine of the angle between them.
    The metric is all about the orientation and not magnitude.
    """
    size = G.number_of_nodes()
    S = lil_matrix((size, size))
    node_index_map = nodes_to_indexes(G)

    for x, y in nx.complement(G).edges():
        _x = node_index_map[x]
        _y = node_index_map[y]

        S[_x, _y] = __cosine_similarity(G, x, y)
        # S[y, x] = S[x, y]

    return S.tocsr()
