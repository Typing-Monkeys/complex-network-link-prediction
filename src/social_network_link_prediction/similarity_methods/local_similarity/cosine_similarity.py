import networkx as nx
import numpy as np
from .common_neighbors import __common_neighbors
from scipy.sparse import lil_matrix, csr_matrix
from social_network_link_prediction.utils import nodes_to_indexes


def __cosine_similarity(G: nx.Graph, x, y) -> float:
    """Compute the Cosine Similarity Index for 2 given nodes."""
    return __common_neighbors(G, x, y) / np.sqrt(G.degree[x] * G.degree[y])


def cosine_similarity(G: nx.Graph) -> csr_matrix:
    """Compute the Cosine Similarity Index (a.k.a. Salton Index) for all nodes in the Graph.
    Each similarity value is defined as:

    .. math::
        S(x, y) = \\frac{|\Gamma(x) \cap \Gamma(y)|}{\sqrt{k_x k_y}}

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
    This similarity index between two nodes is measured by calculating the Cosine of the angle between them. 
    The metric is all about the orientation and not magnitude. 
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
