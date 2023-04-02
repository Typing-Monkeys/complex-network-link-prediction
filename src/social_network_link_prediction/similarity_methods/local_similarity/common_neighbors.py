import networkx as nx
import numpy as np
from social_network_link_prediction.utils import nodes_to_indexes
from scipy.sparse import lil_matrix, csr_matrix


def __common_neighbors(G: nx.Graph, x, y) -> int:
    """Compute the Common Neighbors Index for 2 given nodes."""
    return len(set(G[x]).intersection(set(G[y])))


def common_neighbors(G: nx.Graph) -> csr_matrix:
    """Compute the Common Neighbors Index for all nodes in the Graph.
    Each similarity value is defined as:

    .. math::
        S(x, y) = |\Gamma(x) \cap \Gamma(y)|

    where \\(\\Gamma(x)\\) are the neighbors of the node \\(x\\).

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)

    Returns
    -------
    S: csr_matrix : the Similarity Matrix (in sparse format)

    Notes
    -----
    The likelihood of the existence of a link between \\(x\\) and \\(y\\) increases with the number 
    of common neighbors between them.
    """

    size = G.number_of_nodes()
    S = lil_matrix((size, size))
    name_index_map = list(nodes_to_indexes(G).items())

    for x, y in zip(*np.triu_indices(size)):
        x_node = name_index_map[x][0]
        y_node = name_index_map[y][0]

        S[x, y] = __common_neighbors(G, x_node, y_node)
        S[y, x] = S[x, y]

    return S.tocsr()
