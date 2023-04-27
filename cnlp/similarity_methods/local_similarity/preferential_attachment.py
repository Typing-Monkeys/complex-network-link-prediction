import networkx as nx
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from cnlp.utils import nodes_to_indexes


def __preferential_attachment(G: nx.Graph, x, y, sum: bool = False) -> float:
    """Compute the Preferential Attachment Index for 2 given nodes."""
    return G.degree[x] * G.degree[y] if not sum else G.degree[x] + G.degree[y]


def preferential_attachment(G: nx.Graph, sum: bool = False) -> csr_matrix:
    """Compute the Preferential Attachment Index for all nodes in the Graph.
    Each similarity value is defined as:

    .. math::
        S(x, y) = k_x k_y

    where \\(k_x\\) is the degree of the node \\(x\\).

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)
    sum: bool :
        Replace multiplication with summation when computing the index.
         (Default value = False)

    Returns
    -------
    S: csr_matrix : the Similarity Matrix (in sparse format)

    Notes
    -----
    The idea of preferential attachment is applied to generate a growing
    scale-free network. The term growing represents the incremental nature
    of nodes over time in the network. The likelihood incrementing new
    connection associated with a node \\(x\\) is proportional to
    \\(k_x\\), the degree of the node.

    This index shows the worst performance on most networks.
    The **simplicity** (as it requires the least information
    for the score calculation) and the computational time of this metric
    are the main advantages. PA shows better results if larger
    degree nodes are densely connected,
    and lower degree nodes are rarely connected.

    In the above equation, summation can also be used instead of
    multiplication as an aggregate function (`sum = True`).
    """
    size = G.number_of_nodes()
    S = lil_matrix((size, size))
    node_index_map = nodes_to_indexes(G)

    for x, y in nx.complement(G).edges():
        _x = node_index_map[x]
        _y = node_index_map[y]

        S[_x, _y] = __preferential_attachment(G, x, y, sum=sum)
        # S[y, x] = S[x, y]

    return S.tocsr()
