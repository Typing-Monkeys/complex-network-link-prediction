import networkx as nx
import numpy as np
from social_network_link_prediction.utils import to_adjacency_matrix, only_unconnected
from scipy.sparse import csr_matrix


def local_path_index(G: nx.Graph, epsilon: float, n: int) -> csr_matrix:
    """Compute the Local Path  Index for all nodes in the Graph.
    Each similarity value is defined as:

    .. math::
        S^{LP} = A^{2} + \\epsilon A^{3} +
        \\epsilon^{2} A^{4} + \\ldots + \\epsilon^{n - 2} A^{n}

    where \\(\\epsilon\\) is a free parameter,
    \\(A\\) is the Adjacency Matrix and \\(n\\) is the maximal order.

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)
    epsilon: float :
        free parameter
    n: int :
        maximal order

    Returns
    -------
    S: csr_matrix : the Similarity Matrix (in sparse format)

    Notes
    -----
    This metric has the intent to furnish a good trade-off
    between accuracy and computational complexity.

    Clearly, the measurement converges to common neighbor when
    \\(\\epsilon = 0\\). If there is no direct connection between
    \\(x\\) and \\(y\\), \\((A^{3})_{x,y}\\) is equated to the total
    different paths of length 3 between \\(x\\) and \\(y\\).

    Computing this index becomes more complicated with the increasing
    value of \\(n\\). The LP index outperforms the proximity-based indices,
    such as RA, AA, and CN.
    """
    A = to_adjacency_matrix(G)
    A = A @ A
    S = np.power(epsilon, 0) * (A)

    # Calculate the remaining terms of the sum
    for i in range(1, n - 2):
        A = A @ A
        S += np.power(epsilon, i) * (A)

    return only_unconnected(G, S.tocsr())
