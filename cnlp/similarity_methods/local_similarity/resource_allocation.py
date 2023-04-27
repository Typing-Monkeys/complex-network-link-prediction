import networkx as nx
import numpy as np
from cnlp.utils import nodes_to_indexes
from scipy.sparse import lil_matrix, csr_matrix


def __resource_allocation(G: nx.Graph, x, y) -> float:
    """Compute the Resource Allocation Index for 2 given nodes."""
    return sum([1 / G.degree[z] for z in set(G[x]) & set(G[y])])


def resource_allocation(G: nx.Graph) -> csr_matrix:
    """Compute the Resource Allocation Index for all nodes in the Graph.
    Each similarity value is defined as:

    .. math::
        S(x, y) = \\sum_{z \\in \\Gamma(x) \\cap \\Gamma(y)} \\frac{1}{k_z}

    where \\(\\Gamma(x)\\) are the neighbors of node \\(x\\) and
    \\(k_x\\) is the degree of the node \\(x\\).

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)

    Returns
    -------
    S: csr_matrix : the Similarity Matrix (in sparse format)

    Notes
    -----
    Consider two non-adjacent vertices \\(x\\) and \\(y\\).
    Suppose node \\(x\\) sends some resources to \\(y\\)
    through the common nodes of both \\(x\\) and \\(y\\)
    then the similarity between the two vertices is computed in terms
    of resources sent from \\(x\\) to \\(y\\).

    The difference between Resource Allocation (**RA**) and
    Adamic and Adar (**AA**) is that the RA index
    heavily penalizes to higher degree nodes compared to the AA index.
    Prediction results of these indices become almost the same
    for smaller average degree networks.

    This index shows good performance on heterogeneous
    networks with a high clustering coefficient, especially
    on transportation networks.
    """
    size = G.number_of_nodes()
    S = lil_matrix((size, size))
    node_index_map = nodes_to_indexes(G)

    for x, y in nx.complement(G).edges():
        _x = node_index_map[x]
        _y = node_index_map[y]

        S[_x, _y] = __resource_allocation(G, x, y)
        # S[y, x] = S[x, y]

    return S.tocsr()
