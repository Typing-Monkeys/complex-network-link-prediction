import networkx as nx
import numpy as np
from social_network_link_prediction.utils import nodes_to_indexes
from scipy.sparse import lil_matrix, csr_matrix


def __adamic_adar(G: nx.Graph, x, y) -> float:
    """Compute the Adamic and Adar Index for 2 given nodes."""
    return sum([1 / np.log(G.degree[z]) for z in set(G[x]) & set(G[y])])


def adamic_adar(G: nx.Graph) -> csr_matrix:
    """Compute the Adamic and Adar Index for all nodes in the Graph.
    Each similarity value is defined as:

    .. math::
        S(x, y) = \\sum_{z \\in \\Gamma(x) \\cap \\Gamma(y)}
        \\frac{1}{\\log k_z}

    where \\(k_z\\) is the degree of node \\(z\\)
    and \\(\\Gamma(x)\\) are the neighbors of the node \\(x\\).

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)

    Returns
    -------
    S: csr_matrix : the Similarity Matrix (in sparse format)

    Notes
    -----
    It is clear from the equation that more weights are assigned to
    the common neighbors having smaller degrees.
    This is also intuitive in the real-world scenario, for example,
    a person with more number of friends spend less time/resource
    with an individual friend as compared to the less number of friends.
    """
    size = G.number_of_nodes()
    S = lil_matrix((size, size))
    node_index_map = nodes_to_indexes(G)
    
    for x, y in nx.complement(G).edges():
        _x = node_index_map[x]
        _y = node_index_map[y]

        S[_x, _y] = __adamic_adar(G, x, y)

    return S.tocsr()
