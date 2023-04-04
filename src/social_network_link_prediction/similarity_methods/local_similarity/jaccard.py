import networkx as nx
import numpy as np
from social_network_link_prediction.utils import nodes_to_indexes
from scipy.sparse import lil_matrix, csr_matrix
from .common_neighbors import __common_neighbors


def __jaccard(G: nx.Graph, x, y) -> float:
    """Compute the Jaccard Coefficient for 2 given nodes."""
    return __common_neighbors(G, x, y) / len(set(G[x]).union(set(G[y])))


def jaccard(G: nx.Graph) -> csr_matrix:
    """Compute the Jaccard Coefficient for all nodes in the Graph.
    Each similarity value is defined as:

    .. math::
        S(x, y) = \\frac{|\\Gamma(x) \\cap \\Gamma(y)|}
        {|\\Gamma(x) \\cup \\Gamma(y)|}

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
    The Jaccard coefficient is defined as the probability of selection
    of common neighbors of pairwise vertices from all the neighbors of
    either vertex. The pairwise Jaccard score increases with the number of
    common neighbors between the two vertices considered. Some researcher
    (**Liben-Nowell et al.**) demonstrated that this similarity metric
    performs worse as compared to Common Neighbors.
    """
    size = G.number_of_nodes()
    S = lil_matrix((size, size))
    name_index_map = list(nodes_to_indexes(G).items())

    # itera solo gli indici della matrice traingolare
    # superiore (la matrice è simmetrica)
    # Questo dobrebbe essere molto più veloce di un doppio
    # ciclo for
    for x, y in zip(*np.triu_indices(size)):
        x_node = name_index_map[x][0]
        y_node = name_index_map[y][0]

        S[x, y] = __jaccard(G, x_node, y_node)
        S[y, x] = S[x, y]

    return S.tocsr()
