import networkx as nx
from scipy.sparse import csr_matrix, lil_matrix, linalg, identity
from social_network_link_prediction.utils import to_adjacency_matrix
from social_network_link_prediction.utils import nodes_to_indexes


def rooted_page_rank(G: nx.Graph, alpha: float = .5) -> csr_matrix:
    """Compute the Rooted Page Rank for all nodes in the Graph.
    Each similarity value is defined as:

    .. math::
        S(x, y) = ...

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)
    alpha: float :
         (Default value = 0.5)

    Returns
    -------
    S: csr_matrix : the Similarity Matrix (in sparse format)

    Notes
    -----
    """
    A = to_adjacency_matrix(G)
    D = lil_matrix(A.shape)

    nodes_to_indexes_map = nodes_to_indexes(G)
    for node in G.nodes():
        D[nodes_to_indexes_map[node],
          nodes_to_indexes_map[node]] = G.degree[node]

    D = D.tocsc()
    N_hat = linalg.inv(D) @ A.tocsc()
    eye = identity(A.shape[0], format='csc')
    S = (1 - alpha) * linalg.inv(eye - alpha * N_hat)

    return S.tocsr()
