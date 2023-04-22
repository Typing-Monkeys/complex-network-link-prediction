import networkx as nx
from scipy.sparse import csr_matrix, lil_matrix, linalg, identity
from social_network_link_prediction.utils import nodes_to_indexes, to_adjacency_matrix, only_unconnected


def rooted_page_rank(G: nx.Graph, alpha: float = .5) -> csr_matrix:
    """Compute the Rooted Page Rank for all nodes in the Graph.
    This score is defined as:

    .. math::
        S = (1 - \\alpha) (I - \\alpha \\hat{N})^{-1}

    where \\(\\hat{N} = D^{-1}A\\) is the normalized
    Adjacency Matrix with the diagonal degree matrix
    \\(D[i,i] = \\sum_j A[i,j]\\)

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)
    alpha: float :
        random walk probability
         (Default value = 0.5)

    Returns
    -------
    S: csr_matrix : the Similarity Matrix (in sparse format)

    Notes
    -----
    The idea of PageRank was originally proposed to rank the web pages based on
    the importance of those pages. The algorithm is based on the assumption that
    a random walker randomly goes to a web page with probability \\(\\alpha\\)
    and follows hyper-link embedded in the page with probability \\( (1 - \\alpha ) \\).
    Chung et al. used this concept incorporated with a random walk in
    link prediction framework. The importance of web pages, in a random walk,
    can be replaced by stationary distribution. The similarity between two vertices
    \\(x\\) and \\(y\\) can be measured by the stationary probability of
    \\(x\\) from \\(y\\) in a random walk where the walker moves to an
    arbitrary neighboring vertex with probability \\(\\alpha\\)
    and returns to \\(x\\) with probability \\( ( 1 - \\alpha )\\).
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

    return only_unconnected(G, S.tocsr())
