import networkx as nx
import numpy as np
from social_network_link_prediction.utils import to_adjacency_matrix
from scipy.sparse import csr_matrix


def local_path_index(G: nx.Graph, epsilon: float, n: int) -> csr_matrix:
    """

    Parameters
    ----------
    G: nx.Graph :
        
    epsilon: float :
        
    n: int :
        

    Returns
    -------

    """
    A = to_adjacency_matrix(G)
    A = A @ A
    S = np.power(epsilon, 0) * (A)

    # Calculate the remaining terms of the sum
    for i in range(1, n - 2):
        A = A @ A
        S += np.power(epsilon, i) * (A)

    return S.tocsr()
