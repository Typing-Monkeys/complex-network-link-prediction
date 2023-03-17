import networkx as nx
import numpy as np
from utils import to_adjacency_matrix
from scipy.sparse import csr_matrix


def local_path_index(G: nx.Graph, epsilon: float,
                     n: int) -> np.ndarray | csr_matrix:
    A = to_adjacency_matrix(G, sparse=False)
    A = A @ A
    S = np.power(epsilon, 0) * (A)

    # Calculate the remaining terms of the sum
    for i in range(1, n - 2):
        A = A @ A
        S += np.power(epsilon, i) * (A)

    return S
