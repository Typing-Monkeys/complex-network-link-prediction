import networkx as nx
import numpy as np
from utils import to_adjacency_matrix


def lpi(G: nx.Graph, epsilon: float, n: int) -> np.ndarray:
    A = to_adjacency_matrix(G, sparse=False)

    # Initialize the result matrix to A^2 and result
    Ai = A @ A
    result = Ai

    # Calculate the remaining terms of the sum
    for i in range(3, n):
        Ai = A @ Ai
        result += epsilon**(i - 2) * Ai

    return result
