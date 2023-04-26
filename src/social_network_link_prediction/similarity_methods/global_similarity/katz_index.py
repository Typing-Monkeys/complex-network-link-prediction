import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, linalg, identity
from social_network_link_prediction.utils import to_adjacency_matrix, only_unconnected


def __power_method(A: csr_matrix,
                   max_iterations: int = 100,
                   tol: float = 1e-12,
                   verbose: bool = False):
    """Perfome the Power Method"""
    n = A.shape[0]
    x = np.ones(n) / np.sqrt(n)  # initialize a vector x
    # r = A @ x - np.dot(A @ x, x) * x # residual initialization
    r = A @ x - ((A @ x) @ x) * x
    eigenvalue = x @ (A @ x)  # residual eigenvalue
    # eigenvalue = np.dot(x, A @ x) # residual eigenvalue

    for i in range(max_iterations):
        # Compute the new vector x
        x = A @ x
        # vector normalization
        x = x / np.linalg.norm(x)

        # Residual and eigenvalue computation
        r = A @ x - ((A @ x) @ x) * x
        eigenvalue = x @ (A @ x)
        # r = A @ x - np.dot(A @ x, x) * x
        # eigenvalue = np.dot(x, A @ x)

        # If the norm of r is less than the tolerance, break out of the loop.
        if np.linalg.norm(r) < tol:
            if verbose:
                print('Computation done after {i} steps')
            break

    return eigenvalue, x


def katz_index(G: nx.Graph, beta: int = 1) -> csr_matrix:
    """Compute the Katz Index for all nodes in the Graph.
    Each similarity value is defined as:

    .. math::
        S(x, y) = (I - \\beta A)^{-1} - I

    where \\(I\\) is the Identity Matrix,
    \\(\\beta\\) is a dumping factor that controls the path weights
    and \\(A\\) is the Adjacency Matrix

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)
    beta: int :
        Dumping Factor
         (Default value = 1)

    Returns
    -------
    S: csr_matrix : the Similarity Matrix (in sparse format)

    Notes
    -----
    For the convergence of above equation,

    .. math::
        \\beta < \\frac{1}{\\lambda_1}

    where \\(\\lambda_1\\) is the maximum eighenvalue of the matrix \\(A\\).

    The computational complexity of the given metric is high,
    and it can be roughly estimated to be cubic complexity
    which is not feasible for a large network.
    """
    A = to_adjacency_matrix(G)
    largest_eigenvalue = __power_method(A)  # lambda_1
    if beta >= (1 / largest_eigenvalue[0]):
        print('Warning, Beta should be less than {largest_eigenvalue}')

    eye = identity(A.shape[0], format='csc')
    S = linalg.inv((eye - beta * A.tocsc())) - eye

    return only_unconnected(G, csr_matrix(S))
