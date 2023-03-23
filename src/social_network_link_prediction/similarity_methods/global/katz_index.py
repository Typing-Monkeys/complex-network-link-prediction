import networkx as nx
import numpy as np
from scipy import sparse

def to_adjacency_matrix(G: nx.Graph, sparse=True):
    """
        Dato un grafo ritorna la relativa Matrice di Adiacenza
        
        ARGS
            G: grafo in formato Networkx
            sparse: se True, ritorna una matrice sparsa, altrimenti
                    un numpy array

        RET
            ritorna la matrice di adiacenza (sparse o numpy array)
    """
    
    return nx.adjacency_matrix(G) if sparse else nx.to_numpy_array(G)


def __power_method(A, max_iterations = 100, tol = 1e-12, verbose = False):
    n = A.shape[0]
    x = np.ones(n) / np.sqrt(n) # initialize a vector x
    r = A @ x - np.dot(A @ x, x) * x # residual initialization
    eigenvalue = np.dot(x, A @ x) # residual eigenvalue

    for i in range(max_iterations):
        # Compute the new vector x
        x = A @ x
        # vector normalization
        x = x / np.linalg.norm(x)
        
        # Residual and eigenvalue computation
        r = A @ x - np.dot(A @ x, x) * x
        eigenvalue = np.dot(x, A @ x)
        
        # If the norm of r is less than the tolerance, break out of the loop.
        if np.linalg.norm(r) < tol:
            if verbose:
                print('Computation done after {i} steps')
            break

    return eigenvalue, x


def katz_index(G: nx.Graph, beta: int = 1) -> np.ndarray:
    A = to_adjacency_matrix(G, sparse=False)
    largest_eigenvalue = __power_method(A) #lambda_1
    if beta >= largest_eigenvalue[0]:
        print('Warning, Beta should be less than {largest_eigenvalue}')

    eye = np.identity(A.shape[0])
    S = np.linalg.inv((eye - beta * A))
    print(S)
    S -= eye

    return S




if __name__ == "__main__":
    import matplotlib.pyplot as plt

    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2, 3,4,5,6,7, 8, 9 ,10,11])
    graph.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (4, 7), (6, 2), (1, 5), 
                          (5, 4), (7, 0),(9, 2), (9, 5), (9, 4), (9, 0) ,(8, 2), (8, 5), (8, 4), 
                          (7, 10),(11, 0),(11, 2), (9, 11), (11, 4) ])

    nx.draw(graph, with_labels=True)
    A = to_adjacency_matrix(graph)
    print(__power_method(A, 10, 1e-12))
    print(np.max(np.linalg.eig(A.todense())[0]))
    print()

    S = katz_index(graph)

    print(S)
    #print(S(0,0))
    plt.show()