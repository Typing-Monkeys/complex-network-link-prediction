import networkx as nx
import numpy as np
from scipy.sparse import csc_matrix


def to_adjacency_matrix(G: nx.Graph,
                        sparse: bool = True) -> csc_matrix | np.ndarray:
    """Dato un grafo ritorna la relativa Matrice di Adiacenza

    Parameters
    ----------
    G: nx.Graph :
        grafo di cui ottenere la matrice di adiacenza
    sparse: bool:
        Indica se ritornare una Sparse Matrix (True) o Full Matrix (False)
         (Default value = True)

    Returns
    -------
    sparse | np.ndarray: matrice di adiacenza
    """
    return nx.adjacency_matrix(G) if sparse else nx.to_numpy_array(G)
