import networkx as nx
import numpy as np
from scipy.sparse import csc_matrix


def to_adjacency_matrix(G: nx.Graph,
                        sparse: bool = True) -> csc_matrix | np.ndarray:
    """Convert a ginven Graph in to its Adjacency Matrix

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)
    sparse: bool:
        if True, return the Adjacency Matrix in sparse format,
        otherwise in full format.
         (Default value = True)

    Returns
    -------
    csc_matrix | np.ndarray: the Adjacency Matrix
    """
    # TODO: ricontrollare se i pesi servono
    return nx.adjacency_matrix(G, weight=None) if sparse else nx.to_numpy_array(G, weight=None)
