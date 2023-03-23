import networkx as nx
import numpy as np
from utils import nodes_to_indexes
from scipy.sparse import lil_matrix, csr_matrix


def __common_neighbors(G: nx.Graph, x, y) -> int:
    """
        Calcola l'indice Common Neighbors per la signola
        coppia di nodi
    """

    return len(set(G[x]).intersection(set(G[y])))


def common_neighbors(G: nx.Graph) -> csr_matrix:
    """
        Cacola l'indice Common Neighbors per tutti i nodi 
        del grafo dato.

        ARGS:
            G: grafo networkx 

        RET:
            S np.ndarray: matrice di Similarità 
    """

    size = G.number_of_nodes()
    S = lil_matrix((size, size))
    name_index_map = nodes_to_indexes(G)

    mem = set()    # memoization delle celle calcolate

    for x in G:
        for y in G:
            _x = name_index_map[x]
            _y = name_index_map[y]
            
            if (_x, _y) not in mem:
                S[_x, _y] = __common_neighbors(G, x, y)
                S[_y, _x] = S[_x, _y]

                mem.add((_y, _x))

    return S.tocsr()
