import networkx as nx
import numpy as np
from social_network_link_prediction.utils import nodes_to_indexes
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
    name_index_map = list(nodes_to_indexes(G).items())

    for x, y in zip(*np.triu_indices(size)):
        x_node = name_index_map[x][0]
        y_node = name_index_map[y][0]

        S[x, y] = __common_neighbors(G, x_node, y_node)
        S[y, x] = S[x, y]

    return S.tocsr()
