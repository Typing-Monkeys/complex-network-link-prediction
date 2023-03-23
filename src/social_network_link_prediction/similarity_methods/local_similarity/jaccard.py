import networkx as nx
import numpy as np
from utils import nodes_to_indexes
from scipy.sparse import lil_matrix, csr_matrix
from .common_neighbors import __common_neighbors


def __jaccard(G: nx.Graph, x, y) -> float:
    return __common_neighbors(G, x, y) / len(set(G[x]).union(set(G[y])))


def jaccard(G: nx.Graph) -> csr_matrix:
    size = G.number_of_nodes()
    S = lil_matrix((size, size))
    name_index_map = list(nodes_to_indexes(G).items())

    # itera solo gli indici della matrice traingolare
    # superiore (la matrice è simmetrica)
    # Questo dobrebbe essere molto più veloce di un doppio
    # ciclo for
    for x, y in zip(*np.triu_indices(size)):
        x_node = name_index_map[x][0]
        y_node = name_index_map[y][0]

        S[x, y] = __jaccard(G, x_node, y_node)

    return S.tocsr()
