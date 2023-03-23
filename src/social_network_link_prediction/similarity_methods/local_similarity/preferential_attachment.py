import networkx as nx
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from utils import nodes_to_indexes


def __preferential_attachment(G: nx.Graph, x, y, sum=False) -> float:
    return G.degree[x] * G.degree[y] if not sum else G.degree[x] + G.degree[y]


def preferential_attachment(G: nx.Graph, sum=False) -> csr_matrix:
    size = G.number_of_nodes()
    S = lil_matrix((size, size))
    name_index_map = list(nodes_to_indexes(G).items())

    for x, y in zip(*np.triu_indices(size)):
        x_node = name_index_map[x][0]
        y_node = name_index_map[y][0]

        S[x, y] = __preferential_attachment(G, x_node, y_node, sum=sum)
        S[y, x] = S[x, y]
        
    return S.tocsr()