import networkx as nx
import numpy as np
from social_network_link_prediction.utils import nodes_to_indexes
from scipy.sparse import lil_matrix, csr_matrix


def __path_of_length_three_iter(G: nx.Graph, x, y) -> float:
    k_x = G.degree(x)
    k_y = G.degree(y)

    score = 0

    # Enroll all neighbors
    for u in G[x]:
        for v in G[y]:
            # Base case
            if u == v:
                continue

            # Calcolate the score with the multiply of value of node and divide for degree
            if G.has_edge(u, v):
                a_xu = G[x][u].get(
                    'weight',
                    1)  # prende il 'peso' dell'arco (1 se non ha peso)
                a_uv = G[u][v].get('weight', 1)
                a_vy = G[v][y].get('weight', 1)

                score += (a_xu * a_uv * a_vy) / (k_x * k_y)

    return score


def path_of_length_three(G: nx.Graph) -> csr_matrix:
    """TODO

    Parameters
    ----------
    G: nx.Graph :
        grafo da analizzare

    Returns
    -------
    S: csr_matrix : matrice di Similarità
    """
    size = G.number_of_nodes()
    S = lil_matrix((size, size))
    name_index_map = list(nodes_to_indexes(G).items())

    for x, y in zip(*np.triu_indices(size)):
        x_node = name_index_map[x][0]
        y_node = name_index_map[y][0]

        S[x, y] = __path_of_length_three_iter(G, x_node, y_node)
        S[y, x] = S[x, y]

    return S.tocsr()
