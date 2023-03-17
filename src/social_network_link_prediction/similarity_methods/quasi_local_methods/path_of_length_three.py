import networkx as nx
import numpy as np
from utils import nodes_to_indexes


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
                a_xu = G[x][u].get('weight', 1)  # Change this for unweighted G
                a_uv = G[u][v].get('weight', 1)  # Change this for unweighted G
                a_vy = G[v][y].get('weight', 1)  # Change this for unweighted G

                score += (a_xu * a_uv * a_vy) / (k_x * k_y)

    return score


def path_of_length_three(G:nx.Graph):
    size = G.number_of_nodes()
    S = np.full((size, size), None)
    name_index_map = nodes_to_indexes(G) 

    for x in G:
        for y in G:
            _x = name_index_map[x]
            _y = name_index_map[y]

            if S[_x, _y] is None:
                S[_x, _y] = __path_of_length_three_iter(G, x, y)
                S[_y, _x] = S[_x, _y]

    return S