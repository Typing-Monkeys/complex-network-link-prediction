import networkx as nx
import numpy as np
from social_network_link_prediction.utils import nodes_to_indexes
from scipy.sparse import lil_matrix, csr_matrix


def __path_of_length_three_iter(G: nx.Graph, x, y) -> float:
    """Compute the Path of Length Three Index for 2 given nodes."""
    k_x = G.degree(x)
    k_y = G.degree(y)

    score = 0

    # Enroll all neighbors
    for u in G[x]:
        for v in G[y]:
            # Base case
            if u == v:
                continue

            # Calcolate the score with the multiply of
            # value of node and divide for degree
            if G.has_edge(u, v):
                a_xu = G[x][u].get(
                    'weight',
                    1)  # prende il 'peso' dell'arco (1 se non ha peso)
                a_uv = G[u][v].get('weight', 1)
                a_vy = G[v][y].get('weight', 1)

                score += (a_xu * a_uv * a_vy) / (k_x * k_y)

    return score


def path_of_length_three(G: nx.Graph) -> csr_matrix:
    """Compute the Path of Length Three Index for all nodes in the Graph.
    Each similarity value is defined as:

    .. math::
        S(x, y) = \\sum \\frac{a_{x,u}a_{u,v}a_{v,y}}{k_u k_v}

    where \\(k_x\\) is the degree of node \\(x\\).

    Parameters
    ----------
    G: nx.Graph :
        grafo da analizzare

    Returns
    -------
    S: csr_matrix : matrice di Similarità

    Notes
    -----
    Georg Simmel, a German sociologist, first coined the concept
    “triadic closure” and made popular by Mark Granovetter in his work
    “The Strength of Weak Ties”. The authors proposed a similarity index
    in protein-protein interaction (PPI) network,
    called path of length 3 (or L3) published in the
    Nature Communication. They experimentally show that the triadic closure
    principle (TCP) does not work well with PPI networks. They showed the
    paradoxical behavior of the TCP (i.e., the path of length 2),
    which does not follow the structural and evolutionary mechanism that
    governs protein interaction. The TCP predicts well to the interaction of
    self-interaction proteins (SIPs), which are very small (4%) in PPI networks
    and fails in prediction between SIP and non SIP that amounts to 96%.
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
