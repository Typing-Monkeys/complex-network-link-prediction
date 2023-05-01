"""Collection of Quasi-local Similarity Methods for Link Prediction.

Quasi-local indices have been introduced as a trade-off between
local and global approaches or performance and complexity.
These metrics are as efficient to compute as local indices.
Some of these indices extract the entire topological information
of the network.

The time complexities of these indices are still below compared
to the global approaches.
"""
import networkx as nx
import numpy as np
from cnlp.utils import to_adjacency_matrix, nodes_to_indexes, only_unconnected
from scipy.sparse import csr_matrix, lil_matrix


def local_path_index(G: nx.Graph, epsilon: float, n: int) -> csr_matrix:
    """Compute the Local Path  Index for all nodes in the Graph.
    Each similarity value is defined as:

    .. math::
        S^{LP} = A^{2} + \\epsilon A^{3} +
        \\epsilon^{2} A^{4} + \\ldots + \\epsilon^{n - 2} A^{n}

    where \\(\\epsilon\\) is a free parameter,
    \\(A\\) is the Adjacency Matrix and \\(n\\) is the maximal order.

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)
    epsilon: float :
        free parameter
    n: int :
        maximal order

    Returns
    -------
    S: csr_matrix : the Similarity Matrix (in sparse format)

    Notes
    -----
    This metric has the intent to furnish a good trade-off
    between accuracy and computational complexity.

    Clearly, the measurement converges to common neighbor when
    \\(\\epsilon = 0\\). If there is no direct connection between
    \\(x\\) and \\(y\\), \\((A^{3})_{x,y}\\) is equated to the total
    different paths of length 3 between \\(x\\) and \\(y\\).

    Computing this index becomes more complicated with the increasing
    value of \\(n\\). The LP index outperforms the proximity-based indices,
    such as RA, AA, and CN.
    """
    A = to_adjacency_matrix(G)
    A = A @ A
    S = np.power(epsilon, 0) * (A)

    # Calculate the remaining terms of the sum
    for i in range(1, n - 2):
        A = A @ A
        S += np.power(epsilon, i) * (A)

    return only_unconnected(G, lil_matrix(S))


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

    size = G.number_of_nodes()
    S = lil_matrix((size, size))
    name_index_map = nodes_to_indexes(G)

    for x, y in nx.complement(G).edges():
        _x = name_index_map[x]
        _y = name_index_map[y]

        S[_x, _y] = __path_of_length_three_iter(G, x, y)
        S[_y, _x] = S[_x, _y]

    return S.tocsr()
