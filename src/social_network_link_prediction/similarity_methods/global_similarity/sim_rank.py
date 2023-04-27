import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, identity, csr_matrix
from social_network_link_prediction.utils import only_unconnected


def init_similarity_matrix(G: nx.Graph, n: int) -> lil_matrix:
    """Generate an Identity matrix: the starting Similarity
    Matrix.

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)
    n: int :
       the new matrix size

    Returns
    -------
    sim_matrix: lil_matrix : the starting Similarity Matrix
    """
    # inizializzo la matrice similarity
    # gli elementi con loro stessi (lungo la diagonale) hanno similarità massima
    sim_matrix = identity(n).tolil()
    return sim_matrix


def compute_sim_rank(G: nx.Graph,
                     a,
                     b,
                     sim_matrix: lil_matrix,
                     C: int = 0.8) -> float:
    """Compute the Sim Rank method between the given
    nodes a and b.

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)
    a :
       first node
    b :
       second node
    sim_matrix: lil_matrix :
        the similarity matrix
    C: int :
        free parameter
         (Default value = 0.8)

    Returns
    -------
    new_SimRank: float : the SimRank value between a and b
    """

    # se i nodi sono uguali allora similarità massima
    if (a == b):
        return 1

    a_neigh = list(G.neighbors(a))
    b_neigh = list(G.neighbors(b))
    len_a = len(a_neigh)
    len_b = len(b_neigh)

    # nodi isolati hanno similarità 0
    if (len_a == 0 or len_b == 0):
        return 0

    # mi recupero e sommo i valori di similarità calcolati in precedenza
    simRank_sum = 0
    for i in a_neigh:
        for j in b_neigh:
            simRank_sum += sim_matrix[i, j]
    # moltiplico secondo la definizione del paper
    scale = C / (len_a * len_b)
    new_SimRank = scale * simRank_sum
    return new_SimRank


def sim_rank(G: nx.Graph,
             k: int = 5,
             cutoff: int = 4,
             c: int = 0.8) -> csr_matrix:
    """Compute the SimRank index for all the nodes in the Graph.

    This method is defined as:

    .. math::
        S(x, y) = \\begin{cases}
                \\frac{\\alpha}{k_x k_y} \\sum_{i=1}^{k_x} \\sum_{j=1}^{k_y}
                    S( \\Gamma_i(x), \\Gamma_j(y)) & x \\neq y \\\\
                1 & x = y
            \\end{cases}

    where \\( \\alpha \\in (0,1) \\) is a constant. \\(\\Gamma_i(x)\\) and \\( \\Gamma_j(y) \\)
    are the ith and jth elements in the neighborhood sets.

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)
    k: int :
         (Default value = 5)
    cutoff: int :
         (Default value = 4)
    c: int :
         (Default value = 0.8)

    Returns
    -------
    sim_matrix: csr_matrix : the Similarity Matrix (in sparse format)
    """
    G = nx.convert_node_labels_to_integers(G, 0)

    nodes_num = G.number_of_nodes()
    sim_matrix = init_similarity_matrix(G, nodes_num)

    for a in range(nodes_num):
        for b in range(nodes_num):
            # fa pruning evitando di calcolare la similarità di archi a distanza maggiore di 5
            if (nx.has_path(G, a, b)
                    and (nx.shortest_path_length(G, a, b) > cutoff)):
                sim_matrix[a, b] = 0
            else:
                # se non deve fare pruning si calcola il valore di similarità per i nodi a e b
                for i in range(k):
                    sim_matrix[a, b] = compute_sim_rank(G,
                                                        a,
                                                        b,
                                                        sim_matrix=sim_matrix,
                                                        C=c)

    # imposta a 0 gli elementi della diagonale che prima avevano similarità uguale ad 1
    for a in range(nodes_num):
        sim_matrix[a, a] = 0
    return only_unconnected(G, csr_matrix(sim_matrix))


if __name__ == "__main__":

    # G = nx.karate_club_graph()
    G = nx.gnp_random_graph(150, .01)

    # converte gli id dei nodi in interi che partono da 0
    res = sim_rank(G, k=5)
    # stampa il cammino che è considerato più probabile
    # print(f"Il link più probabile è quello tra i nodi {np.where(res_sparse==res_sparse.max())} , con un valora di similarità di {res_sparse.max()}")
    print(res)
    nx.draw(G, with_labels=True)
    plt.show()
