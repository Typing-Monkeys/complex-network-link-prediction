import networkx as nx
import math
from scipy.sparse import lil_matrix, csr_matrix
from cnlp.utils import nodes_to_indexes
from typing import Generator, List


def path_entropy(G: nx.Graph, max_path: int = 3) -> csr_matrix:
    """Compute the Path Entropy Measure for all nodes in the Graph.

    This Similarity measure between two nodes \\(X\\) and \\(Y\\)
    is calculated with:

    .. math::
        S_{x,y} = -I(L^1_{xy}|U_{i=2}^{maxlen} D_{xy}^i)

    where \\(D^i_{xy}\\) represents the set consisting of all simple
    paths of length i between the two vertices and maxlen is the maximum
    length of simple path of the network.

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)
    max_path: int :
        maximal path length
         (Default value = 3)

    Returns
    -------
    similarity_matrix: csr_matix: the Similarity Matrix (in sparse format)
    """
    similarity_matrix = lil_matrix((G.number_of_nodes(), G.number_of_nodes()))
    nodes_to_indexes_map = nodes_to_indexes(G)
    missing_edges = list(nx.complement(G).edges())

    for elem in missing_edges:
        paths = nx.all_simple_paths(G, elem[0], elem[1], max_path)
        tmp = 0
        for i in range(2, (max_path + 1)):
            tmp += (1 / (i - 1)) * simple_path_entropy(paths=paths, G=G)
        tmp = tmp - new_link_entropy(G, elem[0], elem[1])
        similarity_matrix[nodes_to_indexes_map[elem[0]],
                          nodes_to_indexes_map[elem[1]]] = tmp
    return similarity_matrix.tocsr()


def simple_path_entropy(paths: Generator[list, None, None],
                        G: nx.Graph) -> float:
    """Calcola l'entropia data dalla probabilità che si vengano a creare
    i vari simple paths tra i nodi tra cui si
    vuole fare link prediction

    Parameters
    ----------
    paths: Generator[List] :
        generator ritornato dalla funzione nx.all_simple_paths()

    G: nx.Graph :
        input graph

    Returns
    -------
    float: simple path entropy
    """
    tmp = .0
    for path in paths:
        for a, b in list(nx.utils.pairwise(path)):
            tmp += new_link_entropy(G, a, b)
    return tmp


def new_link_entropy(G: nx.Graph, a, b) -> float:
    """Calcola l'entropia basata sulla probabilità
    a priori della creazione del link diretto
    tra le coppie di noti senza link diretti

    Parameters
    ----------
    G: nx.Graph :
        input graph
    a :
        first node
    b :
        second node

    Returns
    -------
    float: entropy between node A and B
    """
    deg_a = G.degree(a)
    deg_b = G.degree(b)
    M = G.number_of_edges()

    return -1 * math.log2(1 -
                          (math.comb(M - deg_a, deg_b) / math.comb(M, deg_b)))


if __name__ == "__main__":

    graph = nx.karate_club_graph()

    predicted_adj_matrix = path_entropy(graph)

    print(predicted_adj_matrix)
