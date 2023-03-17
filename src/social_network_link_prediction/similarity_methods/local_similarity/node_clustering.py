import networkx as nx
import numpy as np
from utils import to_adjacency_matrix


# TODO: questo metodo funziona solo se le eitchette dei nodi sono interi
def __t(z):
    """
        Numero di Triangoli passanti per il Nodo Z
    """

    # Il seguente link ci dice che per trovare i triangoli
    # passanti per un nodo basta fare A^3, prendere il valore
    # in A[z,z] e dividerlo per 2 (ci sono 2 modi per percorrere un Triangolo)
    # https://github.com/dougissi/counting-triangles

    # TODO: ricontrollare
    return A3[z, z] / 2  # TODO: potrebbe essere / fact(A[z,z])


def __C(G: nx.Graph, z):
    z_degree = G.degree[z]
    return __t(z) / (z_degree * (z_degree - 1))


def __node_clustering(G: nx.Graph, x, y):
    return sum([__C(G, z) for z in (set(G[x]) & set(G[y]))])


def node_clustering(G: nx.Graph) -> np.ndarray:
    global A3

    size = G.number_of_nodes()
    S = np.zeros((size, size))

    # variabile globale che servirà in __t()
    A3 = to_adjacency_matrix(G, sparse=False)  # TODO: sparse True ??
    A3 = A3 @ A3 @ A3  # A^3

    for x in G:
        for y in G:
            S[x, y] = __node_clustering(G, x, y)

    return S


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2, 3])
    graph.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])

    nx.draw(graph, with_labels=True)

    print(node_clustering(graph))

    plt.show()
