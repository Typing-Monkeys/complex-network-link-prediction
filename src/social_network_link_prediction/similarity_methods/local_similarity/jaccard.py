import networkx as nx
import numpy as np


# TODO: vedere se si può riutilizzare codice
#       da Common Nieghbors
def __common_neighbors(G: nx.Graph, x, y) -> int:
    return len(set(G[x]) & set(G[y]))


def __jaccard(G: nx.Graph, x, y) -> float:
    return __common_neighbors(G, x, y) / len(set(G[x]).union(set(G[y])))


# TODO: utilizzare matrici sparse
def jaccard(G: nx.Graph) -> np.ndarray:
    size = G.number_of_nodes()
    S = np.zeros((size, size))

    for x in G:
        for y in G:
            S[x, y] = __jaccard(G, x, y)

    return S


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2, 3])
    graph.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])

    nx.draw(graph, with_labels=True)

    print(jaccard(graph))

    plt.show()
