import networkx as nx
import numpy as np


def __common_neighbors(G: nx.Graph, x, y) -> int:
    return len(set(G[x]) & set(G[y]))


def __hub_depressed(G: nx.Graph, x, y) -> float:
    return __common_neighbors(G, x, y) / max(G.degree[x], G.degree[y])


def hub_depressed(G: nx.Graph) -> np.ndarray:
    size = G.number_of_nodes()
    S = np.zeros((size, size))

    for x in G:
        for y in G:
            S[x, y] = __hub_depressed(G, x, y)

    return S


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2, 3])
    graph.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])

    nx.draw(graph, with_labels=True)

    print(hub_depressed(graph))

    plt.show()
