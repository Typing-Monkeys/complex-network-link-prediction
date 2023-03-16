import networkx as nx
import numpy as np


def __resource_allocation(G: nx.Graph, x, y) -> float:
    return sum([1 / G.degree[z] for z in set(G[x]) & set(G[y])])


def resource_allocation(G: nx.Graph) -> np.ndarray:
    size = G.number_of_nodes()
    S = np.zeros((size, size))

    for x in G:
        for y in G:
            S[x, y] = __resource_allocation(G, x, y)

    return S


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2, 3])
    graph.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])

    nx.draw(graph, with_labels=True)

    print(resource_allocation(graph))

    plt.show()
