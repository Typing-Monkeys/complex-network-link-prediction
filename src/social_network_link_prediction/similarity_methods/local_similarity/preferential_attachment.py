import networkx as nx
import numpy as np


def __preferential_attachment(G: nx.Graph, x, y, sum=False) -> float:
    return G.degree[x] * G.degree[y] if not sum else G.degree[x] + G.degree[y]


def preferential_attachment(G: nx.Graph, sum=False) -> np.ndarray:
    size = G.number_of_nodes()
    S = np.zeros((size, size))

    for x in G:
        for y in G:
            S[x, y] = __preferential_attachment(G, x, y, sum=sum)

    return S


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2, 3])
    graph.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])

    nx.draw(graph, with_labels=True)

    print(preferential_attachment(graph, sum=False))

    plt.show()
