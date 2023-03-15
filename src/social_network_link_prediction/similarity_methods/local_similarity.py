import networkx as nx
import numpy as np


## --- Common Neighbors Index 

def __common_neighbors(G: nx.Graph, x, y) -> int:
    """
        Calcola l'indice Common Neighbors per la signola
        coppia di nodi
    """

    return len(set(G[x]) & set(G[y]))

def common_neighbors(G: nx.Graph) -> np.ndarray :
    """
        Cacola l'indice Common Neighbors per tutti i nodi 
        del grafo dato.

        ARGS:
            G: grafo networkx 

        RET:
            S np.ndarray: matrice di Similarità 
    """

    size = G.number_of_nodes()
    S = np.zeros((size, size))

    for x in G:
        for y in G:
            S[x, y] = __common_neighbors(G, x, y)

    return S


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    graph = nx.Graph()
    graph.add_nodes_from([0, 1, 2, 3])
    graph.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])
    
    nx.draw(graph, with_labels=True)

    print(common_neighbors(graph))

    plt.show()
