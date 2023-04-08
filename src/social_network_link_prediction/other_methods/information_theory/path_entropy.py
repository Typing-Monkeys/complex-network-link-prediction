import networkx as nx
import math
from scipy.sparse import lil_matrix

def path_entropy(G:nx.Graph, max_path:int = 5):

    similarity_matrix = lil_matrix((G.number_of_nodes(), G.number_of_nodes()))

    missing_edges = list(nx.complement(G).edges())

    #nodes = list(G.nodes())
    #n = len(nodes)
    for elem in missing_edges:
        paths = list(nx.all_simple_paths(G, elem[0], elem[1], max_path))
        n_paths = len(paths)
        M = G.number_of_edges()
        k_a = G.degree(elem[0])
        k_b = G.degree(elem[1])
        if n_paths == 0:
            return 0
        entropy = 0
        probability = 0
        weight = 1
        for path in paths:
            tmp = 1
            if len(path) > 2:
                weight = 1 / (1 - len(path))
            else:
                weight = 1
            for i in range(k_b):
                tmp = tmp * (((M - k_a) - i - 1)/(M - i - 1))
            probability = 1 - tmp
            entropy += weight * (-1) * math.log2(probability)
        similarity_matrix[elem[0],elem[1]] = entropy
    return similarity_matrix

if __name__ == "__main__":
    # Load a test graph
    graph = nx.karate_club_graph()

    predicted_adj_matrix = path_entropy(graph)
    # TODO: capire bene come fare la matrice di 
    # similarità perchè qui ci sono solo li archi non presenti prima
    # e non tutti quelli del grafo
    print(predicted_adj_matrix)