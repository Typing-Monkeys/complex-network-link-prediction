import networkx as nx
import math
from scipy.sparse import lil_matrix

def path_entropy(G:nx.Graph, max_path:int = 3) -> lil_matrix:

    similarity_matrix = lil_matrix((G.number_of_nodes(), G.number_of_nodes()))

    missing_edges = list(nx.complement(G).edges())

    for elem in missing_edges:
        paths = nx.all_simple_paths(G, elem[0], elem[1], max_path)
        tmp = 0
        for i in range(2,(max_path+1)):
            tmp += (1 / (i - 1)) * simple_path_entropy(paths = paths, G = G)
        tmp = tmp - new_link_entropy(G, elem[0], elem[1])
        similarity_matrix[elem[0],elem[1]] = tmp
    return similarity_matrix

# Calcola l'entropia data dalla probabilità che si vengano a creare i vari simple paths tra i nodi tra cui si 
# vuole fare link prediction
def simple_path_entropy(paths, G:nx.Graph) -> int: #paths è un generator ritornato dalla funzione nx.all_simple_paths()
    tmp = 0
    for path in paths:
        for a, b in list(nx.utils.pairwise(path)):
            tmp += new_link_entropy(G, a, b)
    return tmp

# Calcola l'entropia basata sulla probabilità a priori della creazione del link diretto
# tra le coppie di noti senza link diretti
def new_link_entropy(G:nx.Graph, a:int, b:int) -> int:
    deg_a = G.degree(a)
    deg_b = G.degree(b)
    M = G.number_of_edges()

    return (-1) * math.log2(1 - (math.comb(M-deg_a, deg_b)/math.comb(M, deg_b)))


if __name__ == "__main__":

    graph = nx.karate_club_graph()

    predicted_adj_matrix = path_entropy(graph)

    print(predicted_adj_matrix)