import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, identity


def init_similarity_matrix(G:nx.Graph, n):
    #inizializzo la matrice similarity
    # gli elementi con loro stessi (lungo la diagonale) hanno similarità massima
    sim_matrix = identity(n).tolil()
    return sim_matrix



# implementazione iterativa 
def compute_sim_rank(G:nx.Graph, a, b, sim_matrix, C = 0.8):

     #se i nodi sono uguali allora similarità massima
    if(a == b):
        return 1

    a_neigh = list(G.neighbors(a))
    b_neigh = list(G.neighbors(b))
    len_a = len(a_neigh)
    len_b = len(b_neigh)
    #nodi isolati hanno similarità 0
    if(len_a == 0 or len_b == 0):
        return 0

    #mi recupero e sommo i valori di similarità calcolati in precedenza
    simRank_sum = 0
    for i in a_neigh:
        for j in b_neigh:
            simRank_sum += sim_matrix[i, j]
    # moltiplico secondo la definizione del paper
    scale = C / (len_a * len_b)
    new_SimRank = scale * simRank_sum
    return new_SimRank



def sim_rank(G:nx.Graph, k = 5, cutoff = 4, c = 0.8):

    nodes_num = G.number_of_nodes()
    sim_matrix = init_similarity_matrix(G, nodes_num)

    for a in range(nodes_num):
        for b in range(nodes_num):
            # fa pruning evitando di calcolare la similarità di archi a distanza maggiore di 5
            if((nx.shortest_path_length(G, a, b) > cutoff)):
                sim_matrix[a, b] = 0
            else:
                # se non deve fare pruning si calcola il valore di similarità per i nodi a e b
                for i in range(k):
                    sim_matrix[a, b] = compute_sim_rank(G, a, b, sim_matrix = sim_matrix, C = c)
    return sim_matrix




if __name__ == "__main__":


    G = nx.karate_club_graph()
    # converte gli id dei nodi in interi che partono da 0
    G = nx.convert_node_labels_to_integers(G,0)

    res_tmp = sim_rank(G, k = 5)
    tmp = np.zeros((G.number_of_nodes(),G.number_of_nodes()))
    res = lil_matrix(tmp,(G.number_of_nodes(),G.number_of_nodes()))
    # crea una nuova matrice di similarità contenente solo le coppie di nodi che non hanno già un cammino
    for i,j in nx.complement(G).edges():
        res[i,j] = res_tmp[i,j]
    res = res.toarray()
    # stampa il cammino che è considerato più probabile
    print(f"Il link più probabile è quello tra i nodi {np.where(res==res.max())} , con un valora di similarità di {res.max()}")
    
    nx.draw(G, with_labels=True)
    plt.show()