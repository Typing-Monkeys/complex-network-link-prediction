import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix


def init_similarity_matrix(G:nx.Graph, n):
    #inizializzo la matrice similarity
    x = np.identity(n)
    sim_matrix = lil_matrix(x, (n, n))
    return sim_matrix



# problematiche con questo metodo, per grafi troppo grandi è lento o raggiunge la profondtà di ricorsione massima
def compute_sim_rank(G:nx.Graph, a, b, C = 0.8, k = 5):
    #print(k)
    if(k == 0):
        if(a == b):
            return 1
        else:
            return 0
    a_neigh = list(G.neighbors(a))
    b_neigh = list(G.neighbors(b))
    a_neigh_num = sum(1 for e in a_neigh)
    b_neigh_num = sum(1 for e in b_neigh)

    summation = 0
    for i in range(a_neigh_num):
        for j in range(b_neigh_num):
            summation = summation + compute_sim_rank(G, a_neigh[i], b_neigh[j], k = k-1)
    res = (C/(a_neigh_num*b_neigh_num))*summation
    
    return res



# implementazione iterativa 
def compute_sim_rank_iterative(G:nx.Graph, a, b, sim_matrix):

    C = 0.8
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



def sim_rank(G:nx.Graph, nodes_num = 0, sim_matrix = None):
    k = 5
    for a in range(nodes_num):
        for b in range(nodes_num):
            # fa pruning evitando di calcolare la similarità di archi a distanza maggiore di 5
            #if((a == b) or (nx.shortest_path_length(G, a, b) > 3)):
            if((nx.shortest_path_length(G, a, b) > 4)):
                sim_matrix[a, b] = 0
            else:
                for i in range(k):
                    sim_matrix[a, b] = compute_sim_rank_iterative(G, a, b, sim_matrix = sim_matrix)
    return sim_matrix




if __name__ == "__main__":


    G = nx.karate_club_graph()
    #G = nx.Graph()
    G.add_edges_from([(1, 2),(1, 3),(1, 4),(2, 4),(2, 5),(5, 6),(5, 7),(6, 7),(6, 8),(7, 8)])
    G = nx.convert_node_labels_to_integers(G,0)
    nodes_num = G.number_of_nodes()
    sim_matrix = init_similarity_matrix(G, nodes_num)
    res_tmp = sim_rank(G, nodes_num = nodes_num, sim_matrix=sim_matrix)
    tmp = np.zeros((nodes_num,nodes_num))
    res = lil_matrix(tmp,(nodes_num,nodes_num))
    for i,j in nx.complement(G).edges():
        res[i,j] = res_tmp[i,j]
    res = res.toarray()
    print(f"Il link più probabile è quello tra i nodi {np.where(res==res.max())} , con un valora di similarità di {res.max()}")
    

    #nx.draw(G, with_labels=True)
    #plt.show()