import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, lil_matrix

def init_similarity_matrix(G:nx.Graph, n):
    #inizializzo la matrice similarity
    x = np.identity(n)
    sim_matrix = lil_matrix(x, (n, n))
    return sim_matrix


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



def sim_rank(G:nx.Graph, nodes_num = 0, sim_matrix = None):
    for a in range(nodes_num):
        for b in range(nodes_num):
            if(a == b):
                sim_matrix[a, b] = 0
            else:
                sim_matrix[a, b] = compute_sim_rank(G, a, b) 
    return sim_matrix




if __name__ == "__main__":

    G = nx.Graph()
    G.add_edges_from([(1, 2),(1, 3),(1, 4),(2, 4),(2, 5),(5, 6),(5, 7),(6, 7),(6, 8),(7, 8)])
    G = nx.convert_node_labels_to_integers(G,0)
    nodes_num = G.number_of_nodes()
    sim_matrix = init_similarity_matrix(G, nodes_num)
    res_tmp = sim_rank(G, nodes_num = nodes_num, sim_matrix=sim_matrix)
    tmp = np.zeros((nodes_num,nodes_num))
    res = lil_matrix(tmp,(nodes_num,nodes_num))
    for i,j in nx.complement(G).edges():
        res[i,j] = res_tmp[i,j]
        if(res[i,j] == res_tmp.toarray().max()):
            print(f"Il link più probabile è quello tra i nodi {i} e {j}, con un valora di similarità di {res_tmp.toarray().max()}")
        
    #nx.draw(G, with_labels=True)
    #plt.show()