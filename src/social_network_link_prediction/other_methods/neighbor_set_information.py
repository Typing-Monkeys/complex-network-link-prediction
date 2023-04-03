import networkx as nx
import numpy as np
import scipy.sparse as scipy
import math
import itertools


'''
Il modello di link prediction basato su information theory che sfrutta la neighbor set information è un approccio utilizzato per prevedere la probabilità di esistenza di un link tra due nodi in una rete. In questo modello, l'informazione contenuta nei neighbor set dei due nodi in questione viene utilizzata per stimare la probabilità di connessione.

L'idea alla base di questo modello è che i nodi che hanno molti neighbor in comune sono più propensi a essere connessi tra loro rispetto a nodi con neighbor set diversi. Questo perché i nodi con neighbor set simili tendono a essere coinvolti in attività simili all'interno della rete, come ad esempio partecipare agli stessi gruppi o condividere gli stessi interessi.

Per utilizzare questa informazione per prevedere la probabilità di connessione tra due nodi, il modello utilizza l'entropia di Shannon, una misura dell'incertezza di una distribuzione di probabilità. In particolare, l'entropia viene calcolata sui neighbor set dei due nodi, e la differenza tra le entropie dei due set viene utilizzata per stimare la probabilità di connessione.

'''


# definition of the 2 indormation used in this information theory approach: overlapping nodes of different sets and
# the exsistence of link across different sets 
#
def overlap_info(G:nx.Graph, x, y, edge_num):
    # ottenimento dei dati da cui ottenere le informazioni
    o_nodes = nx.common_neighbors(G, x, y)
    p_prob_overlap = -np.log2(prior(x, y, G, edge_num))

    # utilizzo delle informazioni per stimarsi la likelihood
    # con gli overlapping nodes
    coeffZ = 0
    zOverlap = 0
    cnOverlap = 0
    for z in o_nodes:
        # degree of z
        kz = G.degree(z)
        coeffZ = 1 / (kz * (kz-1))   
        # sum over edges = neighbors of z
        for m, n in itertools.combinations(G.neighbors(z), 2):
            priorInfo = -np.log2(prior(m, n, G, edge_num))
            likelihoodInfo = -np.log2(likelihood(z, G))     
            # combine mutual information
            zOverlap += 2 * (priorInfo -likelihoodInfo)
    # add average mutual information per neighbor
    cnOverlap += coeffZ * zOverlap 
    s_Overlap = cnOverlap - p_prob_overlap
    return s_Overlap

     


# funzione che calcola la probabilità a priori dati due nodi e 
# un grafo riferita alla probabilità con cui non si forma un cammino
# tra i due nodi    
def prior(m, n, G, edge_num):
    kn = G.degree(n)
    km = G.degree(m)

    return 1 - math.comb(edge_num-kn, km)/math.comb(edge_num, km)

# probabilità condizionata che in questo caso è definita come il clustering
# coefficient dei common neighbor dei nodi x e y
def likelihood(z, G):
    kz = G.degree(z)
    N_triangles = nx.triangles(G, z)
    N_triads = math.comb(kz, 2)

    return N_triangles / N_triads



def MI(G : nx.Graph, lalla: int):
    I_Oxy = 0
    s_xy = []
    edge_num = G.number_of_edges()
    node_num = G.number_of_nodes()
    res = np.zeros((node_num, node_num))
    edge_num = G.number_of_edges()
    res_sparse = scipy.csr_matrix(res, shape=(node_num, node_num))

    for i,j in nx.complement(G).edges():
        I_Oxy = overlap_info(G, i, j, edge_num)
        res_sparse[i-1,j-1] = I_Oxy
    
    return res_sparse


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    G = nx.Graph()
    G.add_edges_from([(1, 2),(1, 3),(1, 4),(2, 4),(2, 5),(5, 6),(5, 7),(6, 7),(6, 8),(7, 8)])

    nx.draw(G, with_labels=True)

    ranking = MI(G, 0.1)
    # da aggiungere informazioni dei nodi che hanno fatto ottentere il 
    # ranking migliore
    print(ranking)
    '''
    max_ranking = max(ranking, key=lambda x: x['rank'])['rank']
    #G.add_edge(max_ranking)
    new_link = 0
    for i in ranking:
        if i["rank"] == max_ranking:
            new_link = i["nodes"]
    print(ranking)
    '''

    plt.show()
