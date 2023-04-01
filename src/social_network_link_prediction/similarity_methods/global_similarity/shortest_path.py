import networkx as nx
from scipy.sparse import csr_matrix, lil_matrix

def shortest_path(G: nx.Graph, cutoff: int = None) -> csr_matrix:
    dim = G.number_of_nodes()
    if cutoff is None:
        cutoff = dim

    lengths = dict(nx.all_pairs_shortest_path_length(G, cutoff))
    
    S = lil_matrix((dim, dim))
    for source_node in lengths.keys():
        for dest_node in lengths[source_node].keys():
            S[source_node, dest_node] = -lengths[source_node][dest_node]

    #andrebbero normalizzati i valori per riportarli tra 0 e 1? o lasciamo i negativi?
    return S.tocsr()
