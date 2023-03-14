import networkx as nx

def to_adjacency_matrix(G: nx.Graph, sparse=True):
    """
        Dato un grafo ritorna la relativa Matrice di Adiacenza
        
        ARGS
            G: grafo in formato Networkx
            sparse: se True, ritorna una matrice sparsa, altrimenti
                    un numpy array

        RET
            ritorna la matrice di adiacenza (sparse o numpy array)
    """
    
    return nx.adjacency_matrix(G) if sparse else nx.to_numpy_array(G)