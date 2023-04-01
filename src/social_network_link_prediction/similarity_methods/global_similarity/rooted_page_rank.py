import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, linalg, identity
from social_network_link_prediction.utils import to_adjacency_matrix

# il nome va bene così???
# bisogna rimappare i nomi dei nodi?
def RPR(G: nx.Graph, alpha: np.float32 = .5) -> csr_matrix:
    A = to_adjacency_matrix(G)
    D = lil_matrix(A.shape)#diagonal matrix with degree
    for node in G.nodes():
        D[node, node] = G.degree[node]
    
    D = D.tocsc()
    N_hat = linalg.inv(D) @ A.tocsc()
    eye = identity(A.shape[0], format='csc')
    S = (1 - alpha) * linalg.inv(eye - alpha*N_hat) #formula corretta?

    return S.tocsr()
