import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix, lil_matrix, linalg
from functools import partial
from social_network_link_prediction.utils import to_adjacency_matrix
from social_network_link_prediction.utils import nodes_to_indexes

def link_prediction_rwr(G, c = 0.05, max_iters = 10):
    
    A = to_adjacency_matrix(G)
    m,n = A.shape

    D = lil_matrix(A.shape)

    nodes_to_indexes_map = nodes_to_indexes(G)
    for node in G.nodes():
        D[nodes_to_indexes_map[node], nodes_to_indexes_map[node]] = G.degree[node]
    
    D = D.tocsc()
    W_normalized = linalg.inv(D) @ A.tocsc()

    random_walk_with_restart_fn = partial(random_walk_with_restart, W_normalized = W_normalized, c = c, max_iters= max_iters)
    
    #Run the function starting from each node of the graph
    similarity = [random_walk_with_restart_fn(e) for e in np.identity(m)]
    similarity_matrix = csr_matrix(similarity)
    return similarity_matrix

def random_walk_with_restart(e, W_normalized, c = 0.2, max_iters = 100):    
    old_e = e
    err = 1.

    for i in range(max_iters):
        e = (c * (W_normalized @ old_e)) + ((1 - c) * e)
        err = np.linalg.norm(e - old_e, 1)
        if err <= 1e-6:
            break
        old_e = e
    return e
  

if __name__ == '__main__':

    graph = nx.karate_club_graph()
    
    # apply random walk with restart on this network
    predicted_adj_matrix = link_prediction_rwr(graph, c = 0.05, max_iters=10)
    #predicted_adj_matrix = run_rwr(graph, R = 0.2, max_iters=1000)

    adj_matrix = nx.to_numpy_array(graph)

    # Initialize the feature and coefficient matrices randomly
    num_nodes = adj_matrix.shape[0]
    # Get the predicted edges
    predicted_edges = []
    for u in range(adj_matrix.shape[0]):
        for v in range(u + 1, adj_matrix.shape[1]):
            if graph.has_edge(u, v):
                continue
            w = predicted_adj_matrix[u, v]
            predicted_edges.append((u, v, w))

    # Sort the predicted edges by weight in descending order
    predicted_edges.sort(key=lambda x: x[2], reverse=True)

    # Print the predicted edges with their weight score
    print("Top Predicted edges:")
    for edge in predicted_edges[:10]:
        print(f"({edge[0]}, {edge[1]}): {edge[2]}")