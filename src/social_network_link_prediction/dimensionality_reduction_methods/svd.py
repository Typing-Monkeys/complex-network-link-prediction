import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

def link_prediction_svd(G: nx.Graph, k = 5) -> csr_matrix:

    # Create the adjacency matrix of the graph
    adj_matrix = nx.adjacency_matrix(G)
    
    # Perform the singular value decomposition
    U, S, Vt = svds(csr_matrix(adj_matrix).astype(float), k=k)
    S = np.diag(S)
    
    # Compute the predicted adj_matrix
    predicted_adj_matrix = U.dot(S).dot(Vt)
    
    return predicted_adj_matrix

if __name__ == "__main__":
    # Load a test graph
    graph = nx.karate_club_graph()

    # Draw the test graph
    nx.draw(graph, with_labels=True)

    # Compute the SVD of the similarity matrix and do link prediction on it
    predicted_adj_matrix = link_prediction_svd(graph)

    # adj_matrix of the starting graph
    adj_matrix = nx.adjacency_matrix(graph)

    # Get the predicted eges
    predicted_edges = []
    for u in range(adj_matrix.shape[0]):
        for v in range(u+1, adj_matrix.shape[1]):
            if graph.has_edge(u, v):
                continue
            w = predicted_adj_matrix[u, v]
            predicted_edges.append((u, v, w))
            
    # Normalize the predicted edge weights to the range [0, 1]
    max_weight = max(predicted_edges, key=lambda x: x[2])[2]
    for i in range(len(predicted_edges)):
        predicted_edges[i] = (predicted_edges[i][0], predicted_edges[i][1], predicted_edges[i][2]/max_weight)
    
    # Sort the predicted edges by weight in descending order
    predicted_edges.sort(key=lambda x: x[2], reverse=True)
    
    # Print the predicted edges with their probability score
    print("Top 10 Predicted edges:")
    for edge in predicted_edges[:10]:
        print("({0}, {1}): {2}".format(edge[0], edge[1], round(edge[2], 3)))

    plt.show()