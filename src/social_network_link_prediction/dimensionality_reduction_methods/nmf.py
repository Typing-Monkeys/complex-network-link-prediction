import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix


def link_prediction_nmf(graph, num_features=2, num_iterations=100):
    
    adj_matrix = nx.to_numpy_array(graph)

    # Initialize the feature and coefficient matrices randomly
    num_nodes = adj_matrix.shape[0]

    feature_matrix = np.random.rand(num_nodes, num_features)

    coefficient_matrix = np.random.rand(num_features, num_nodes)


    # Perform NMF using multiplicative updates
    for i in range(num_iterations):
        # Update the coefficient matrix
        numerator = feature_matrix.T @ adj_matrix
        denominator = ((feature_matrix.T @ feature_matrix) @ coefficient_matrix)

        coefficient_matrix *= (numerator / denominator)

        numerator = adj_matrix @ coefficient_matrix.T
        denominator = (feature_matrix @ (coefficient_matrix @ coefficient_matrix.T))

        feature_matrix *= (numerator / denominator)

    # Compute the predicted adjacency matrix
    predicted_adj_matrix = feature_matrix @ coefficient_matrix

    return csr_matrix(predicted_adj_matrix)


if __name__ == "__main__":
    # Load a test graph
    graph = nx.karate_club_graph()

    # Compute the SVD of the similarity matrix and do link prediction on it
    predicted_adj_matrix =  link_prediction_nmf(graph)
    
    adj_matrix = nx.to_numpy_array(graph)

    # Initialize the feature and coefficient matrices randomly
    num_nodes = adj_matrix.shape[0]
    # Get the edges with the highest predicted probabilities
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            prob = predicted_adj_matrix[i, j]
            edges.append((i, j, prob))

    edges = sorted(edges, key=lambda x: x[2], reverse=True)[:10]

    # Normalize the probabilities to be between 0 and 100
    max_prob = max([e[2] for e in edges])
    for edge in edges:
        edge_prob = (edge[2] / max_prob) * 100
        print(f"({edge[0]}, {edge[1]}) - Probability: {edge_prob:.2f}%")