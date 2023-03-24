import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def link_prediction_nmf(G, num_features=2, num_iterations=100, learning_rate=0.01):
    # Get the adjacency matrix of the graph
    adj_matrix = nx.to_numpy_array(G)

    # Initialize the feature and coefficient matrices randomly
    num_nodes = adj_matrix.shape[0]
    feature_matrix = np.random.rand(num_nodes, num_features)
    coefficient_matrix = np.random.rand(num_features, num_nodes)

    # Perform NMF using multiplicative updates
    for i in range(num_iterations):
        # Update the coefficient matrix
        numerator = np.dot(feature_matrix.T, adj_matrix)
        denominator = np.dot(np.dot(feature_matrix.T, feature_matrix), coefficient_matrix)
        coefficient_matrix *= (numerator / denominator)

        # Update the feature matrix
        numerator = np.dot(adj_matrix, coefficient_matrix.T)
        denominator = np.dot(feature_matrix, np.dot(coefficient_matrix, coefficient_matrix.T))
        feature_matrix *= (numerator / denominator)

    # Compute the predicted adjacency matrix
    predicted_adj_matrix = np.dot(feature_matrix, coefficient_matrix)

    # Replace diagonal elements with zeros (no self-loops)
    np.fill_diagonal(predicted_adj_matrix, 0)

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

if __name__ == "__main__":
    # Load a test graph
    graph = nx.karate_club_graph()

    # Draw the test graph
    nx.draw(graph, with_labels=True)

    # Compute the SVD of the similarity matrix and do link prediction on it
    link_prediction_nmf(graph)

    plt.show()