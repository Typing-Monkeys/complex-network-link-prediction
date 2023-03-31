import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF

def link_prediction_nmf(G, num_features=2, num_iterations=100, learning_rate=0.01):
    
    # Get the adjacency matrix of the graph
    adj_matrix = nx.adjacency_matrix(G)
    #adj_matrix = nx.to_numpy_array(G)


    # Initialize the feature and coefficient matrices randomly
    num_nodes = adj_matrix.shape[0]

    #TODO: Farlo funzionare con le matrici sparse.
    # Se si usano le matrici sparse al posto di queste di numpy non funziona 
    feature_matrix = np.random.rand(num_nodes, num_features)
    #feature_matrix = sp.rand(num_nodes, num_features)
    coefficient_matrix = np.random.rand(num_features, num_nodes)
    #coefficient_matrix = sp.rand(num_features, num_nodes)


    # Perform NMF using multiplicative updates
    for i in range(num_iterations):
        # Update the coefficient matrix
        #numerator = np.dot(feature_matrix.T, adj_matrix)
        numerator = feature_matrix.T @ adj_matrix
        #denominator = np.dot(np.dot(feature_matrix.T, feature_matrix), coefficient_matrix)
        denominator = ((feature_matrix.T @ feature_matrix) @ coefficient_matrix)
        print(numerator.shape)
        print(denominator.shape)
        print(coefficient_matrix.shape)
        print((numerator / denominator).shape)
        coefficient_matrix *= (numerator / denominator)
        #coefficient_matrix = csr_matrix(coefficient_matrix) @ (numerator / denominator)

        # Update the feature matrix
        #numerator = np.dot(adj_matrix, coefficient_matrix.T)
        #denominator = np.dot(feature_matrix, np.dot(coefficient_matrix, coefficient_matrix.T))
        numerator = adj_matrix @ coefficient_matrix.T
        denominator = (feature_matrix @ (coefficient_matrix @ coefficient_matrix.T))
        print(numerator.shape)
        print(denominator.shape)
        print(feature_matrix.shape)
        feature_matrix *= (numerator / denominator)
        #feature_matrix = csr_matrix(feature_matrix) @ (numerator / denominator)
        print("-------------------")

    # Compute the predicted adjacency matrix
    #predicted_adj_matrix = np.dot(feature_matrix, coefficient_matrix)
    predicted_adj_matrix = feature_matrix @ coefficient_matrix

    # Replace diagonal elements with zeros (no self-loops)
    #np.fill_diagonal(predicted_adj_matrix, 0)

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


def link_prediction_nmf2(graph):
    # Get the adjacency matrix of the graph
    adj_matrix = nx.to_numpy_array(graph)
    #adj_matrix = nx.adjacency_matrix(graph)

    num_nodes = adj_matrix.shape[0]

    # Perform NMF decomposition on the adjacency matrix
    model = NMF(n_components=2, init='random', random_state=0)
    W = model.fit_transform(adj_matrix)
    H = model.components_

    # Compute the predicted adjacency matrix
    predicted_adj_matrix = np.dot(W, H)

    # Replace diagonal elements with zeros (no self-loops)
    #np.fill_diagonal(predicted_adj_matrix, 0)

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
    print("---------------------------")
    link_prediction_nmf2(graph)

    #plt.show()