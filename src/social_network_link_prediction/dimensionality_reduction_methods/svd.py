import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def singular_value_decomposition(G, k = 5):
    # k is the number of singular vectors to use for link prediction
    # Compute the similarity matrix from the graph
    A = nx.to_numpy_array(G)
    S = np.dot(A, A.T)

    # Perform singular value decomposition on the similarity matrix
    U, sigma, VT = np.linalg.svd(S, full_matrices=False)

    # Compute the reduced SVD
    Uk = U[:, :k]
    Vk = VT[:k, :]
    Sk = np.diag(sigma[:k])

    # Compute the predicted similarity matrix
    S_pred = np.dot(Uk, np.dot(Sk, Vk))

    # Get the edges that are missing from the graph
    missing_edges = []
    for u, v in nx.non_edges(G):
        missing_edges.append((u, v))

    # Compute the predicted scores for the missing edges
    scores = {}
    for u, v in missing_edges:
        score = S_pred[u, v]
        scores[(u, v)] = score

    # Sort the predicted scores in descending order
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Print the top 10 predicted edges
    print("Top 10 predicted edges:")
    for edge, score in sorted_scores[:10]:
        print(edge, score)

if __name__ == "__main__":
    # Load a test graph
    graph = nx.karate_club_graph()

    # Draw the test graph
    nx.draw(graph, with_labels=True)

    # Compute the SVD of the similarity matrix and do link prediction on it
    singular_value_decomposition(graph)

    plt.show()