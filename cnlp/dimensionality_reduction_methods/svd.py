import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, spdiags
from scipy.sparse.linalg import svds
from cnlp.utils import to_adjacency_matrix, only_unconnected


def link_prediction_svd(G: nx.Graph,
                        k: int = 5,
                        normalize: bool = False) -> csr_matrix:
    """Compute the SVD Decomposition for the Graph Adjacency Matrix.
    The similarity decinoisutuin is defined as:

    .. math::
        X_\\pm \\approx F G^T

    where \\(F \\in \\mathbb{R}^{p \\times k}\\) contains
    the bases of the latent space and is called the basis matrix;
    \\(G \\in \\mathbb{R}^{n \\times k}\\) contains combination of coefficients
    of the bases for reconstructing the matrix \\(X\\), and is called
    the coefficient matrix; \\(k\\) is the dimention of the latent space
    (\\(k<n\\)) and \\(n\\) is the nunber of data vector
    (as columns) in \\(X\\).

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)
    k: int :
        dimention of the latent space (must be \\(< n\\))
         (Default value = 5)
    normalize: bool :
        if True, normalize the output values
         (Default value = False)

    Returns
    -------
    predicted_adj_matrix: csr_matrix : the Similarity Matrix (in sparse format)

    Notes
    -----
    Typically, the latent features are extracted and using these features,
    each vertex is represented in latent space, and such representations are
    used in a supervised or unsupervised framework for link prediction.
    To further improve the prediction results, some additional node/link or
    other attribute information can be used.

    In most of the works, non-negative matrix factorization has been used.
    Some authors also applied the singular value decomposition technique.
    """

    # Create the adjacency matrix of the graph
    adj_matrix = to_adjacency_matrix(G)

    # Perform the singular value decomposition
    U, S, Vt = svds(csr_matrix(adj_matrix).astype(float), k=k)

    # Make the diagonal matrix sparse
    S = spdiags(S, [0], k, k)

    # Compute the predicted adj_matrix
    predicted_adj_matrix = ((U @ S) @ Vt)

    if normalize:
        # Normalize the predicted edge weights to the range [0, 1]
        rows, cols = np.nonzero(predicted_adj_matrix)
        max_weight = np.max(predicted_adj_matrix[rows, cols])
        for i in range(len(rows)):
            predicted_adj_matrix[rows[i], cols[i]] /= max_weight

    return only_unconnected(G, csr_matrix(predicted_adj_matrix))


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
        for v in range(u + 1, adj_matrix.shape[1]):
            if graph.has_edge(u, v):
                continue
            w = predicted_adj_matrix[u, v]
            predicted_edges.append((u, v, w))

    # Normalize the predicted edge weights to the range [0, 1]
    max_weight = max(predicted_edges, key=lambda x: x[2])[2]
    for i in range(len(predicted_edges)):
        predicted_edges[i] = (predicted_edges[i][0], predicted_edges[i][1],
                              predicted_edges[i][2] / max_weight)

    # Sort the predicted edges by weight in descending order
    predicted_edges.sort(key=lambda x: x[2], reverse=True)

    # Print the predicted edges with their weight score
    print("Top 10 Predicted edges:")
    for edge in predicted_edges[:10]:
        print(f"({edge[0]}, {edge[1]}): {round(edge[2], 3)}")

    plt.show()
