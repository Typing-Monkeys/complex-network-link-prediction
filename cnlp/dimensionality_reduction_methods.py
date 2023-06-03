"""Dimensionality Reduction based Methods for Link Prediction.

The curse of dimensionality is a well-known
problem in machine learning.
Some researchers employ dimension reduction techniques
to tackle the above problem and apply it in the link prediction scenario.

References
----------
- [Link prediction on evolving data using matrix and tensor
factorizations](https://doi.org/10.1109/ICDMW.2009.54)
- [Nonnegative matrix factorization algorithms for link prediction in temporal
networks using graph
communicability](https://doi.org/10.1016/j.patcog.2017.06.025)
- [Temporal-relational classifiers for prediction in
evolving domains](https://doi.org/10.1109/ICDM.2008.125)
- [Link prediction via matrix
factorization](http://dl.acm.org/citation.cfm?id=2034117.2034146)
- [Link prediction based on non-negative matrix
factorization](https://doi.org/10.1371/journal.pone.0182968)
- [A perturbation-based framework for
link prediction via non-negative matrix
factorization](https://doi.org/10.1038/srep38938)
- [Link prediction in dynamic networks based on non-negative matrix
factorization](https://doi.org/10.26599/BDMA.2017.9020002)
- [Link prediction algorithm by
matrix factorization based on importance of
edges](https://doi.org/10.16451/j.cnki.issn1003-6059.201802006)
"""
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, spdiags
from cnlp.utils import to_adjacency_matrix, only_unconnected
from scipy.sparse.linalg import svds


def link_prediction_svd(G: nx.Graph,
                        k: int = 5,
                        normalize: bool = False) -> csr_matrix:
    """Compute the SVD Decomposition for the Graph Adjacency Matrix.
    The similarity decomposition is defined as:

    .. math::
        X_\\pm \\approx F G^T

    where \\(F \\in \\mathbb{R}^{p \\times k}\\) contains
    the bases of the latent space and is called the basis matrix;
    \\(G \\in \\mathbb{R}^{n \\times k}\\) contains combination of coefficients
    of the bases for reconstructing the matrix \\(X\\), and is called
    the coefficient matrix; \\(k\\) is the dimension of the latent space
    (\\(k<n\\)) and \\(n\\) is the number of data vector
    (as columns) in \\(X\\).

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)
    k: int :
        dimension of the latent space (must be \\(< n\\))
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

    References
    ----------
    [Link prediction using matrix factorization with
    bagging](https://doi.org/10.1109/ICIS.2016.7550942)
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


def link_prediction_nmf(graph: nx.Graph,
                        num_features: int = 2,
                        num_iterations: int = 100,
                        seed: int = 69) -> csr_matrix:
    """Compute the _Non-negative Matrix Factorization_ Decomposition
    for the Graph Adjacency Matrix.
    The similarity decomposition is defined as:

    .. math::
        X_\\pm \\approx F_+ G^T_+

    where \\(F \\in \\mathbb{R}^{p \\times k}\\) contains
    the bases of the latent space and is called the basis matrix;
    \\(G \\in \\mathbb{R}^{n \\times k}\\) contains combination of coefficients
    of the bases for reconstructing the matrix \\(X\\), and is called
    the coefficient matrix; \\(k\\) is the dimension of the latent space
    ( \\(k<n\\) ) and \\(n\\) is the number of data vector
    (as columns) in \\(X\\).

    Parameters
    ----------
    graph: nx.Graph :
        input Graph (a networkx Graph)
    num_features: int :
        dimension of the latent space (must be \\(< n\\))
         (Default value = 2)
    num_iterations: int :
        max number of iteration for the algorithm convergence
         (Default value = 100)
    seed: int :
        the seed for the random initialization
         (Default value = 69)

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

    References
    ----------
    [Convex and semi-nonnegative matrix
    factorizations](https://doi.org/10.1109/TPAMI.2008.277)
    """

    adj_matrix = to_adjacency_matrix(graph, sparse=False)

    # Initialize the feature and coefficient matrices randomly
    num_nodes = adj_matrix.shape[0]

    np.random.seed(seed)

    feature_matrix = np.random.rand(num_nodes, num_features)

    coefficient_matrix = np.random.rand(num_features, num_nodes)

    # Perform NMF using multiplicative updates
    for i in range(num_iterations):
        # Update the coefficient matrix
        numerator = feature_matrix.T @ adj_matrix
        denominator = (
            (feature_matrix.T @ feature_matrix) @ coefficient_matrix)

        coefficient_matrix *= (numerator / denominator)

        numerator = adj_matrix @ coefficient_matrix.T
        denominator = (
            feature_matrix @ (coefficient_matrix @ coefficient_matrix.T))

        feature_matrix *= (numerator / denominator)

    # Compute the predicted adjacency matrix
    predicted_adj_matrix = feature_matrix @ coefficient_matrix

    return only_unconnected(graph, csr_matrix(predicted_adj_matrix))


if __name__ == "__main__":
    # Load a test graph
    graph = nx.karate_club_graph()

    # Compute the SVD of the similarity matrix and do link prediction on it
    predicted_adj_matrix = link_prediction_nmf(graph)

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
