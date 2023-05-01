"""Collection of Global Similarity Methods for Link Prediction.

Global indices are computed using entire
topological information of a network.
The computational complexities of such methods
are higher and seem to be infeasible for large networks.
"""
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix, linalg, identity, lil_matrix, hstack, lil_array
from cnlp.utils import nodes_to_indexes, only_unconnected, to_adjacency_matrix


def katz_index(G: nx.Graph, beta: int = 1) -> csr_matrix:
    """Compute the Katz Index for all nodes in the Graph.
    Each similarity value is defined as:

    .. math::
        S(x, y) = (I - \\beta A)^{-1} - I

    where \\(I\\) is the Identity Matrix,
    \\(\\beta\\) is a dumping factor that controls the path weights
    and \\(A\\) is the Adjacency Matrix

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)
    beta: int :
        Dumping Factor
         (Default value = 1)

    Returns
    -------
    S: csr_matrix : the Similarity Matrix (in sparse format)

    Notes
    -----
    For the convergence of above equation,

    .. math::
        \\beta < \\frac{1}{\\lambda_1}

    where \\(\\lambda_1\\) is the maximum eighenvalue of the matrix \\(A\\).

    The computational complexity of the given metric is high,
    and it can be roughly estimated to be cubic complexity
    which is not feasible for a large network.
    """

    def __power_method(A: csr_matrix,
                       max_iterations: int = 100,
                       tol: float = 1e-12,
                       verbose: bool = False):
        """Perfome the Power Method"""
        n = A.shape[0]
        x = np.ones(n) / np.sqrt(n)  # initialize a vector x
        # r = A @ x - np.dot(A @ x, x) * x # residual initialization
        r = A @ x - ((A @ x) @ x) * x
        eigenvalue = x @ (A @ x)  # residual eigenvalue
        # eigenvalue = np.dot(x, A @ x) # residual eigenvalue

        for i in range(max_iterations):
            # Compute the new vector x
            x = A @ x
            # vector normalization
            x = x / np.linalg.norm(x)

            # Residual and eigenvalue computation
            r = A @ x - ((A @ x) @ x) * x
            eigenvalue = x @ (A @ x)
            # r = A @ x - np.dot(A @ x, x) * x
            # eigenvalue = np.dot(x, A @ x)

            # If the norm of r is less than the tolerance, break out of the loop.
            if np.linalg.norm(r) < tol:
                if verbose:
                    print(f'Computation done after {i} steps')
                break

        return eigenvalue, x

    A = to_adjacency_matrix(G)
    largest_eigenvalue = __power_method(A)  # lambda_1
    if beta >= (1 / largest_eigenvalue[0]):
        print(f'Warning, Beta should be less than {largest_eigenvalue}')

    eye = identity(A.shape[0], format='csc')
    S = linalg.inv((eye - beta * A.tocsc())) - eye

    return only_unconnected(G, csr_matrix(S))


def link_prediction_rwr(G: nx.Graph,
                        c: int = 0.05,
                        max_iters: int = 10) -> csr_matrix:
    """Compute the Random Walk with Restart Algorithm.

    The similarity between two nodes is defined as:

    .. math::
        S(x, y) = q_{xy} + q_{yx}

    where \\(q_x\\) is defined as \\( (1-\\alpha) (I - \\alpha P^T)^{-1} e_x\\)
    and \\(e_x\\) is the seed vector of length \\(|V|\\).

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)
    c: int :
        TODO
         (Default value = 0.05)
    max_iters: int :
        max number of iteration for the algorithm convergence
         (Default value = 10)

    Returns
    -------
    similarity_matrix: csr_matrix : the Similarity Matrix (in sparse format)

    Notes
    -----
    Let \\(\\alpha\\) be a probability that a random walker
    iteratively moves to an arbitrary neighbor and returns to the same
    starting vertex with probability \\( (1 - \\alpha )\\).
    Consider \\(q_{xy}\\) to be the probability that a random walker
    who starts walking from vertex x and located at the vertex y in steady-state.

    The seed vector \\(e_x\\) consists of zeros for all components except the
    elements \\(x\\) itself.

    The transition matrix \\(P\\) can be expressed as

    .. math::
        P_{xy} = \\begin{cases}
                \\frac{1}{k_x} & \\text{if } x \\text{ and } y \\text{ are connected,} \\\\
                0 & \\text{otherwise.}
            \\end{cases}
    """

    def random_walk_with_restart(e: lil_array,
                                 W_normalized: csr_matrix,
                                 c: int = 0.05,
                                 max_iters: int = 100) -> lil_array:
        """Generates the probability vector

        Parameters
        ----------
        e: lil_array :
            input probability vector
        W_normalized: csr_matrix :
            TODO
        c: int :
            TODO
             (Default value = 0.05)
        max_iters: int :
            max number of iteration for the algorithm convergence
             (Default value = 100)

        Returns
        -------
        e: lil_array : the updated probability vector
        """
        # Initialize the current probability vector to the initial one and the error to 1
        old_e = e
        err = 1.

        # Perform the random walk with restart until the maximum number
        # of iterations is reached or the error becomes less than 1e-6
        for _ in range(max_iters):
            e = (c * (W_normalized @ old_e)) + ((1 - c) * e)
            err = linalg.norm(e - old_e, 1)
            if err <= 1e-6:
                break
            old_e = e

        # Return the current probability vector
        return e

    # Convert the graph G into an adjacency matrix A
    A = to_adjacency_matrix(G)

    # Extract the number of nodes of matrix A
    m = A.shape[0]

    # Initialize the diagonal matrix D as a sparse lil_matrix
    D = lil_matrix(A.shape)

    # Create a map that associates each node with a row index in matrix A
    nodes_to_indexes_map = nodes_to_indexes(G)

    # Build the diagonal matrix D so that the elements on the diagonal
    # are equal to the degree of the corresponding node
    for node in G.nodes():
        D[nodes_to_indexes_map[node],
          nodes_to_indexes_map[node]] = G.degree[node]

    # Convert the diagonal matrix D into csc_matrix format
    D = D.tocsc()

    try:
        # Build the normalized transition matrix W_normalized
        W_normalized = linalg.inv(D) @ A.tocsc()
    except RuntimeError as e:
        print('Possible presence of singleton nodes in the graph G')
        print(e)
        exit(1)

    # Initialize an matrix to hold the similarities between node pairs
    # We put an initial column made of Zeros so we can use the hstack
    # method later on and keep the code more clean
    similarity_matrix = csr_matrix((m, 1))

    # For each node i, create a probability vector and perform the
    # random walk with restart starting from that node
    for i in range(m):
        e = lil_array((m, 1))
        e[i, 0] = 1
        # Concatenate the similarity vectors into a similarity matrix
        # The use of hstack allows the lil_array returned from the
        # random walk function to be transposed and added to the
        # similarity matrix as a new column in just one line of code
        similarity_matrix = hstack([
            similarity_matrix,
            random_walk_with_restart(e=e,
                                     W_normalized=W_normalized,
                                     c=c,
                                     max_iters=max_iters)
        ])

    # Return the similarity matrix and remove the fisrt column
    # In order to keep the results consistent without the added column of zeros at the beginning
    return only_unconnected(G, csr_matrix(similarity_matrix)[:, 1:])


def rooted_page_rank(G: nx.Graph, alpha: float = .5) -> csr_matrix:
    """Compute the Rooted Page Rank for all nodes in the Graph.
    This score is defined as:

    .. math::
        S = (1 - \\alpha) (I - \\alpha \\hat{N})^{-1}

    where \\(\\hat{N} = D^{-1}A\\) is the normalized
    Adjacency Matrix with the diagonal degree matrix
    \\(D[i,i] = \\sum_j A[i,j]\\)

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)
    alpha: float :
        random walk probability
         (Default value = 0.5)

    Returns
    -------
    S: csr_matrix : the Similarity Matrix (in sparse format)

    Notes
    -----
    The idea of PageRank was originally proposed to rank the web pages based on
    the importance of those pages. The algorithm is based on the assumption that
    a random walker randomly goes to a web page with probability \\(\\alpha\\)
    and follows hyper-link embedded in the page with probability \\( (1 - \\alpha ) \\).
    Chung et al. used this concept incorporated with a random walk in
    link prediction framework. The importance of web pages, in a random walk,
    can be replaced by stationary distribution. The similarity between two vertices
    \\(x\\) and \\(y\\) can be measured by the stationary probability of
    \\(x\\) from \\(y\\) in a random walk where the walker moves to an
    arbitrary neighboring vertex with probability \\(\\alpha\\)
    and returns to \\(x\\) with probability \\( ( 1 - \\alpha )\\).
    """
    A = to_adjacency_matrix(G)
    D = lil_matrix(A.shape)

    nodes_to_indexes_map = nodes_to_indexes(G)
    for node in G.nodes():
        D[nodes_to_indexes_map[node],
          nodes_to_indexes_map[node]] = G.degree[node]

    D = D.tocsc()
    N_hat = linalg.inv(D) @ A.tocsc()
    eye = identity(A.shape[0], format='csc')
    S = (1 - alpha) * linalg.inv(eye - alpha * N_hat)

    return only_unconnected(G, S.tocsr())


def shortest_path(G: nx.Graph, cutoff: int = None) -> csr_matrix:
    """Compute the Shortest Path Index for all nodes in the Graph.
    Each similarity value is defined as:

    .. math::
        S(x, y) = - |d(x,y)|

    where Dijkstra algorithm  is applied to efficiently
    compute the shortest path \\(d(x, y)\\) between the
    node pair \\( (x, y) \\).

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)
    cutoff: int :
        max path length
         (Default value = None)

    Returns
    -------
    S: csr_matrix : the Similarity Matrix (in sparse format)

    Notes
    -----
    Liben-Nowell et al. provided the shortest path with its negation as a
    metric to link prediction.

    The prediction accuracy
    of this index is low compared to most local indices.
    """
    dim = G.number_of_nodes()
    if cutoff is None:
        cutoff = dim

    lengths = dict(nx.all_pairs_shortest_path_length(G, cutoff))
    nodes_to_indexes_map = nodes_to_indexes(G)
    prexisting_links = list(G.edges())

    S = lil_matrix((dim, dim))
    for source_node in lengths.keys():
        for dest_node in lengths[source_node].keys():
            # If the link already exists in the starting graph the computation is skipped
            if (nodes_to_indexes_map[source_node],
                    nodes_to_indexes_map[dest_node]) not in prexisting_links:
                S[nodes_to_indexes_map[source_node],
                  nodes_to_indexes_map[dest_node]] = -lengths[source_node][
                      dest_node]

    return S.tocsr()


def sim_rank(G: nx.Graph,
             k: int = 5,
             cutoff: int = 4,
             c: int = 0.8) -> csr_matrix:
    """Compute the SimRank index for all the nodes in the Graph.

    This method is defined as:

    .. math::
        S(x, y) = \\begin{cases}
                \\frac{\\alpha}{k_x k_y} \\sum_{i=1}^{k_x} \\sum_{j=1}^{k_y}
                    S( \\Gamma_i(x), \\Gamma_j(y)) & x \\neq y \\\\
                1 & x = y
            \\end{cases}

    where \\( \\alpha \\in (0,1) \\) is a constant. \\(\\Gamma_i(x)\\) and \\( \\Gamma_j(y) \\)
    are the ith and jth elements in the neighborhood sets.

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)
    k: int :
         (Default value = 5)
    cutoff: int :
         (Default value = 4)
    c: int :
         (Default value = 0.8)

    Returns
    -------
    sim_matrix: csr_matrix : the Similarity Matrix (in sparse format)
    """

    def init_similarity_matrix(G: nx.Graph, n: int) -> lil_matrix:
        """Generate an Identity matrix: the starting Similarity
        Matrix.

        Parameters
        ----------
        G: nx.Graph :
            input Graph (a networkx Graph)
        n: int :
           the new matrix size

        Returns
        -------
        sim_matrix: lil_matrix : the starting Similarity Matrix
        """
        # inizializzo la matrice similarity
        # gli elementi con loro stessi (lungo la diagonale) hanno similarità massima
        sim_matrix = identity(n).tolil()
        return sim_matrix

    def compute_sim_rank(G: nx.Graph,
                         a,
                         b,
                         sim_matrix: lil_matrix,
                         C: int = 0.8) -> float:
        """Compute the Sim Rank method between the given
        nodes a and b.

        Parameters
        ----------
        G: nx.Graph :
            input Graph (a networkx Graph)
        a :
           first node
        b :
           second node
        sim_matrix: lil_matrix :
            the similarity matrix
        C: int :
            free parameter
             (Default value = 0.8)

        Returns
        -------
        new_SimRank: float : the SimRank value between a and b
        """

        # se i nodi sono uguali allora similarità massima
        if (a == b):
            return 1

        a_neigh = list(G.neighbors(a))
        b_neigh = list(G.neighbors(b))
        len_a = len(a_neigh)
        len_b = len(b_neigh)

        # nodi isolati hanno similarità 0
        if (len_a == 0 or len_b == 0):
            return 0

        # mi recupero e sommo i valori di similarità calcolati in precedenza
        simRank_sum = 0
        for i in a_neigh:
            for j in b_neigh:
                simRank_sum += sim_matrix[i, j]
        # moltiplico secondo la definizione del paper
        scale = C / (len_a * len_b)
        new_SimRank = scale * simRank_sum
        return new_SimRank

    G = nx.convert_node_labels_to_integers(G, 0)

    nodes_num = G.number_of_nodes()
    sim_matrix = init_similarity_matrix(G, nodes_num)

    for a in range(nodes_num):
        for b in range(nodes_num):
            # fa pruning evitando di calcolare la similarità di archi a distanza maggiore di 5
            if (nx.has_path(G, a, b)
                    and (nx.shortest_path_length(G, a, b) > cutoff)):
                sim_matrix[a, b] = 0
            else:
                # se non deve fare pruning si calcola il valore di similarità per i nodi a e b
                for i in range(k):
                    sim_matrix[a, b] = compute_sim_rank(G,
                                                        a,
                                                        b,
                                                        sim_matrix=sim_matrix,
                                                        C=c)

    # imposta a 0 gli elementi della diagonale che prima avevano similarità uguale ad 1
    for a in range(nodes_num):
        sim_matrix[a, a] = 0
    return only_unconnected(G, sim_matrix)
