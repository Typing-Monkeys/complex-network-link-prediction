"""Collection of Local Similarity Methods for Link Prediction.

Local indices are generally calculated using information about
common neighbors and node degree.
These indices **consider immediate neighbors of a node**.
"""
import networkx as nx
import numpy as np
from cnlp.utils import nodes_to_indexes
from scipy.sparse import lil_matrix, csr_matrix


def adamic_adar(G: nx.Graph) -> csr_matrix:
    """Compute the Adamic and Adar Index for all nodes in the Graph.
    Each similarity value is defined as:

    .. math::
        S(x, y) = \\sum_{z \\in \\Gamma(x) \\cap \\Gamma(y)}
        \\frac{1}{\\log k_z}

    where \\(k_z\\) is the degree of node \\(z\\)
    and \\(\\Gamma(x)\\) are the neighbors of the node \\(x\\).

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)

    Returns
    -------
    S: csr_matrix : the Similarity Matrix (in sparse format)

    Notes
    -----
    It is clear from the equation that more weights are assigned to
    the common neighbors having smaller degrees.
    This is also intuitive in the real-world scenario, for example,
    a person with more number of friends spend less time/resource
    with an individual friend as compared to the less number of friends.
    """

    def __adamic_adar(G: nx.Graph, x, y) -> float:
        """Compute the Adamic and Adar Index for 2 given nodes."""
        return sum([1 / np.log(G.degree[z]) for z in set(G[x]) & set(G[y])])

    size = G.number_of_nodes()
    S = lil_matrix((size, size))
    node_index_map = nodes_to_indexes(G)

    for x, y in nx.complement(G).edges():
        _x = node_index_map[x]
        _y = node_index_map[y]

        S[_x, _y] = __adamic_adar(G, x, y)

    return S.tocsr()


def __common_neighbors(G: nx.Graph, x, y) -> int:
    """Compute the Common Neighbors Index for 2 given nodes."""
    return len(set(G[x]).intersection(set(G[y])))


def common_neighbors(G: nx.Graph) -> csr_matrix:
    """Compute the Common Neighbors Index for all nodes in the Graph.
    Each similarity value is defined as:

    .. math::
        S(x, y) = |\\Gamma(x) \\cap \\Gamma(y)|

    where \\(\\Gamma(x)\\) are the neighbors of the node \\(x\\).

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)

    Returns
    -------
    S: csr_matrix : the Similarity Matrix (in sparse format)

    Notes
    -----
    The likelihood of the existence of a link between \\(x\\)
    and \\(y\\) increases with the number of common neighbors between them.
    """
    size = G.number_of_nodes()
    S = lil_matrix((size, size))
    node_index_map = nodes_to_indexes(G)

    for x, y in nx.complement(G).edges():
        _x = node_index_map[x]
        _y = node_index_map[y]

        S[_x, _y] = __common_neighbors(G, x, y)
        # S[_y, _x] = S[_x, _y]

    return S.tocsr()


def cosine_similarity(G: nx.Graph) -> csr_matrix:
    """Compute the Cosine Similarity Index
    (a.k.a. Salton Index) for all nodes in the Graph.
    Each similarity value is defined as:

    .. math::
        S(x, y) = \\frac{|\\Gamma(x) \\cap \\Gamma(y)|}{\\sqrt{k_x k_y}}

    where \\(\\Gamma(x)\\) are the neighbors of node \\(x\\)
    and \\(k_x\\) is the degree of the node \\(x\\).

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)

    Returns
    -------
    S: csr_matrix : the Similarity Matrix (in sparse format)

    Notes
    -----
    This similarity index between two nodes is measured by
    calculating the Cosine of the angle between them.
    The metric is all about the orientation and not magnitude.
    """

    def __cosine_similarity(G: nx.Graph, x, y) -> float:
        """Compute the Cosine Similarity Index for 2 given nodes."""
        return __common_neighbors(G, x, y) / np.sqrt(G.degree[x] * G.degree[y])

    size = G.number_of_nodes()
    S = lil_matrix((size, size))
    node_index_map = nodes_to_indexes(G)

    for x, y in nx.complement(G).edges():
        _x = node_index_map[x]
        _y = node_index_map[y]

        S[_x, _y] = __cosine_similarity(G, x, y)
        # S[y, x] = S[x, y]

    return S.tocsr()


def hub_depressed(G: nx.Graph) -> csr_matrix:
    """Compute the Hub Depressed Index for all nodes in the Graph.
    Each similarity value is defined as:

    .. math::
        S(x, y) = \\frac{2 |\\Gamma(x) \\cap \\Gamma(y)|}{\\max(k_x, k_y)}

    where \\(\\Gamma(x)\\) are the neighbors of node \\(x\\)
    and \\(k_x\\) is the degree of the node \\(x\\).


    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)

    Returns
    -------
    S: csr_matrix : the Similarity Matrix (in sparse format)

    Notes
    -----
    This index is the same as the previous one but with the
    opposite goal as it avoids the formation of
    links between hubs and low degree nodes in the networks.
    The Hub depressed index promotes the links evolution
    between the hubs as well as the low degree nodes.
    """

    def __hub_depressed(G: nx.Graph, x, y) -> float:
        """Compute the Hub Depressed Index for 2 given nodes."""
        return __common_neighbors(G, x, y) / max(G.degree[x], G.degree[y])

    size = G.number_of_nodes()
    S = lil_matrix((size, size))
    node_index_map = nodes_to_indexes(G)

    for x, y in nx.complement(G).edges():
        _x = node_index_map[x]
        _y = node_index_map[y]

        S[_x, _y] = __hub_depressed(G, x, y)
        # S[y, x] = S[x, y]

    return S.tocsr()


def hub_promoted(G: nx.Graph) -> csr_matrix:
    """Compute the Hub Promoted Index for all nodes in the Graph.
    Each similarity value is defined as:

    .. math::
        S(x, y) = \\frac{2 |\\Gamma(x) \\cap \\Gamma(y)|}{\\min(k_x, k_y)}

    where \\(\\Gamma(x)\\) are the neighbors of node \\(x\\)
    and \\(k_x\\) is the degree of the node \\(x\\).

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)

    Returns
    -------
    S: csr_matrix : the Similarity Matrix (in sparse format)

    Notes
    -----
    This similarity index promotes the formation of links between
    the sparsely connected nodes and hubs.
    It also tries to prevent links formation between the hub nodes.
    """

    def __hub_promoted(G: nx.Graph, x, y) -> float:
        """Compute the Hub Promoted Index for 2 given nodes."""
        return __common_neighbors(G, x, y) / min(G.degree[x], G.degree[y])

    size = G.number_of_nodes()
    S = lil_matrix((size, size))
    node_index_map = nodes_to_indexes(G)

    for x, y in nx.complement(G).edges():
        _x = node_index_map[x]
        _y = node_index_map[y]

        S[_x, _y] = __hub_promoted(G, x, y)
        # S[y, x] = S[x, y]

    return S.tocsr()


def jaccard(G: nx.Graph) -> csr_matrix:
    """Compute the Jaccard Coefficient for all nodes in the Graph.
    Each similarity value is defined as:

    .. math::
        S(x, y) = \\frac{| \\Gamma(x) \\cap \\Gamma(y)|}{| \\Gamma(x) \\cup \\Gamma(y)|}

    where \\(\\Gamma(x)\\) are the neighbors of the node \\(x\\).

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)

    Returns
    -------
    S: csr_matrix : the Similarity Matrix (in sparse format)

    Notes
    -----
    The Jaccard coefficient is defined as the probability of selection
    of common neighbors of pairwise vertices from all the neighbors of
    either vertex. The pairwise Jaccard score increases with the number of
    common neighbors between the two vertices considered. Some researcher
    (**Liben-Nowell et al.**) demonstrated that this similarity metric
    performs worse as compared to Common Neighbors.
    """

    def __jaccard(G: nx.Graph, x, y) -> float:
        """Compute the Jaccard Coefficient for 2 given nodes."""
        total_neighbor_number = len(set(G[x]).union(set(G[y])))
        if total_neighbor_number == 0:
            return 0
        
        return __common_neighbors(G, x, y) / total_neighbor_number

    size = G.number_of_nodes()
    S = lil_matrix((size, size))
    node_index_map = nodes_to_indexes(G)

    for x, y in nx.complement(G).edges():
        _x = node_index_map[x]
        _y = node_index_map[y]

        S[_x, _y] = __jaccard(G, x, y)
        # S[y, x] = S[x, y]

    return S.tocsr()


def node_clustering(G: nx.Graph) -> csr_matrix:
    """Compute the Hub Depressed Index for all nodes in the Graph.
    Each similarity value is defined as:

    .. math::
        S(x, y) = \\sum_{z \\in \\Gamma(x) \\cap \\Gamma(y)} C(z)

    where

    .. math::
        C(z) = \\frac{t(z)}{k_z(k_z - 1)}

    is the clustering coefficient of node \\(z\\), \\(t(z)\\)
    is the total triangles passing through the node \\(z\\),
    \\(\\Gamma(x)\\) are the neighbors of node \\(x\\)
    and \\(k_x\\) is the degree of the node \\(x\\).

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)

    Returns
    -------
    S: csr_matrix : the Similarity Matrix (in sparse format)

    Notes
    -----
    This index is also based on the clustering coefficient property
    of the network in which the clustering coefficients of all
    the common neighbors of a seed node pair are computed
    and summed to find the final similarity score of the pair.
    """

    def __t(G: nx.Graph, z) -> int:
        """Number of triangles passing through the node z"""
        return nx.triangles(G, z)

    def __C(G: nx.Graph, z) -> float:
        """Clustering Coefficient"""
        z_degree = G.degree[z]

        # avoiding 0 divition error
        if z_degree == 1:
            return 0

        return __t(G, z) / (z_degree * (z_degree - 1))

    def __node_clustering(G: nx.Graph, x, y) -> float:
        """Compute the Node Clustering Coefficient for 2 given nodes."""
        return sum([__C(G, z) for z in (set(G[x]) & set(G[y]))])

    size = G.number_of_nodes()
    S = lil_matrix((size, size))
    node_index_map = nodes_to_indexes(G)

    for x, y in nx.complement(G).edges():
        _x = node_index_map[x]
        _y = node_index_map[y]

        S[_x, _y] = __node_clustering(G, x, y)
        # S[y, x] = S[x, y]

    return S.tocsr()


def preferential_attachment(G: nx.Graph, sum: bool = False) -> csr_matrix:
    """Compute the Preferential Attachment Index for all nodes in the Graph.
    Each similarity value is defined as:

    .. math::
        S(x, y) = k_x k_y

    where \\(k_x\\) is the degree of the node \\(x\\).

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)
    sum: bool :
        Replace multiplication with summation when computing the index.
         (Default value = False)

    Returns
    -------
    S: csr_matrix : the Similarity Matrix (in sparse format)

    Notes
    -----
    The idea of preferential attachment is applied to generate a growing
    scale-free network. The term growing represents the incremental nature
    of nodes over time in the network. The likelihood incrementing new
    connection associated with a node \\(x\\) is proportional to
    \\(k_x\\), the degree of the node.

    This index shows the worst performance on most networks.
    The **simplicity** (as it requires the least information
    for the score calculation) and the computational time of this metric
    are the main advantages. PA shows better results if larger
    degree nodes are densely connected,
    and lower degree nodes are rarely connected.

    In the above equation, summation can also be used instead of
    multiplication as an aggregate function (`sum = True`).
    """

    def __preferential_attachment(G: nx.Graph,
                                  x,
                                  y,
                                  sum: bool = False) -> float:
        """Compute the Preferential Attachment Index for 2 given nodes."""
        return G.degree[x] * G.degree[y] if not sum else G.degree[
            x] + G.degree[y]

    size = G.number_of_nodes()
    S = lil_matrix((size, size))
    node_index_map = nodes_to_indexes(G)

    for x, y in nx.complement(G).edges():
        _x = node_index_map[x]
        _y = node_index_map[y]

        S[_x, _y] = __preferential_attachment(G, x, y, sum=sum)
        # S[y, x] = S[x, y]

    return S.tocsr()


def resource_allocation(G: nx.Graph) -> csr_matrix:
    """Compute the Resource Allocation Index for all nodes in the Graph.
    Each similarity value is defined as:

    .. math::
        S(x, y) = \\sum_{z \\in \\Gamma(x) \\cap \\Gamma(y)} \\frac{1}{k_z}

    where \\(\\Gamma(x)\\) are the neighbors of node \\(x\\) and
    \\(k_x\\) is the degree of the node \\(x\\).

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)

    Returns
    -------
    S: csr_matrix : the Similarity Matrix (in sparse format)

    Notes
    -----
    Consider two non-adjacent vertices \\(x\\) and \\(y\\).
    Suppose node \\(x\\) sends some resources to \\(y\\)
    through the common nodes of both \\(x\\) and \\(y\\)
    then the similarity between the two vertices is computed in terms
    of resources sent from \\(x\\) to \\(y\\).

    The difference between Resource Allocation (**RA**) and
    Adamic and Adar (**AA**) is that the RA index
    heavily penalizes to higher degree nodes compared to the AA index.
    Prediction results of these indices become almost the same
    for smaller average degree networks.

    This index shows good performance on heterogeneous
    networks with a high clustering coefficient, especially
    on transportation networks.
    """

    def __resource_allocation(G: nx.Graph, x, y) -> float:
        """Compute the Resource Allocation Index for 2 given nodes."""
        return sum([1 / G.degree[z] for z in set(G[x]) & set(G[y])])

    size = G.number_of_nodes()
    S = lil_matrix((size, size))
    node_index_map = nodes_to_indexes(G)

    for x, y in nx.complement(G).edges():
        _x = node_index_map[x]
        _y = node_index_map[y]

        S[_x, _y] = __resource_allocation(G, x, y)
        # S[y, x] = S[x, y]

    return S.tocsr()


def sorensen(G: nx.Graph) -> csr_matrix:
    """Compute the Sorensen Index for all nodes in the Graph.
    Each similarity value is defined as:

    .. math::
        S(x, y) = \\frac{2 |\\Gamma(x) \\cap \\Gamma(y)|}{k_x + k_y}

    where \\(\\Gamma(x)\\) are the neighbors of node \\(x\\)
    and \\(k_x\\) is the degree of the node \\(x\\).

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)

    Returns
    -------
    S: csr_matrix : the Similarity Matrix (in sparse format)

    Notes
    -----
    It is very similar to the Jaccard index. **McCune et al.** show
    that it is more robust than Jaccard against the outliers.
    """

    def __sorensen(G: nx.Graph, x, y) -> float:
        """Compute the Sorensen Index for 2 given nodes."""
        return (2 * __common_neighbors(G, x, y)) / (G.degree[x] + G.degree[y])

    size = G.number_of_nodes()
    S = lil_matrix((size, size))
    node_index_map = nodes_to_indexes(G)

    for x, y in nx.complement(G).edges():
        _x = node_index_map[x]
        _y = node_index_map[y]

        S[_x, _y] = __sorensen(G, x, y)
        # S[y, x] = S[x, y]

    return S.tocsr()
