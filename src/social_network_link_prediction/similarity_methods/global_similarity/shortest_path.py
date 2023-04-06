import networkx as nx
from scipy.sparse import csr_matrix, lil_matrix
from social_network_link_prediction.utils import nodes_to_indexes


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
    cutoff: int : # TODO
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

    S = lil_matrix((dim, dim))
    for source_node in lengths.keys():
        for dest_node in lengths[source_node].keys():
            S[nodes_to_indexes_map[source_node],
              nodes_to_indexes_map[dest_node]] = -lengths[source_node][
                  dest_node]

    return S.tocsr()
