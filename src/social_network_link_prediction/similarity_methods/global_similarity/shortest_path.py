import networkx as nx
from scipy.sparse import csr_matrix, lil_matrix
from social_network_link_prediction.utils import nodes_to_indexes


def shortest_path(G: nx.Graph, cutoff: int = None) -> csr_matrix:
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
