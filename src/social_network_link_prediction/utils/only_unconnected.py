import networkx as nx
from scipy.sparse import csr_matrix
from .nodes_to_indexes import nodes_to_indexes

def only_unconnected(graph: nx.Graph, sim_matrix: csr_matrix) -> csr_matrix:
    """Filter the given matrix and return only previously unconnected
    nodes "similarity" values

    Parameters
    ----------
    graph: nx.Graph :
        input graph
    sim_matrix: csr_matrix :
        similarity matrix

    Returns
    -------
    sim_matrix: csr_matrix : the similarity matrix without the previously
    connected nodes similarity
    """
    node_idexies_map = nodes_to_indexes(graph)

    for x, y in graph.edges():
        sim_matrix[node_idexies_map[x], node_idexies_map[y]] = 0

    sim_matrix.eliminate_zeros()

    return sim_matrix.tocsr()