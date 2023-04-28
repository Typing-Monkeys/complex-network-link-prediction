"""Some utilities functions."""
import networkx as nx
import numpy as np
from typing import Union
from scipy.sparse import csr_matrix, csc_matrix


def nodes_to_indexes(G: nx.Graph) -> dict[any, int]:
    """Node Label - Index encoder

    Associate, for each node label, and index starting from 0.

    Parameters
    ----------
    G: nx.Graph :
        the graph from which you want the node-to-index mapping

    Returns
    -------
    Dict[Any, int]: the encoding Node Label - Index dictionary

    Notes
    -----
    The method `Graph.nodes` return the nodes in the exactly same order, and
    the first node (at index 0) represent the index 0 in the Adjacency Matrix
    obtained with the method `Graph.to_adjacency_matrix` or
    `Graph.to_numpy_array`.
    """
    return {node_name: index for index, node_name in enumerate(G.nodes)}


def to_adjacency_matrix(G: nx.Graph,
                        sparse: bool = True) -> Union[csc_matrix, np.ndarray]:
    """Convert a ginven Graph in to its Adjacency Matrix

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)
    sparse: bool:
        if True, return the Adjacency Matrix in sparse format,
        otherwise in full format.
         (Default value = True)

    Returns
    -------
    csc_matrix | np.ndarray: the Adjacency Matrix
    """
    # TODO: ricontrollare se i pesi servono
    return nx.adjacency_matrix(
        G, weight=None) if sparse else nx.to_numpy_array(G, weight=None)


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
