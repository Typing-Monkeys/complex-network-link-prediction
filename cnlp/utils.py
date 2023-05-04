"""Some utilities functions."""
import networkx as nx
import numpy as np
from typing import Union
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from typing import Dict, List, Tuple


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


def only_unconnected(graph: nx.Graph, sim_matrix: lil_matrix) -> csr_matrix:
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

    sim_matrix = sim_matrix.tocsr()
    sim_matrix.eliminate_zeros()

    return sim_matrix.tocsr()


def get_top_predicted_link(predicted_adj_matrix: csr_matrix,
                           number_of_nodes: int,
                           pct_new_link: float,
                           name_index_map: Dict[any, int],
                           verbose: bool = False) -> List[Tuple[any, any]]:
    """Get the edges (new link) with the highest predicted probabilities
                           verbose: bool = False):

    Parameters
    ----------
    predicted_adj_matrix: csr_matrix,
        input Similarity Matrix
    number_of_nodes: int :
        number of node in the graph
    pct_new_link: float :
        top x% best new links
    name_index_map: Dict[any, int] :
        node to index (starting from 0) mapping
    verbose: bool :
        if True print some usefull outputs
         (Default value = False)

    Returns
    -------
    new_link: List[Tuple[any, any]] : top % new link predicted
    """
    max_possible_edges = predicted_adj_matrix.nnz
    number_of_new_link = int(np.ceil(max_possible_edges / 100 * pct_new_link))
    edges = []
    new_link = []
    for i in range(number_of_nodes):
        for j in range(i + 1, number_of_nodes):
            prob = predicted_adj_matrix[i, j]
            edges.append((i, j, prob))

    edges = sorted(edges, key=lambda x: x[2],
                   reverse=True)[:number_of_new_link]

    new_link = []
    for edge in edges:
        if verbose:
            print(
                f"({name_index_map[edge[0]][0]}, {name_index_map[edge[1]][0]}) - Similarity: {edge[2]:.2f}"
            )

        new_link.append(
            (name_index_map[edge[0]][0], name_index_map[edge[1]][0]))

    return new_link
