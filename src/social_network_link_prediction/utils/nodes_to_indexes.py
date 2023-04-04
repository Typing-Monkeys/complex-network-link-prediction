import networkx as nx
from typing import Dict, Any


def nodes_to_indexes(G: nx.Graph) -> Dict[Any, int]:
    """Node Label - Index encoder

    Associate, for each node label, and index starting from 0.

    Parameters
    ----------
    G: nx.Graph :
        grafo di cui ottenere la mappatura Nodo-indice

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
