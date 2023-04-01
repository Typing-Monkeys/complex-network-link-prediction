import networkx as nx


def nodes_to_indexes(G: nx.Graph) -> dict:
    """TODO

    Parameters
    ----------
    G: nx.Graph :
        grafo di cui ottenere la mappatura Nodo-indice

    Returns
    -------
    Dict[Any, int]: dizionario di mappatura Nodo-indice
    """
    return {node_name: index for index, node_name in enumerate(G.nodes)}
