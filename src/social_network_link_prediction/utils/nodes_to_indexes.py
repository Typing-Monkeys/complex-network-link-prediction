import networkx as nx


# TODO: documentazione !!
def nodes_to_indexes(G: nx.Graph) -> dict:
    """

    Parameters
    ----------
    G: nx.Graph :
        

    Returns
    -------

    """
    return {node_name: index for index, node_name in enumerate(G.nodes)}
