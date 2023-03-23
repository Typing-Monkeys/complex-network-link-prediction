import networkx as nx


# TODO: documentazione !!
def nodes_to_indexes(G: nx.Graph) -> dict:
    return {node_name: index for index, node_name in enumerate(G.nodes)}


def get_node_from_index(data: dict, index: int) -> str:
    """
        Data una mappatura nodo-indice ritorna
        il nodo dato un indice.

        ARGS:
            - data: dizionario con mappatura nodo-indice
            - index: indice di cui si vuole conoscere il nodo

        RET:
            - str: nodo mappato con il dato indice
    """

    # Tutta questa operazione dovrebbe avere costo O(1)
    # vedi https://www.geeksforgeeks.org/python-get-key-from-value-in-dictionary

    key_list = list(data.keys())
    val_list = list(data.values())

    return key_list[val_list.index(index)]
