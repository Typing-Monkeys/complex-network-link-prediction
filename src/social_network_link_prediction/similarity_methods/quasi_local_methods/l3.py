import networkx as nx

def l3(graph, x, y):
    
    # Calolate degree node x,y
    
    k_x = graph.degree(x)
    k_y = graph.degree(y)
    
    # Initializate score
    score = 0

    # Enroll all neighbors 
    for u in graph.neighbors(x):
        for v in graph.neighbors(y):
            # Base case
            if u == v:
                continue
            # Calcolate the score with the multiply of value of node and divide for degree
            if graph.has_edge(u, v):
                a_xu = graph.get_edge_data(x, u)['weight'] # Change this for unweighted graph
                a_uv = graph.get_edge_data(u, v)['weight'] # Change this for unweighted graph
                a_vy = graph.get_edge_data(v, y)['weight'] # Change this for unweighted graph
                score += (a_xu * a_uv * a_vy) / (k_x * k_y)
    return score

if __name__ == "__main__":

    '''
        If we have a unweighted graph the weight have value of 1 
    '''
    
    """
        LP3 -> graph weight
    """

    # create an empty graph
    G = nx.Graph()

    # add nodes
    G.add_nodes_from([1, 2, 3, 4])

    # add weighted edges
    G.add_weighted_edges_from([(1, 2, 0.5), (2, 3, 1.0), (3, 4, 2.0), (4, 1, 1.5)])


    # access edge weights
    '''
        Show weights of graph

        print(G[1][2]['weight'])  # print 0.5
        print(G[3][4]['weight'])  # print 2.0
    '''

    result = l3(G,1,2)

    print(result)