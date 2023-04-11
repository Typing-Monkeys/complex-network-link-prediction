import networkx as nx
from scipy.sparse import lil_matrix, linalg, hstack, lil_array, csr_matrix
from social_network_link_prediction.utils import to_adjacency_matrix
from social_network_link_prediction.utils import nodes_to_indexes

def link_prediction_rwr(G:nx.Graph, c:int = 0.05, max_iters:int = 10) -> csr_matrix:
    
    # Convert the graph G into an adjacency matrix A
    A = to_adjacency_matrix(G)
    
    # Extract the number of nodes of matrix A
    m = A.shape[0]

    # Initialize the diagonal matrix D as a sparse lil_matrix
    D = lil_matrix(A.shape)

    # Create a map that associates each node with a row index in matrix A
    nodes_to_indexes_map = nodes_to_indexes(G)

    # Build the diagonal matrix D so that the elements on the diagonal are equal to the degree of the corresponding node
    for node in G.nodes():
        D[nodes_to_indexes_map[node], nodes_to_indexes_map[node]] = G.degree[node]
    
    # Convert the diagonal matrix D into csc_matrix format
    D = D.tocsc()
    
    # Build the normalized transition matrix W_normalized
    W_normalized = linalg.inv(D) @ A.tocsc()

    # Initialize an matrix to hold the similarities between node pairs
    # We put an initial column made of Zeros so we can use the hstack method later on and keep the code more clean
    similarity_matrix = csr_matrix((m,1))

    # For each node i, create a probability vector and perform the random walk with restart starting from that node
    for i in range(m):
        e = lil_array((m,1))
        e[i,0] = 1
        # Concatenate the similarity vectors into a similarity matrix
        # The use of hstack allows the lil_array returned from the random walk fuction to be trasposed and added to the similarity matrix
        # As a new column in just one line of code
        similarity_matrix = hstack([similarity_matrix, random_walk_with_restart(e = e, W_normalized = W_normalized, c = c, max_iters= max_iters)])

    # Return the similarity matrix and remove the fisrt column
    # In order to keep the results consistent without the added column of zeros at the beginning
    return csr_matrix(similarity_matrix)[:,1:]


def random_walk_with_restart(e:lil_array, W_normalized:csr_matrix, c:int = 0.05, max_iters:int = 100) -> lil_array:
    
    # Initialize the current probability vector to the initial one and the error to 1
    old_e = e
    err = 1.

    # Perform the random walk with restart until the maximum number of iterations is reached or the error becomes less than 1e-6
    for _ in range(max_iters):
        e = (c * (W_normalized @ old_e)) + ((1 - c) * e)
        err = linalg.norm(e - old_e, 1)
        if err <= 1e-6:
            break
        old_e = e

    # Return the current probability vector
    return e

if __name__ == '__main__':

    graph = nx.karate_club_graph()
    
    # Apply random walk with restart on this network
    predicted_adj_matrix = link_prediction_rwr(graph, c = 0.05, max_iters=10)

    adj_matrix = nx.to_numpy_array(graph)

    # Initialize the feature and coefficient matrices randomly
    num_nodes = adj_matrix.shape[0]
    # Get the predicted edges
    predicted_edges = []
    for u in range(adj_matrix.shape[0]):
        for v in range(u + 1, adj_matrix.shape[1]):
            if graph.has_edge(u, v):
                continue
            w = predicted_adj_matrix[u, v]
            predicted_edges.append((u, v, w))

    # Sort the predicted edges by weight in descending order
    predicted_edges.sort(key=lambda x: x[2], reverse=True)

    # Print the predicted edges with their weight score
    print("Top Predicted edges:")
    for edge in predicted_edges[:10]:
        print(f"({edge[0]}, {edge[1]}): {edge[2]}")
