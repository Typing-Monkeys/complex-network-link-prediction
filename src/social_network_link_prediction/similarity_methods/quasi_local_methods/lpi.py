import networkx as nx
import numpy as np


def lpi(adj_matrix, epsilon, n):
    # Convert the adjacency matrix to a NumPy array float
    A = np.array(adj_matrix, dtype=float)
    
    # Initialize the result matrix to A^2 and result
    Ai = A @ A
    result = Ai
    # Calculate the remaining terms of the sum
    for i in range(3, n): 
        Ai = A @ Ai
        result += epsilon**(i-2) * Ai

    return result


if __name__ == "__main__":

    '''
        LPI: 
            Parameters: 
                        adj_matrix,
                        epsilon,
                        n    
    '''

    epsilon = 0.1
    n = 4

    adj_matrix = [[0, 1, 1], 
                  [1, 0, 1], 
                  [1, 1, 0]]
    
    result_lpi = lpi(adj_matrix, epsilon, n)
    print(result_lpi)
    