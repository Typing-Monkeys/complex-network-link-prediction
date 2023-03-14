import networkx as nx
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse import triu
from scipy.special import factorial
from utils import to_adjacency_matrix

def __matrix_power(A, k):
    result = A.copy()
    for _ in range(k-1):
        result @= A

    return result

def __number_of_k_length_cycles(A: csr_matrix, k):
    return A.trace() / factorial(k)

def __number_of_k_length_paths(A: csr_matrix, k):
    return triu(A).sum()/factorial(k)

def __generalized_clustering_coefficient(A: csr_matrix):
    # trace = 
    pass

def clustering(G: nx.Graph, k):
    A = to_adjacency_matrix(G)
    k_power_A = __matrix_power(A, k)
    num_cycle = __number_of_k_length_cycles(k_power_A, k)
    total_path = __number_of_k_length_paths(k_power_A, k)

    print(k_power_A.todense())
    print(num_cycle)
    print(total_path)
