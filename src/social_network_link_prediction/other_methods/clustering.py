import networkx as nx
from scipy import sparse
from scipy.sparse import csr_matrix
from scipy.sparse import triu
from scipy.special import factorial
from utils import to_adjacency_matrix
import matplotlib.pyplot as plt


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


graph = nx.Graph()
graph.add_nodes_from([0, 1, 2, 3, 4, 5, 6,7, 8])
graph.add_edges_from([(0, 1), (1, 2), (2, 3), (4, 5),(5, 6),(6, 4), (7,7), (8,0), (3,8)])


# print(find_cycle(graph,4))
# print(find_cycles(graph, 4))
# print(find_cycles(G, 3)))


V = graph.number_of_nodes()
 
def DFS(graph, marked, n, vert, start, count):
    # mark the vertex vert as visited
    marked[vert] = True
  
    # if the path of length (n-1) is found
    if n == 0:
        # mark vert as un-visited to make
        # it usable again.
        marked[vert] = False
  
        # Check if vertex vert can end with
        # vertex start
        if graph[vert][start] == 1:
            count = count + 1
            return count
        else:
            return count
  
    # For searching every possible path of
    # length (n-1)
    for i in range(V):
        if marked[i] == False and graph[vert][i] == 1:
 
            # DFS for searching path by decreasing
            # length by 1
            count = DFS(graph, marked, n-1, i, start, count)
  
    # marking vert as unvisited to make it
    # usable again.
    marked[vert] = False
    return count
  
# Counts cycles of length
# N in an undirected
# and connected graph.
def countCycles(G, n):

    # all vertex are marked un-visited initially.
    marked = [False] * V
    
    graph = nx.adjacency_matrix(G).toarray()

    # Searching for cycle by using v-n+1 vertices
    count = 0
    for i in range(V-(n-1)):
        count = DFS(graph, marked, n-1, i, i, count)
  
        # ith vertex is marked as visited and
        # will not be visited again.
        marked[i] = True
     
    return int(count/2)
  
# main :
from networkx.algorithms.community import k_clique_communities


def find_cycles(graph, k):
    cycles = 0
    for vertex in graph:
        visited = set()
        stack = [(vertex, [vertex])]
        while stack:
            (v, path) = stack.pop()
            visited.add(v)
            if len(path) == k and v in graph[path[0]]:
                cycles += 1
            if len(path) < k:
                for neighbor in graph[v]:
                    if neighbor not in visited:
                        stack.append((neighbor, path + [neighbor]))
    return cycles

# Pare che funziona
def find_all_cycles(G, source=None, cycle_length_limit=None):
    """forked from networkx dfs_edges function. Assumes nodes are integers, or at least
    types which work with min() and > ."""
    if source is None:
        # produce edges for all components
        nodes=[list(i)[0] for i in nx.connected_components(G)]
    else:
        # produce edges for components with source
        nodes=[source]
    # extra variables for cycle detection:
    cycle_stack = []
    output_cycles = set()
    
    def get_hashable_cycle(cycle):
        """cycle as a tuple in a deterministic order."""
        m = min(cycle)
        mi = cycle.index(m)
        mi_plus_1 = mi + 1 if mi < len(cycle) - 1 else 0
        if cycle[mi-1] > cycle[mi_plus_1]:
            result = cycle[mi:] + cycle[:mi]
        else:
            result = list(reversed(cycle[:mi_plus_1])) + list(reversed(cycle[mi_plus_1:]))
        return tuple(result)
    
    for start in nodes:
        if start in cycle_stack:
            continue
        cycle_stack.append(start)
        
        stack = [(start,iter(G[start]))]
        while stack:
            parent,children = stack[-1]
            try:
                child = next(children)
                
                if child not in cycle_stack:
                    cycle_stack.append(child)
                    stack.append((child,iter(G[child])))
                else:
                    i = cycle_stack.index(child)
                    if i < len(cycle_stack) - 2: 
                      output_cycles.add(get_hashable_cycle(cycle_stack[i:]))
                
            except StopIteration:
                stack.pop()
                cycle_stack.pop()
    
    return [list(i) for i in output_cycles]


n = 4
print("Total cycles of length ",n," are ",countCycles(graph, n))
print(list(k_clique_communities(graph, n)))
print(nx.find_cycle(graph))
print(find_cycles(graph, n))
print(nx.cycle_basis(graph))

print("last")
print(find_all_cycles(graph, cycle_length_limit=4))
nx.draw(graph, with_labels=True)

plt.show()