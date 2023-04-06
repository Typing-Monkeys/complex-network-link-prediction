import networkx as nx
import numpy as np
from networkx.algorithms.community import louvain_communities
from social_network_link_prediction.utils import to_adjacency_matrix
from scipy.sparse import csr_matrix, lil_matrix as lil_pump
from typing import List, Set
from itertools import permutations, product


def __random_changes(A_0: lil_pump, p: np.float32=.05) -> lil_pump:
    A_chaos = A_0.copy()
    num_changes = int(A_chaos.shape[0]**2 * p)
    indexes = []

    while len(indexes) < num_changes:
        x = np.random.randint(0, high=A_chaos.shape[0])
        y = np.random.randint(0, high=A_chaos.shape[0])

        if x == y:
            continue
        
        if (x,y) in indexes or (y,x) in indexes:
            continue

        indexes.append((x,y))

    for x,y in indexes:
        A_chaos[x,y] = abs(A_0[x, y] - 1)
        A_chaos[y,x] = A_chaos[x,y]
    
    return A_chaos


def __generate_samples(A_0: lil_pump, n: int, p: np.float32=.05, seed: int=42) -> list:
    samples_list = []
     
    for _ in range(n):
        A_chaos = __random_changes(A_0, p)
        G_chaos = nx.Graph(A_chaos)
        res = louvain_communities(G_chaos, seed)
        samples_list.append(res)

    return samples_list

def __link_reliability(A_0: lil_pump, samples_list: List[List[Set]], x: int, y: int):
    summing_result = 0
    for sample in samples_list:
        x_block = __get_node_block(sample, x)
        y_block = __get_node_block(sample, y)

        l_xy = np.sum([A_0[x, y] for x,y in product(x_block, y_block)]) + 1
        r_xy = (len(x_block) * len(y_block)) + 2

        tmp_ratio = l_xy / r_xy

        summing_result += ...


def stochastic_block_model(G: nx.Graph, n: int, p: np.float32=.05, seed: int=42) -> csr_matrix:
    A_0 = to_adjacency_matrix(G)
    samples_list = __generate_samples(A_0, p)
 
    R = lil_pump()
    # coppie di nodi non collegati = []
    
    # for x, y in coppie di nodi non collegati:
    #     R[x, y] = __link_reliability(A_0, samples_list, x, y)

    
    return R.tocsr()


def __get_node_block(sample: int, x: int):
    for i in sample:
        if x in i:
            return i

    return -1


if __name__ == "__main__":
    
    G = nx.karate_club_graph()
    A = to_adjacency_matrix(G)
    # print(stochastic_block_model())

    # print(__random_changes(A))

    # print()
    s = __generate_samples(A, 10)
    # print(s)
    # print(len(s))
    # for i in s:
    #     print(len(i))
    #     print(i)
    print(__link_reliability(A, s, 0, 10))