import networkx as nx
import numpy as np
from networkx.algorithms.community import louvain_communities
from social_network_link_prediction.utils import to_adjacency_matrix
from scipy.sparse import csr_matrix, lil_matrix
from scipy.special import comb
from typing import List, Set
from itertools import product, combinations_with_replacement


def __random_changes(A_0: csr_matrix, p: float = .05, seed: int = 42) -> lil_matrix:
    np.random.seed(seed)
    A_chaos = A_0.copy().tolil()
    num_changes = int(A_chaos.shape[0]**2 * p)
    indexes = []

    while len(indexes) < num_changes:
        x = np.random.randint(0, high=A_chaos.shape[0])
        y = np.random.randint(0, high=A_chaos.shape[0])

        if x == y:
            continue

        if (x, y) in indexes or (y, x) in indexes:
            continue

        indexes.append((x, y))

    for x, y in indexes:
        A_chaos[x, y] = abs(A_0[x, y] - 1)
        A_chaos[y, x] = A_chaos[x, y]

    return A_chaos


def __generate_samples(A_0: csr_matrix,
                       n: int,
                       p: np.float32 = .05,
                       seed: int = 42) -> List[List[Set]]:
    samples_list = []

    for _ in range(n):
        A_chaos = __random_changes(A_0, p, seed)
        G_chaos = nx.Graph(A_chaos)
        res = louvain_communities(G_chaos, seed)
        samples_list.append(res)

    return samples_list


def __get_node_block(sample: List[Set], x: int) -> Set:
    for i in sample:
        if x in i:
            return i


def __l(A_0: lil_matrix, alpha: Set, beta: Set) -> int:
    return np.sum([A_0[x, y] for x, y in product(alpha, beta)])


def __r(alpha: Set, beta: Set) -> int:
    len_a = len(alpha)
    len_b = len(beta)

    if alpha == beta:
        return len_a * (len_a - 1)

    return (len_a * len_b)


def __H(A_0: csr_matrix, P: List[Set]) -> float:
    res = 0
    for a, b in combinations_with_replacement(P, 2):
        r = __r(a, b)
        l0 = __l(A_0, a, b)

        res += np.log(r + 1) + np.log(comb(r, l0))

    return res


def __link_reliability(A_0: csr_matrix, samples_list: List[List[Set]], x: int,
                       y: int) -> float:
    summing_result = 0
    HP_mem = []

    for sample in samples_list:
        x_block = __get_node_block(sample, x)
        y_block = __get_node_block(sample, y)

        l_xy = __l(A_0, x_block, y_block) + 1
        r_xy = __r(x_block, y_block) + 2

        second_term = np.exp(-__H(A_0, sample))
        HP_mem.append(second_term)

        tmp_ratio = l_xy / r_xy

        summing_result += tmp_ratio * second_term

    return summing_result / np.sum(HP_mem)


def stochastic_block_model(G: nx.Graph,
                           n: int,
                           p: float = .05,
                           seed: int = 42) -> csr_matrix:
    A_0 = to_adjacency_matrix(G)
    samples_list = __generate_samples(A_0, n, p, seed=seed)

    R = lil_matrix(A_0.shape)

    for x, y in nx.complement(G).edges():
        R[x, y] = __link_reliability(A_0, samples_list, x, y)

    return R.tocsr()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    G = nx.karate_club_graph()
    A = to_adjacency_matrix(G)

    res = stochastic_block_model(G, 10)

    predicted_edges = []
    for u in range(res.shape[0]):
        for v in range(u + 1, res.shape[1]):
            if G.has_edge(u, v):
                continue
            w = res[u, v]
            predicted_edges.append((u, v, w))

    # Sort the predicted edges by weight in descending order
    predicted_edges.sort(key=lambda x: x[2], reverse=True)

    # Print the predicted edges with their weight score
    print("Top Predicted edges:")
    for edge in predicted_edges[:50]:
        print(f"({edge[0]}, {edge[1]}): {edge[2]}")

    nx.draw(G)
    plt.show()
