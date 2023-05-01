import networkx as nx
import numpy as np
import scipy.sparse as scipy
import math
import itertools
from cnlp.utils import nodes_to_indexes
from scipy.sparse import lil_matrix, csr_matrix
from typing import Generator


def MI(G: nx.Graph) -> scipy.csr_matrix:
    """Neighbor Set Information

    Il modello di link prediction basato su information theory che sfrutta la neighbor
    set information è un approccio utilizzato per prevedere la probabilità di esistenza
    di un link tra due nodi in una rete. In questo modello, l'informazione contenuta nei
    neighbor set dei due nodi in questione viene utilizzata per stimare la probabilità
    di connessione.

    L'idea alla base di questo modello è che i nodi che hanno molti neighbor in comune
    sono più propensi a essere connessi tra loro rispetto a nodi con neighbor set diversi.
    Questo perché i nodi con neighbor set simili tendono a essere coinvolti in attività
    simili all'interno della rete, come ad esempio
    partecipare agli stessi gruppi o condividere gli stessi interessi.

    Per utilizzare questa informazione per prevedere la probabilità di connessione tra due nodi,
    il modello utilizza l'entropia di Shannon, una misura dell'incertezza di una
    distribuzione di probabilità.
    In particolare, l'entropia viene calcolata sui neighbor set dei due nodi, e la differenza tra le
    entropie dei due set viene utilizzata per stimare la probabilità di connessione.

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)

    Returns
    -------
    res_sparse: csr_matrix : the Similarity Matrix (in sparse format)
    """

    def overlap_info(G: nx.Graph, x, y, edge_num: int) -> float:
        """Two Information Definition.
        Overlapping nodes of different sets and
        the existence of link across different sets

        Parameters
        ----------
        G: nx.Graph :
            input graph
        x :
            first node
        y :
            second node
        edge_num: int :
            TODO !

        Returns
        -------
        s_Overlap: float : TODO
        """
        # ottenimento dei dati da cui ottenere le informazioni
        o_nodes = nx.common_neighbors(G, x, y)
        p_prob_overlap = -np.log2(prior(x, y, G, edge_num))

        # utilizzo delle informazioni per stimarsi la likelihood
        # con gli overlapping nodes
        coeff = 0
        overlap_info_value = 0
        overlap = 0

        for z in o_nodes:
            # degree of z
            kz = G.degree(z)

            coeff = 1 / (kz * (kz - 1))

            # sum over edges = neighbors of z
            overlap = 0
            for m, n in itertools.combinations(G.neighbors(z), 2):
                priorInfo = -np.log2(prior(m, n, G, edge_num))
                likelihoodInfo = -np.log2(likelihood(z, G))
                # print(f"a = {x}, b = {y}, priorInfo = { priorInfo},
                #   lilelihoodInfo = {likelihoodInfo}")
                # combine mutual information
                overlap += 2 * (priorInfo - likelihoodInfo)
                # print(f"a = {x}, b = {y}, zOverlap = { 2*(priorInfo -likelihoodInfo)}")

        # add average mutual information per neighbor
        overlap_info_value += coeff * overlap
        s_Overlap = overlap_info_value - p_prob_overlap
        return s_Overlap

    def prior(m, n, G: nx.Graph, edge_num: int) -> float:
        """Calcola la probabilità a priori dati due nodi e
        un grafo riferita alla probabilità con cui non si forma un cammino
        tra i due nodi

        Parameters
        ----------
        m :
            first node
        n :
            second node
        G: nx.Graph :
            input graph
        edge_num: int :
            TODO

        Returns
        -------
        float: the prior probability
        """
        kn = G.degree(n)
        km = G.degree(m)

        return 1 - math.comb(edge_num - kn, km) / math.comb(edge_num, km)

    def likelihood(z, G: nx.Graph) -> float:
        """probabilità condizionata che in questo caso è definita come il clustering
        coefficient dei common neighbor dei nodi x e y

        Parameters
        ----------
        z :
            input node
        G: nx.Graph :
            input graph

        Returns
        -------
        float: TODO
        """
        kz = G.degree(z)
        N_triangles = nx.triangles(G, z)
        N_triads = math.comb(kz, 2)

        return N_triangles / N_triads

    I_Oxy = 0
    edge_num = G.number_of_edges()
    node_num = G.number_of_nodes()
    edge_num = G.number_of_edges()
    res_sparse = scipy.lil_matrix((node_num, node_num))

    nodes_to_indexes_map = nodes_to_indexes(G)
    for i, j in nx.complement(G).edges():
        I_Oxy = overlap_info(G, i, j, edge_num)
        res_sparse[nodes_to_indexes_map[i], nodes_to_indexes_map[j]] = I_Oxy

    return res_sparse.tocsr()


def path_entropy(G: nx.Graph, max_path: int = 3) -> csr_matrix:
    """Compute the Path Entropy Measure for all nodes in the Graph.

    This Similarity measure between two nodes \\(X\\) and \\(Y\\)
    is calculated with:

    .. math::
        S_{x,y} = -I(L^1_{xy}|U_{i=2}^{maxlen} D_{xy}^i)

    where \\(D^i_{xy}\\) represents the set consisting of all simple
    paths of length i between the two vertices and maxlen is the maximum
    length of simple path of the network.

    Parameters
    ----------
    G: nx.Graph :
        input Graph (a networkx Graph)
    max_path: int :
        maximal path length
         (Default value = 3)

    Returns
    -------
    similarity_matrix: csr_matix: the Similarity Matrix (in sparse format)
    """

    def simple_path_entropy(paths: Generator[list, None, None],
                            G: nx.Graph) -> float:
        """Calcola l'entropia data dalla probabilità che si vengano a creare
        i vari simple paths tra i nodi tra cui si
        vuole fare link prediction

        Parameters
        ----------
        paths: Generator[List] :
            generator ritornato dalla funzione nx.all_simple_paths()

        G: nx.Graph :
            input graph

        Returns
        -------
        float: simple path entropy
        """
        tmp = .0
        for path in paths:
            for a, b in list(nx.utils.pairwise(path)):
                tmp += new_link_entropy(G, a, b)
        return tmp

    def new_link_entropy(G: nx.Graph, a, b) -> float:
        """Calcola l'entropia basata sulla probabilità
        a priori della creazione del link diretto
        tra le coppie di noti senza link diretti

        Parameters
        ----------
        G: nx.Graph :
            input graph
        a :
            first node
        b :
            second node

        Returns
        -------
        float: entropy between node A and B
        """
        deg_a = G.degree(a)
        deg_b = G.degree(b)
        M = G.number_of_edges()

        return -1 * np.log2(1 - (math.comb(M - deg_a, deg_b) /
                                   math.comb(M, deg_b)))

    similarity_matrix = lil_matrix((G.number_of_nodes(), G.number_of_nodes()))
    nodes_to_indexes_map = nodes_to_indexes(G)
    missing_edges = list(nx.complement(G).edges())

    for elem in missing_edges:
        paths = nx.all_simple_paths(G, elem[0], elem[1], max_path)
        tmp = 0
        for i in range(2, (max_path + 1)):
            tmp += (1 / (i - 1)) * simple_path_entropy(paths=paths, G=G)
        tmp = tmp - new_link_entropy(G, elem[0], elem[1])
        similarity_matrix[nodes_to_indexes_map[elem[0]],
                          nodes_to_indexes_map[elem[1]]] = tmp
    return similarity_matrix.tocsr()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    G = nx.karate_club_graph()

    # converte gli id dei nodi in interi affinche possano essere usati come indici
    # G_to_int = nx.convert_node_labels_to_integers(G, 0)
    nx.draw(G, with_labels=True)

    ranking = MI(G)
    # da aggiungere informazioni dei nodi che hanno fatto ottentere il
    # ranking migliore

    # va preso il risultato più piccolo perchè si tratta di entropia
    print(ranking)
    for i, j in nx.complement(G).edges():
        if (ranking[i, j] == ranking.toarray().min()):
            print(
                f"Il link più probabile tra quelli possibili è tra {i} e {j}, con un valore di {ranking[i,j]}"
            )
    plt.show()
