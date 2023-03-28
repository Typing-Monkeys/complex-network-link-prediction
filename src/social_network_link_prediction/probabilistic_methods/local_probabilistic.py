import networkx as nx
import numpy as np
from collections import deque
from .MarkovRandomField import MarkovRandomField


# Il primo passo per questo approccio di link prediction è quello di derivarsi il primo tipo di 
# feature utili, ovvereo le Co-occurence probability features. Per farlo si implementano 2 passaggi

# il primo passaggio deriva un local neighborhood set per la coppia di nodi dati

# il secondo costruisce un modello probabilistico grafico (MRF) sfruttando il central neighborhood set derivato prima
# pre stimare la co-occurence probability dei 2 nodi


# derivazione del  central neighborhood set 
def enumerate_simple_paths(G, s, t, K):
    P = set() # Inizializza l'insieme vuoto di percorsi semplici di lunghezza K
    # Cerca i vicini a distanza (K-1) da s senza visitare t e salva tutte le informazioni sul percorso
    N = bfs(G, s, K-1, t)
    for e in N:
        # Se e e t sono connessi in G, aggiungi il percorso s -> e -> t a P
        if G.has_edge(e, t):
            P.add(tuple(nx.shortest_path(G, s, e) + nx.shortest_path(G, e, t)[1:])) # Aggiunge il percorso alla lista di percorsi P
    return P


# esegue la ricerca in ampiezza e restituisce un insieme di nodi a distanza (K-1) da s senza visitare t
def bfs(G, s, K, t=None):
    visited = set()
    queue = deque([(s, 0)])
    neighbors = set()
    while queue:
        node, dist = queue.popleft()
        visited.add(node)
        if dist == K:
            neighbors.add(node)
            continue
        for neighbor in G.neighbors(node):
            if neighbor not in visited and neighbor != t:
                queue.append((neighbor, dist+1))
    return neighbors


# Funzione che ci fa ottenere i nodi che vanno a formare il central neighborhood
# e che quindi risultano essere i più rilevanti. 
def select_central_node_set(G, s, t, maxSize):
    C = set() # Inizializza l'insieme vuoto del set di nodi centrali
    i = 2
    while i < 4:
        Pi = enumerate_simple_paths(G, s, t, i)
        # Ordina tutti i percorsi in Pi per lunghezza e score di frequenza
        sorted_paths = sorted(Pi, key=lambda x: (len(x), -sum(G.degree(node) for node in x)))
        for p in sorted_paths:
            # Aggiungi tutti i nodi lungo il percorso p a C se la dimensione di C è inferiore a maxSize
            if len(C) < maxSize:
                C |= set(p)
            else:
                break # Termina il ciclo se la dimensione di C raggiunge maxSize
        i += 1
    return C


# Costruzione del modello MRF locale per fare inferenza

# apprendere il markov random field per derivare le feature di co occurence probability
def co_occurrence_probability_feature_induction(C, s, t, NDI):
    R = set() # Inizializza l'insieme vuoto dei pattern rilevanti a C
    for ndi in NDI:
        if ndi.issubset(C):
            R.add(ndi)
    # Apprendi un modello MRF su C utilizzando R
    M = learn_mrf(C, R)
    # Inferisci la probabilità di co-occorrenza di s e t da M
    f = inference(M, s, t)
    return f


def learn_mrf(C, R):
    # Ottieni tutte le variabili in C e inizializza M
    variables = list(C)
    M = create_mrf(variables)
    # Aggiungi tutti i vincoli in R a M
    for constraint in R:
        M.add_constraint(constraint)
    # Aggiorna M fino a quando tutti i vincoli sono soddisfatti
    while not all(M.constraints_satisfied()):
        for constraint in R:
            M.enforce_constraint(constraint)
    return M



def inference(M, s, t):
    # Inferisci la probabilità di co-occorrenza di s e t da M
    probability = M.get_edge_weight(s, t)
    return probability


def create_mrf(variables):
    # Inizializza un modello MRF con le variabili date
    M = MarkovRandomField()
    for variable in variables:
        M.add_node(variable)
    return M


# derivazione delle informazioni topologiche
# la Katz mesure è una della misure topologiche più efficaci per link prediction

def get_path_number(G:nx.Graph, s, t, lenght):
    paths = [p for p in nx.all_simple_paths(G, source=s, target=t, cutoff=lenght)]
    num_of_path = len(paths)

    return num_of_path



# derivazione del valore Katz senza usare matrici di adiacenza, così da rendere il tutto più scalabile
# (inversione di matrici grandi è molto costosa)
def akatz(G:nx.Graph, s, t, beta, k = 4):
    # somma pesata del numero di cammini che connettono 2 nodi
    # lunghezza minore ha peso maggiore
    katz_value = 0
    for i in range(k):
        p_i = get_path_number(G, s, t, i)
        katz_value += beta*p_i
    return katz_value

# Per combinare le due feature ottenute si usa un framework di supervised learning



if __name__ == "__main__":
    import matplotlib.pyplot as plt

    G = nx.Graph()
    G.add_edges_from([(1, 2),(1, 3),(1, 4),(2, 4),(2, 5),(5, 6),(5, 7),(6, 7),(6, 8),(7, 8)])

    nx.draw(G, with_labels=True)
    f = co_occurrence_probability_feature_induction

    plt.show()