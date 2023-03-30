import networkx as nx
from sknetwork.data import house, karate_club, load_konect


class Configs:
    dataset_easy = house()
    dataset_medium = karate_club()
    dataset_hard = load_konect('ego-facebook', verbose=False)["adjacency"]
    timeout = 5 * 60  # 5 minuit

    @staticmethod
    def load_easy_dataset():
        adj = Configs.dataset_easy
        g = nx.from_numpy_array(adj)

        return g, adj

    @staticmethod
    def load_medium_dataset():
        adj = Configs.dataset_medium
        g = nx.from_numpy_array(adj)

        return g, adj

    @staticmethod
    def load_hard_dataset():
        adj = Configs.dataset_hard
        g = nx.from_numpy_array(adj)

        return g, adj
