import networkx as nx
from sknetwork.data import house, karate_club, load_konect, miserables


class Configs:
    __dataset = miserables(metadata=True)
    __graph_normal = None
    __graph_labels = None

    #dataset_easy = house()
    #dataset_medium = karate_club()
    #dataset_hard = load_konect('ego-facebook', verbose=False)["adjacency"]
    #timeout = 5 * 60  # 5 minuit

    @staticmethod
    def load_normal_dataset():
        if Configs.__graph_normal is None:
            Configs.__graph_normal = nx.to_networkx_graph(Configs.__dataset['adjacency'])

        return Configs.__graph_normal

    @staticmethod
    def load_labels_dataset():
        if Configs.__graph_labels is None:
            if Configs.__graph_normal is None:
                Configs.load_normal_dataset()
            
            names = Configs.__dataset['names']
            map = {idx: name for idx, name in enumerate(names)}

            Configs.__graph_labels = nx.relabel_nodes(Configs.__graph_normal, map)

        return Configs.__graph_labels
