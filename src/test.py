import networkx as nx
import numpy as np
from social_network_link_prediction.similarity_methods.quasi_local_methods import lpi
import matplotlib.pyplot as plt

graph = nx.Graph()
graph.add_nodes_from([0, 1, 2, 3])
graph.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])

nx.draw(graph, with_labels=True)

print(lpi.lpi(graph, 0.1, 4))

plt.show()
