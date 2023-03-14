from social_network_link_prediction.other_methods import clustering
import networkx as nx
import matplotlib.pyplot as plt

graph = nx.Graph()

# # Add nodes to the graph
graph.add_nodes_from([0, 1, 2, 3])

# Add edges to the graph
graph.add_edges_from([(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)])


clustering(graph, 4)

nx.draw(graph)
plt.show()