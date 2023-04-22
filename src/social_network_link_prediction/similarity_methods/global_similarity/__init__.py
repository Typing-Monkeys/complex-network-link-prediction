"""Collection of Global Similarity Methods for Link Prediction.

Global indices are computed using entire
topological information of a network.
The computational complexities of such methods
are higher and seem to be infeasible for large networks.
"""
from .katz_index import katz_index
from .rooted_page_rank import rooted_page_rank
from .shortest_path import shortest_path
from .random_walk import link_prediction_rwr
