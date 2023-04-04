"""Collection of Global Similarity Methods for Link Prediction.

Global indices are computed using entire
topological information of a network.
The computational complexities of such methods
are higher and seem to be infeasible for large networks.
"""
from .katz_index import katz_index
