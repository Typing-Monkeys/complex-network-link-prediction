"""Collection of Quasi-local Similarity Methods for Link Prediction.

Quasi-local indices have been introduced as a trade-off between
local and global approaches or performance and complexity.
These metrics are as efficient to compute as local indices.
Some of these indices extract the entire topological information
of the network.

The time complexities of these indices are still below compared
to the global approaches.
"""
from .path_of_length_three import path_of_length_three
from .LPI import local_path_index
