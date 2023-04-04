"""Collection of Local Similarity Methods for Link Prediction.

Local indices are generally calculated using information about
common neighbors and node degree.
These indices **consider immediate neighbors of a node**.
"""
from .common_neighbors import common_neighbors
from .jaccard import jaccard
from .admic_adar import adamic_adar
from .preferential_attachment import preferential_attachment
from .cosine_similarity import cosine_similarity
from .hub_promoted import hub_promoted
from .hub_depressed import hub_depressed
from .sorensen import sorensen
from .node_clustering import node_clustering
from .resource_allocation import resource_allocation
