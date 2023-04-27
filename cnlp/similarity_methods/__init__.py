"""Similarity based Methods for Link Prediction.

Similarity-based metrics are the simplest one in link prediction,
in which for each pair \\(x\\) and \\(y\\) , a similarity score
\\(S(x, y)\\) is calculated.
The score \\(S(x, y)\\) is based on the structural or node’s properties of
the considered pair. The non-observed links (i.e., \\(U - E^T\\)) are assigned
scores according to their similarities. **The pair of nodes having a higher
score represents the predicted link between them**.
The similarity measures between every pair can be _calculated using several
properties of the network_, one of which is structural property.
Scores based on this property can be grouped in several categories
like **local** and **global**, and so on.
"""
