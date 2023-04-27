"""Dimentionality Reduction based Methods for Link Prediction.

The curse of dimensionality is a well-known
problem in machine learning.
Some researchers employ dimension reduction techniques
to tackle the above problem and apply it in the link prediction scenario.
"""
from .svd import link_prediction_svd
from .nmf import link_prediction_nmf
