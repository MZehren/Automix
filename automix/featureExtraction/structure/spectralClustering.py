"""
Generate the structure of the song with the mean square energy (MSE)
"""

import numpy as np

from automix.featureExtraction.estimator import Estimator, Parameter
from automix.model.classes.segment import Segment
from automix.model.classes.signal import Signal

from . import spectralClusteringSegmenter as segmenter


class SpectralClustering(Estimator):
    # TODO: covert the low threshold in percentile instead of absolute value
    def __init__(self, input="path", output="spectralClustering", cachingLevel=0, forceRefreshCache=False):
        """
        """
        self.inputs = [input]
        self.outputs = [output]
        self.parameters = {}
        self.cachingLevel = cachingLevel
        self.forceRefreshCache = forceRefreshCache

    def predictOne(self, path: str):

        X_cqt, X_timbre, beat_intervals = segmenter.features(path)

        boundaries, beat_intervals, labels = segmenter.lsd(X_cqt, X_timbre, beat_intervals, {"num_types": False})
        result = Signal(labels, times=[beat_intervals[i][0] for i in boundaries[:-1]], sparse=True)
        return (result,)

