"""
Get the peaks exceding a threshold. Returns only the maximum value in the threshold exceed
"""

import sys
from typing import List

import numpy as np
from scipy.ndimage import filters

from automix.featureExtraction.estimator import Estimator, Parameter
from automix.model.classes.signal import Signal, SparseSegmentSignal


class CoreFinder(Estimator):
    def __init__(self,
                 inputValues="samples",
                 inputGrid="period",
                 parameterIncludeBorders=True,
                 outputPeaks="core",
                 cachingLevel=0,
                 forceRefreshCache=False):
        """Look at the rms of the signal for a segment and label it as a core if it's abov the rms of the full track

        """
        self.parameters = {"includeBorders": Parameter(parameterIncludeBorders)}
        self.inputs = [inputValues, inputGrid]
        self.outputs = [outputPeaks]
        self.cachingLevel = cachingLevel
        self.forceRefreshCache = forceRefreshCache

    def _rms(self, values):
        return np.sqrt(np.mean(np.square(values), axis=0))

    def predictOne(self, values: Signal, grid: Signal):
        mean = self._rms(values)
        times = grid.times
        if self.parameters["includeBorders"].value:
            times = [0] + list(times) + [99999]
        positionTuples = [(times[i], times[i + 1]) for i in range(len(times) - 1)]

        result = SparseSegmentSignal([self._rms(values.getValues(start, stop)) > mean for start, stop in positionTuples],
                                     [(start, stop) for start, stop in positionTuples])
        return (result, )
