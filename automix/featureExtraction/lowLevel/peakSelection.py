"""
From multiple list of peaks, returns the k-first peaks timewise.

Add functionnalities to 
"""

import collections
import logging as log
from typing import List

import numpy as np
from scipy.ndimage import filters

from automix.featureExtraction.estimator import Estimator, Parameter
from automix.model.classes.signal import Signal


class PeakSelection(Estimator):
    """
    Merge from a list of signals all the values in one signal.

        * parameterClusterDistance: merge the values close to each other in the same value
        * parameterRelativeDistance: Discard the values with a position exceeding % pf the duration of the track
        * parameterSalienceTreshold: Discard the position where the segment in the grid following it has a value below this threshold in the inputSalience
        * parameterSalienceWindow: The size of the grid window to compute the salience
        * parameterMergeFunction: How the value of the merged peaks is computed
        * parameterAbsoluteTop: Absolute number of peaks to select. Set to None to keep all the peaks
    """

    def __init__(
            self,
            parameterAbsoluteTop=None,  # Absolute number of peaks to select. Set to None to kepp all the peaks
            parameterClusterDistance=0,  # The distance in seconds to cluster multiple features occuring at the same time.
            parameterMergeFunction=np.sum,  # How the clustered peaks' values are merged
            parameterRelativeDistance=1,  # Return peaks only within in the beginning % of the track
            parameterSalienceTreshold=0,  # Return peaks preceding a segment having at least this quantity in the Salience feature
            parameterSalienceWindow=8,  # the size of the window suceeding the peak to compute the salience
            inputPeaks=["cqtAmplitudeCheckerboardPeaks"],  # The peaks filtered
            inputGrid="strongBeats",  # The grid to compute the salience window and the duration of the track
            inputSalience=["cqtAmplitudeRMSE"],  # The list of features used for the salience
            outputPeaks="selectedPeaks",  # Name of the output
            outputNonSalient="nonSalientPeaks",
            cachingLevel=0,
            forceRefreshCache=True): #As long as there is no way of updating the cache when the input changes
        """
        Estimator selecting peaks from multiple list of peaks.
        """
        self.parameters = {
            "absoluteTop": Parameter(parameterAbsoluteTop),
            "clusterDistance": Parameter(parameterClusterDistance),
            "relativeDistance": Parameter(parameterRelativeDistance),
            "salienceTreshold": Parameter(parameterSalienceTreshold),
            "salienceWindow": Parameter(parameterSalienceWindow),
            "mergeFunction": Parameter(parameterMergeFunction)
        }
        self.inputs = [inputPeaks, inputGrid, inputSalience]
        self.outputs = [outputPeaks, outputNonSalient]
        self.cachingLevel = cachingLevel
        self.forceRefreshCache = forceRefreshCache

    def predictOne(self, peakSignals: List[Signal], grid: Signal, salienceSignals: List[Signal]):
        # Cluster the peaks to remove close outliers
        peaks = Signal.clusterSignals(peakSignals,
                                      minDistance=self.parameters["clusterDistance"].value,
                                      mergeValue=self.parameters["mergeFunction"].value)

        # Get the Salience of the following segment
        peaks, nonSalientPeaks = self.getSalientPoints(salienceSignals, grid, peaks)

        # Filter the peaks too far away from the start of the track
        peaks = self.getEarlyPeaks(peaks, grid)

        # Get the first absolute k-beats TODO: Set the selection to an "or" ? -> I don't like it so much, because we can't
        # disable the position filtering with an or
        peaks = Signal(peaks.values[:self.parameters["absoluteTop"].value],
                       times=peaks.times[:self.parameters["absoluteTop"].value],
                       sparse=True)

        return (peaks, nonSalientPeaks)

    def getEarlyPeaks(self, peaks, grid):
        """
        Filter the peaks by relative distance from the start
        """
        if self.parameters["relativeDistance"].value < 1:
            earlyPeaks = [
                i for i, pos in enumerate(peaks.times) if pos <= grid.duration * self.parameters["relativeDistance"].value
            ]
            # if len(earlyPeaks) == 0:
            #     earlyPeaks = [peaks[0]]
            peaks = Signal([peaks.values[i] for i in earlyPeaks], times=[peaks.times[i] for i in earlyPeaks])
        return peaks

    def getSalientPoints(self, salienceSignals, grid, peaks):
        """
        split peaks signal into two: 
        Salient points, and non-salient points
        """
        if self.parameters["salienceTreshold"].value:
            salience = [
                self.getSalience(pos, salienceSignals, grid, self.parameters["salienceWindow"].value) for pos in peaks.times
            ]
            salientPoints = [i for i, v in enumerate(salience) if v >= self.parameters["salienceTreshold"].value]
            nonSalientPoints = [i for i, v in enumerate(salience) if v < self.parameters["salienceTreshold"].value]

            # if there is no point above the threshold of salience, just return the most salient one
            if len(salientPoints) == 0 and len(salience) > 0:
                salientPoints = [np.argmax(salience)]
                nonSalientPoints = [p for p in nonSalientPoints if p not in salientPoints] 
            
            nonSalient = Signal([peaks.values[i] for i in nonSalientPoints],
                                times=[peaks.times[i] for i in nonSalientPoints],
                                sparse=True)
            peaks = Signal([peaks.values[i] for i in salientPoints], times=[peaks.times[i] for i in salientPoints])
            return peaks, nonSalient
        else:
            return peaks, Signal([], times=[])

    def getSalience(self, point, features: List[Signal], grid: Signal, window):
        """
        Return a salience of the window following the point
        """
        score = 0
        for feature in features:
            try:
                amount = feature.getValues(point, grid.getTime(grid.getIndex(point) + window))
            except IndexError as e:
                amount = [0]  #TODO sometimes the posiiton is beyond the grid ?

            score += np.mean(amount) if len(amount) else 0
        return score / len(features) if len(features) != 0 else 0
