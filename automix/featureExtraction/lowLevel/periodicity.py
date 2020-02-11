"""
container for the Periodicity estimator
"""
from typing import List

import librosa
import numpy as np

from automix.featureExtraction.estimator import Estimator, Parameter
from automix.model.classes.signal import Signal


class Periodicity(Estimator):
    """
    Estimator infering the periodicity of a track

    inputFeatures: List of signals
        The signals can be sparse or dense. The amplitude and the number of values on the period are taken into account.
    
    outputPeriod: A sparse signal with a value at each point on the period.

    paraeterDistanceMetric: compute how the values for the peaks is determined
        RMS, SUM, Veire=SUM+Mult

    parameterFeatureAggregation: how the feature are aggregated
        qualitatively = by counting the number of features in agreement
        Quantitatively = by summing the score of each feature
    """

    def __init__(self,
                 inputFeatures=["cqtAmplitudeCheckerboard"],
                 inputGrid="strongBeats",
                 outputPeriod="period",
                 parameterDistanceMetric="RMS",
                 parameterFeatureAggregation="quantitative",
                 parameterPeriod=2,
                 cachingLevel=2,
                 forceRefreshCache=True): #As long as there is no way of updating the cache when the input changes
        self.inputs = [inputFeatures, inputGrid]
        self.outputs = [outputPeriod]
        self.parameters = {"period": Parameter(parameterPeriod), "distanceMetric": Parameter(parameterDistanceMetric), "featureAggregation": Parameter(parameterFeatureAggregation)}
        self.cachingLevel = cachingLevel
        self.forceRefreshCache = forceRefreshCache

    def predictOne(self, inputFeatures: List[Signal], inputGrid: Signal):
        # for period in self.parameters["period"].value:
        period = self.parameters["period"].value
        phase = self.getPhase(period, inputFeatures, inputGrid)

        return (Signal(inputGrid.values[phase::period], times=inputGrid.times[phase::period], sparse=True), )

    def getPhase(self, period, features, inputGrid):
        """
        Get the phase of the track depending on all the features specified and the period
        TODO: The phase should be computed with all the features combined. not with all the features independent
        """
        # Equal weight per feature
        if self.parameters["featureAggregation"].value == "quantitative":
            phasePerFeature = []
            for feature in features:
                bestPhase = np.argmax(self.findPhaseLocal(period, feature, inputGrid))
                phasePerFeature.append(bestPhase)
            counts = np.bincount(phasePerFeature)
            quantitative = np.argmax(counts)
            return quantitative
        elif self.parameters["featureAggregation"].value == "qualitative":
            # Equal weight maybe ? but qualitative value per phase for each feature
            overalScore = np.zeros(period)
            for feature in features:
                score = self.findPhaseLocal(period, feature, inputGrid)
                overalScore = np.add(score, overalScore)
            qualitative = np.argmax(overalScore)
            return qualitative
        else:
            raise Exception("bad feature aggregation parameter")

        # different weight per feature
        # binValues = []
        # for phase in range(period):
        #     binValues.append([])
        #     for feature in features:
        #         binValues[phase] = [feature.getValue(inputGrid.times[i]) for i in range(phase, len(inputGrid), period)]
        #         binValues[phase] = [v for v in binValues[phase] if v is not None]
        # # Veire's method. the best candidate is maximizing the number of peaks in phase AND the amplitude of the peaks
        # binProduct = [np.sum(values) * len(values) for values in binValues]
        # return np.argmax(binProduct)

    def findPhaseLocal(self, period: int, signal: Signal, grid: Signal, toleranceWindow=0.1):
        """
        find the phase of the signal based on it's amplitude at the grid positions and the number of peaks
        - signal: works best with a discrete signal as no aglomeration is done
        - grid: positions of the beats
        - period: the periodicity to test
        - tolerance window: if not at 0, returns the closest value in the signal to the grid, within the tolerance window

        test:
        # result = findPhase(Signal(np.ones(5), times=np.array([0, 4, 8, 9, 12])+1), Signal(np.ones(16), times=range(16)), 
            period=4)
        # print(result) = 1
        """
        phases = []
        for phase in range(period):
            values = [signal.getValue(grid.times[i], toleranceWindow=toleranceWindow) for i in range(phase, len(grid), period)]
            values = [v for v in values if v is not None]
            if self.parameters["distanceMetric"].value == "RMS":
                value = np.sqrt(np.mean(np.array(values)**2))
            elif self.parameters["distanceMetric"].value == "sum":
                value = np.sum(values)
            elif self.parameters["distanceMetric"].value == "Veire":
                value = np.sum(values) * len(values)
            else:
                raise Exception("Bad distance metric parameter" + self.parameters["distanceMetric"].value )
            phases.append(value)

        # bestPhase = np.argmax(phases)
        return phases


# p = Periodicity(parameterPeriod=4)
# print(p.predictOne([Signal(1, times=[5, 9, 14]), Signal(1, times=[6, 10])], Signal(1, times=range(30)))[0].times)
# print(p.predictOne([Signal(1, times=[5, 9, 6, 10, 14])], Signal(1, times=range(30)))[0].times)
