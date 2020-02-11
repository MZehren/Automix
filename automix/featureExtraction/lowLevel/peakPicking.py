"""
Get the peaks exceding a threshold. Returns only the maximum value in the threshold exceed
"""

from typing import List

import numpy as np
from scipy.ndimage import filters

from automix.featureExtraction.estimator import Estimator, Parameter
from automix.model.classes.signal import Signal


class PeakPicking(Estimator):
    def __init__(self,
                 parameterMedianSize=16,
                 parameterRelativeThreshold=0.5,
                 parameterThresholdIndex=1,
                 parameterMinDistance=0,
                 inputSignal="barMSE",
                 outputPeaks="peaks",
                 cachingLevel=2,
                 forceRefreshCache=False):
        """
        Estimator computing returning peaks from a signal.
        Based on the method from MAXIMUM FILTER VIBRATO SUPPRESSION FOR ONSET DETECTION 2013 

        Parameters
        ----------
        parameterRelativeThreshold (optional float):
            return the maximum value of a continuous window exceeding the threshold in % of the max value

        parameterThresholdIndex
            Limit the search of the maximum value in this part of the signal (1 = 100% of the track, 0.5 = 50% from the start of 
            the track)

        parameterMinDistance
            return the highest peaks in a window of this size, 
            This value filter peaks within distance striclty inferior.
            (If min distance is set to 8 ticks, two peaks 8 ticks appart can be return) 
            
        parameterMedianSize
            When computing peaks without a static threshold


        windowSize (optional int):


        distance (optional int)
            min distance in indexes between two peaks
        """
        self.parameters = {
            "medianSize": Parameter(parameterMedianSize),
            "relativeThreshold": Parameter(parameterRelativeThreshold),
            "thresholdIndex": Parameter(parameterThresholdIndex),
            "minDistance": Parameter(parameterMinDistance)
        }
        self.inputs = [inputSignal]
        self.outputs = [outputPeaks]
        self.cachingLevel = cachingLevel
        self.forceRefreshCache = forceRefreshCache

    def predictOne(self, values: Signal):
        listV = np.array(values.values)
        if self.parameters["relativeThreshold"].value:
            # compute the thrshold at x times the maximum value
            threshold = np.max(
                listV[:int(len(listV) * self.parameters["thresholdIndex"].value)]) * self.parameters["relativeThreshold"].value
            peaks, peaksValues = self.staticThreshold(listV, threshold, self.parameters["minDistance"].value)
        else:
            peaks, peaksValues = self.adaptiveThreshold(listV, L=self.parameters["medianSize"].value)

        result = Signal(peaksValues, times=[values.times[peak] for peak in peaks], sparse=True)
        return (result, )

    def staticThreshold(self, values: List, threshold: float, minDistance: int):
        """
        get the peaks in the list of values. return a list of idx and a list of values

        pk = PeakPicking(minDistance=2, threshold=5)
        print(pk.predictOne([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])) -> ([9], [10])
        print(pk.predictOne([10, 2, 3, 4, 5, 6, 7, 8, 9, 10])) -> ([0,9], [10,10])

        Parameters:
            signal (Signal): Signal to extract peak from

        Returns:
            boundaries (List<int>): The index of the peaks

            values (List<T>): The value of the peaks
        """
        peaksValue = []
        peaksPosition = []
        mySortedList = sorted([(i, value) for i, value in enumerate(values)], key=lambda x: x[1], reverse=True)
        for i, value in mySortedList:
            # Check if the value is > threshold
            if value >= threshold:
                # Check if it's the maximum in a window size
                # TODO: simplify the implementation like meanThreshold
                isMaximum = value == np.max(values[max(i - minDistance, 0):i + minDistance + 1])
                if isMaximum:
                    peaksValue.append(value)
                    peaksPosition.append(i)
            else:
                break

        return peaksPosition, peaksValue

    def meanThreshold(self, values: List, threshold: float, minDistance: int):
        """
        get the peaks in the list of values. return a list of idx and a list of values

        pk = PeakPicking(minDistance=2, threshold=5)
        print(pk.predictOne([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])) -> ([9], [10])
        print(pk.predictOne([10, 2, 3, 4, 5, 6, 7, 8, 9, 10])) -> ([0,9], [10,10])

        Parameters:
            signal (Signal): Signal to extract peak from

        Returns:
            boundaries (List<int>): The index of the peaks

            values (List<T>): The value of the peaks
        """
        peaksValue = []
        peaksPosition = []
        lookdistance = 1
        mySortedList = sorted([(i, value) for i, value in enumerate(values)], key=lambda x: x[1], reverse=True)
        for i, value in mySortedList:  #For all the values by decreasing order TODO: implement a different distance to the peak ?
            if value >= threshold:
                isMaximum = value == np.max(values[max(i - minDistance, 0):i + minDistance + 1])
                isAboveMean = value >= np.mean(values[max(i - minDistance, 0):i + minDistance + 1]) + threshold
                if isMaximum and isAboveMean:
                    peaksValue.append(value)
                    peaksPosition.append(i)
            else:
                break

        return peaksPosition, peaksValue

    def adaptiveThreshold(self, nc: List, L=16):
        """
        Obtain peaks from a novelty curve using an adaptive threshold.
        Foote 2000's method, implementation by msaf
        """
        offset = nc.mean() / 20.

        smooth_nc = filters.gaussian_filter1d(nc, sigma=4)  # Smooth out nc

        th = filters.median_filter(smooth_nc, size=L) + offset
        # th = filters.gaussian_filter(nc, sigma=L/2., mode="nearest") + offset

        peaks = []
        for i in range(1, smooth_nc.shape[0] - 1):
            # is it a peak?
            if smooth_nc[i - 1] < smooth_nc[i] and smooth_nc[i] > smooth_nc[i + 1]:
                # is it above the threshold?
                if smooth_nc[i] > th[i]:
                    peaks.append(i)

        # plt.plot(old)
        # plt.plot(nc)
        # plt.plot(th)
        # for peak in peaks:
        #     plt.axvline(peak)
        # plt.show()

        return peaks, [nc[peak] for peak in peaks]



# TODO: Implement the aggregation of the peaks from multiple features to multiply.
# aggregation:
#     independant: tells if the selected peaks are from all the features independently
#     multiply: The peaks values are multiplied together and works only when the features agree with each others

# newCurve = np.ones(len(track.features[features[0]].values))
# for feature in features:
#     newCurve = np.multiply(newCurve, track.features[feature].values)
# peakSignals = pp.predictOne(Signal(newCurve, times=track.features[features[0]].times))
# newCues = peakSignals[0].times
