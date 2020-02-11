"""
Generate the structure of the song with the mean square energy (MSE)
"""

import numpy as np

from automix.featureExtraction.estimator import Estimator, Parameter
from automix.model.classes.segment import Segment
from automix.model.classes.signal import Signal

from automix.featureExtraction.structure.eval import evalStructureMIR, segmentsToMirEval


class SalientPointDetection(Estimator):
    # TODO: covert the low threshold in percentile instead of absolute value
    def __init__(self,
                 parameterDiffThreshold=5,
                 parameterRatioThreshold=0,
                 parameterKMax=0,
                 parameterzeroThreshold=0.000001,
                 inputSignal="barMSE",
                 outputBoundaries="boundaries",
                 outputLabels="labels",
                 outputOnsets="onsets",
                 cachingLevel=0,
                 forceRefreshCache=False):
        """
        Estimator computing the structure of a track based on salient points in the input signal
        the input signal should not be too noisy. It works best when the input signal is already averaged per beats or per downbeats 

        Parameters
        ----------
        diffThreshold (optional float):
            threshold setting the difference between the previous sample and the current one to label it as a segment.
            this difference is computed in term of diffThreshold * (mean difference between each sample)

        kMax (optional int):
            If you don't want to use a factor for the saillance detection, you can specify to return the k-most saillant points.

        ratioThreshold (optional float):
            thresold setting the ratio difference between two point to consider them saillant

        zeroThreshold (float):
            Indicating the threshold under which an amplitude should be considered as being zero.
            it's usefull to set the first segment where the difference between the samples is not big because the signal is rizing from silence.
        """
        self.parameters = {
            "diffThreshold": Parameter(parameterDiffThreshold),
            "ratioThreshold": Parameter(parameterRatioThreshold),
            "kMax": Parameter(parameterKMax),
            "zeroThreshold": Parameter(parameterzeroThreshold)
        }
        raise DeprecationWarning()
        self.inputs = [inputSignal]
        self.outputs = [outputBoundaries, outputLabels, outputOnsets]
        self.cachingLevel = cachingLevel
        self.forceRefreshCache = forceRefreshCache

    def _getAutomaticThreshold(self, signal, bins=10):
        """based on the method in https://ant-s4.unibw-hamburg.de/dafx/papers/
        DAFX02_Duxbury_Sandler_Davis_note_onset_detection.pdf"""
        onsets, absOnsets = self._getOnsets(signal)
        count = np.histogram(absOnsets, bins=np.linspace(min(absOnsets), max(absOnsets), bins))
        t = 1
        dCount = [count[0][i + t] - count[0][i] / t for i in range(len(count[0]) - t)]
        ddCount = [dCount[i + t] - dCount[i] / t for i in range(len(dCount) - t)]
        threshold = count[1][np.argmax(ddCount) + 2 * t]
        return self._getBoundaries(signal, absOnsets, threshold, self.parameters["zeroThreshold"].value), onsets

    def _getDiffThresholdBoundaries(self, signal):
        onsets, absOnsets = self._getOnsets(signal)
        threshold = self.parameters["diffThreshold"].value * np.mean(absOnsets)
        return self._getBoundaries(signal, absOnsets, threshold, self.parameters["zeroThreshold"].value), onsets

    def _getKMaxBoundaries(self, signal):
        onsets, absOnsets = self._getOnsets(signal)
        threshold = list(sorted(absOnsets))[-self.parameters["kMax"].value]
        return self._getBoundaries(signal, absOnsets, threshold, self.parameters["zeroThreshold"].value), onsets

    def _getOnsets(self, signal):
        """
        Returns a first order difference of the signal and the absolute first order difference

        """
        diff = np.diff(signal)
        return Signal(diff, times=signal.getTimes()[1:]), Signal(np.abs(diff), times=signal.getTimes()[1:])

    def _getBoundaries(self, signal, onsets, threshold, zeroThreshold):
        """
        Return the points with an onset above the threshold or samples after a zero amplitude sample.
        """
        # i+1 because the onsets are shifted to the left relative to the signal
        return [
            i + 1 for i, diff in enumerate(onsets)
            if diff >= threshold or (signal[i] < zeroThreshold and signal[i + 1] > self.parameters["zeroThreshold"].value)
        ]

    def _getRatioThresholdBoundaries(self, signal):
        onsets = Signal([signal[i + 1] / signal[i] if signal[i] != 0 else 10000 for i in range(len(signal) - 1)],
                        times=signal.getTimes()[1:])
        incTH = self.parameters["ratioThreshold"].value
        decTH = 1. / incTH
        return [i + 1 for i, ratio in enumerate(onsets) if ratio >= incTH or ratio <= decTH], onsets

    def predictOne(self, signal):
        """
        get the structure from the saillant points

        Parameters:
            signal (Signal): Signal to create segments from
        """
        # get the structure from the boundaries and the phase.
        if self.parameters["kMax"].value:
            sailantIndexes, onsets = self._getKMaxBoundaries(signal)
        elif self.parameters["diffThreshold"].value:
            sailantIndexes, onsets = self._getDiffThresholdBoundaries(signal)
        elif self.parameters["ratioThreshold"].value:
            sailantIndexes, onsets = self._getRatioThresholdBoundaries(signal)
        else:
            sailantIndexes, onsets = self._getAutomaticBoundaries(signal)

        # appending the last elements as the end of the signal
        sailantIndexes.append(len(signal) - 1)

        # quantize the boundaries to the closest loop grid from the rising
        # boundaries.
        # phase = 0
        # if self.parameters["loopLength"].value != -1:
        #     phase = (np.argmax([
        #         np.sum([
        #             absMSADiff[i]
        #             for i in range(j, len(absMSADiff), self.parameters["loopLength"].value)
        #         ]) for j in range(self.parameters["loopLength"].value)
        #     ]) + 1) % self.parameters["loopLength"].value

        #     boundaries = quantization.quantize(
        #         range(phase, len(grid), self.parameters["loopLength"].value), boundaries)

        #     phase = 3 if phase == 0 else phase - 1

        # convert the boundaries to the real times
        # Because of the anacrusis which is not present in the grid, the
        # boundaries indexes are shifted by one
        # [grid[i - 1] for i in boundaries]
        boundaries = signal.getTimes(sailantIndexes)
        # same with the phase which goes from 0 to three

        # Add the labels
        segmentedSignal = [signal.values[sailantIndexes[i]:sailantIndexes[i + 1]] for i in range(len(sailantIndexes) - 1)]
        labels = ["Start"] + [
            Segment.getLabel(segment, np.mean(signal.values), i, len(segmentedSignal))
            for i, segment in enumerate(segmentedSignal)
        ]

        # segments = [
        #     Segment(
        #         labels[i],
        #         start=boundariesTime[i],
        #         barStart=boundaries[i],
        #         end=boundariesTime[i + 1],
        #         barEnd=boundaries[i + 1],
        #         duration=boundariesTime[i + 1] - boundariesTime[i],
        #         barDuration=boundaries[i + 1] - boundaries[i],
        #         onsetIntensity=msaDiff[boundaries[i] - 1] / max(msaDiff))
        #     for i in range(len(boundaries) - 1)
        # ]
        # TODO: why not use that : boundaries = Signal(onsetIntensity, times=[boundariesTime])

        # loopBoundaries = [grid[i] for i in range(phase, len(grid), self.parameters["loopLength"].value)]
        return (boundaries, labels, onsets)

    def evaluate(self, X, y):
        y_ = [segment.start for segment in self.predict(X)]
        return [evalStructureMIR(segmentsToMirEval(y_[i]), segmentsToMirEval(y[i])) for i in range(len(X))]
