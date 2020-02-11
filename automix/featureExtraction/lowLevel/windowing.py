"""
container for the bandSpectrogram estimator
"""
import copy
from typing import List

import librosa
import numpy as np
from essentia import standard

from automix.featureExtraction.estimator import Estimator, Parameter
from automix.model.classes.signal import Signal
from automix.utils import normalize

# TODO Is it windowing if we are computing only one value per window ? Change the name
class Windowing(Estimator):
    def __init__(self,
                 parameterWindow="rectangular",
                 parameterAggregation="rmse",
                 parameterSteps=1,
                 parameterPanning=0,
                 inputSamples="samples",
                 inputGrid="downbeats",
                 output="RMSE",
                 cachingLevel=0,
                 forceRefreshCache=False):
        """
        Create a window of the input signal at each grid tick

        parameterWindow: the name of the window function to apply:
        - rectangular: Only window currently implemented

        parameterAggregation: What to do to the values in the window
        - None
        - rmse
        - sum

        parameterPanning: How much do you shift, in ratio of the median distance between the grid ticks, the windows boundaries.
        Use a neagtive value (ie -0.25) to shift the windows 1/4 of the grid to the left.

        parameterSteps: TODO
        """
        # parameterLength: TODO implement
        #  parameterBands=[[20, 250], [250, 3000], [3000, 22000]],
        self.parameters = {
            "window": Parameter(parameterWindow),
            "aggregation": Parameter(parameterAggregation),
            "steps": Parameter(parameterSteps),
            "panning": Parameter(parameterPanning)
        }
        self.inputs = [inputSamples, inputGrid]
        self.outputs = [output]
        self.cachingLevel = cachingLevel
        self.forceRefreshCache = forceRefreshCache

    def predictOne(self, signal: Signal, grid: Signal):
        """
        Returns the amplitude or RMSE of the signal between grid ticks
        """
        # get the bands. Should I use librosa.fft_frequencies() instead ?
        # if self.parameters["bands"].value:
        #     y = np.array(samples.values)
        #     sr = samples.sampleRate
        #     n_fft = 2048  # default value
        #     srSTFT = sr / (n_fft / 4)  # Sample rate of the STFT
        #     d = librosa.stft(y, n_fft=n_fft)
        #     # Perception correction
        #     perceivedD = librosa.perceptual_weighting(np.abs(d)**2, librosa.ffharmonicRMSEt_frequencies(sr=sr, n_fft=n_fft))
        #     # perceivedD = np.abs(d)

        #     dBands = [[
        #         np.mean(frame[int(band[0] * len(frame) / (sr / 2)):int(band[1] * len(frame) / (sr / 2))])
        #         for frame in np.ndarray.transpose(perceivedD)
        #     ] for band in self.parameters["bands"].value]

        #     # normalize based on the highest and lowest values accross all frequencies.
        #     # then, because of the perception correction, and the replayGain normalization,
        #     # the highest peak (amplitude of one) should be the same accross all frequencies
        #     dBands = normalize(dBands)

        #     rparameterStepsT, grid, square=False) for band in dBands], )  # barBandMSE

        # else:parameterSteps
        if self.parameters["steps"].value > 1:
            grid = self._subdivide(grid, self.parameters["steps"].value)
        return (self._getWindows(signal,
                                 grid,
                                 window=self.parameters["window"].value,
                                 aggregation=self.parameters["aggregation"].value), )

    def _subdivide(self, grid, steps):
        newTimes = []
        for i in range(len(grid.times) - 1):
            newTimes = np.concatenate(
                (newTimes, np.arange(grid.times[i], grid.times[i + 1], (grid.times[i + 1] - grid.times[i]) / steps)))

        newTimes = np.concatenate((newTimes, [grid.times[-1]]))  # TODO: clean that

        return Signal(np.ones(len(newTimes)), times=newTimes)

    def _getWindows(self,
                    signal: Signal,
                    grid: Signal,
                    addAnacrusis=False,
                    addAfterLastBeat=False,
                    window="square",
                    aggregation='rmse'):
        """
        Get the root mean square amplitude between each tick of the grid (in seconds).
        addAnacrusis add also the energy from the first sample in the signal to the first tick of the grid,
        and the last tick of the grid to the last sample of the signal.
        return eg [0.1,0.2,0.1,0.2,0.8,0.9,0.8,0.9]
        """

        result = []
        times = copy.copy(grid.times)
        # pan times
        panning = self.parameters["panning"].value * np.median(np.diff(times))
        times = [time - panning for time in times]
        # if addAnacrusis:
        #     times = np.insert(times, 0, 0)  # TODO make it faster by not creating a new array
        #     annacrusisValues = signal.getValues(0, times[])
        #     if len(annacrusisValues):
        #         result.append(self._getWindow(annacrusisValues, window, aggregation))
        #     else:  # If the first tick is at 0, then the anacrusis is 0, or [0 ,..., 0] if the signal is multidimensional
        #         result.append(signal.values[0] * 0.)

        for i in range(len(grid) - 1):
            result.append(self._getWindow(signal.getValues(times[i], times[i + 1]), signal.sampleRate, window, aggregation))

        # if addAfterLastBeat:
        #     afterValues = signal.getValues(grid.times[-1], signal.duration)
        #     if len(afterValues):
        #         result.append(self._getWindow(afterValues, window, aggregation))
        #     else:
        #         result.append(signal.values[0] * 0.)
        # else:
        #     times = times[:-1]

        return Signal(result, times=grid.times[:-1])

    def _getWindow(self, signal, sr, window, aggregation):
        """
        do the aggregation of the samples inside the windo
        """
        if aggregation == "rmse":
            return np.sqrt(np.mean(np.square(signal), axis=0))

        elif aggregation == "sum":
            return np.sum(signal, axis=0)

        elif aggregation == "replayGain":
            return standard.ReplayGain(sampleRate=sr)(signal)

        return signal
