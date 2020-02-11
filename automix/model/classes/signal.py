"""
container for the Signal class
"""

import collections
import numbers
from bisect import bisect_left
from typing import List, Tuple

import numpy as np
from scipy.stats import mode


class Signal(collections.Iterable):
    '''
    Class used as a data structure associated with a signal
    the values have to be associated with either a sanpleRate or an array of positions which has to be monotonly increasing

    Attributes:
        values (list<float>): the amplitude of the signal at each time step.
        sampleRate (float, optional): the sample rate of the signal
        times (list<float>, optional): the positions of each sample

        If values is a scalar and times is set, an array of values equal to the scalar is created
    '''

    def __init__(self,
                 values: np.array,
                 sampleRate: float = None,
                 times: List[float] = None,
                 duration: float = -1,
                 sparse: bool = False):

        if isinstance(values, numbers.Number) and times is not None:
            self.values = np.ones(len(times)) * values
        else:
            self.values = values

        self.sparse = sparse
        self.sampleRate = sampleRate

        if times is not None and len(times):
            self.values = [x for _, x in sorted(zip(times, self.values))]
            times.sort()
        self.times = times

        self.duration = duration
        if sampleRate is not None and duration == -1:
            self.duration = float(len(values)) / sampleRate
        elif times is not None and len(times) and duration == -1:
            self.duration = times[-1]

    @property
    def values(self):
        return self._values

    @values.setter
    def values(self, values):
        """
        Force the values to be a numpy array
        """
        self._values = values if isinstance(values, np.ndarray) else np.array(values)

    @property
    def times(self):
        """
        return the times of all the values
        or compute it if only the sample rate is available
        """

        if self._times is None and self.sampleRate:
            self._times = np.array([i / self.sampleRate for i, _ in enumerate(self.values)])

        return self._times

    @times.setter
    def times(self, times):
        """
        Force the times to be a numpy array
        """
        if times is not None:
            self._times = times if isinstance(times, np.ndarray) else np.array(times)
            assert len(self._times) == len(self._values)
            for i in range(len(times) - 2):
                assert times[i] <= times[i + 1]
        else:
            self._times = None

    def __iter__(self):
        # return (x for x in list.__iter__(self.values) if x is not None)
        return self.values.__iter__()

    def __len__(self):
        return len(self.values)

    def __getitem__(self, i):
        return self.values[i]

    def addSignal(self, signal):
        """
        Concatenate another signal to this one, while taking care of the timesteps
        """
        snapDistance = self.getTimeStep()
        for i in range(len(signal)):
            value = signal.values[i]
            position = signal.getTime(i)
            localIndex = self.getIndex(position, toleranceWindow=snapDistance)

            if localIndex:  # The value is located close to an existing value in this signal
                self.values[localIndex] += value
            else:  # The value is located far from an existing value in this signal
                self.insert(value, position)

    def insert(self, value, time):
        """
        Insert a value at a specific time
        """
        closestIndex = self.getIndex(time)
        if closestIndex is None:
            newIndex = 0
        else:
            newIndex = closestIndex if time < self.getTime(closestIndex) else closestIndex + 1

        if len(self.times) == len(self.values):
            self.values.insert(newIndex, value)
            self.times.insert(newIndex, time)
        elif self.sampleRate:
            raise NotImplementedError()

    def getTimeStep(self):
        """
        return the time between two ticks of the signal
        """
        if self.sampleRate:
            return (1 / self.sampleRate)
        elif len(self.times) > 1:
            return mode(np.diff(self.times))[0][0]
        else:
            return 0

    def getTime(self, index):
        """
        Returns the position in seconds of the index provided

        Parameter:
            index(integer)
        """
        # raise DeprecationWarning()
        if self.sampleRate is not None:
            return index / self.sampleRate
        else:
            return self.times[index]

    def getIndexes(self, times, toleranceWindow=0.1):
        """
        See getIndex

        Parameters:
            time (List<float>): The time from which we want the closest index
            toleranceWindow (float): the tolerance window (it's useful for float error)
        """
        return [self.getIndex(i, toleranceWindow=toleranceWindow) for i in times]

    def takeClosest(self, myList, myNumber):
        """
        Assumes myList is sorted. Returns closest index to myNumber.

        If two numbers are equally close, return the smallest number.

        TODO: save a set of times to retreive automatically the value
        """
        pos = bisect_left(myList, myNumber)
        if pos == 0:
            return 0
        if pos == len(myList):
            return len(myList) - 1
        before = myList[pos - 1]
        after = myList[pos]
        if after - myNumber < myNumber - before:
            return pos
        else:
            return pos - 1

    def getIndex(self, time, toleranceWindow=None):
        """
        Returns the closest index to the time specified
        If the time difference to the closest beat is more than the tolerance Window, then return None.

        Parameters:
            time (float): The time from which we want the closest index
            toleranceWindow (float): the tolerance window (it's useful for float error)
        """
        if self.sampleRate:
            closestIndex = round(self.sampleRate * time)
            if toleranceWindow is None or abs(closestIndex / self.sampleRate - time) <= toleranceWindow:
                return int(closestIndex)

        elif self.times is not None and len(self.times):
            # raise NotImplementedError()
            # closestIndex = np.argmin([abs(localTime - time) for localTime in self.times])
            closestIndex = self.takeClosest(self.times, time)  # TODO, times has to be sorted, check that
            if toleranceWindow is None or abs(self.times[closestIndex] - time) <= toleranceWindow:
                return closestIndex

    def getValues(self, startPosition, stopPosition, translation=0):
        """
        returns the values between exactly the positions specified
        if translation is set, translate the boundaries toward the translation direction to compensate
        the imprecision of the signal.
        """
        if self.sampleRate:
            indexA = self.getIndex(startPosition)
            indexB = self.getIndex(stopPosition)
        else:
            try:
                indexA = [i for i, t in enumerate(self.times) if t >= startPosition + translation][0]
                indexB = [i for i, t in enumerate(self.times) if t < stopPosition + translation][-1] + 1
            except IndexError:
                return []

        if indexA is not None and indexB is not None:
            return self.values[indexA:indexB]
        else:
            return []

    def getValue(self, position, toleranceWindow=None):
        """
        returns the value at a specific position in seconds
        if toleranceWindow = None, it is considered infinite
        """
        index = self.getIndex(position, toleranceWindow=toleranceWindow)
        if index is not None:
            return self.values[index]
        else:
            return None

    def jsonEncode(self):
        """
        json encoding
        """
        return self.__dict__

    def quantizeTo(self,
                   grid: 'Signal',
                   maxThreshold=-1,
                   removeOutOfBound=True,
                   removeDuplicatedValues=True,
                   mergeDuplicate=np.sum):
        """
        Align values to a grid
        Parameters:
            maxThreshold specify the maximum distance between the value and the closest tick in the grid.
                by default it'll take half of the most common difference in the grid
            removeOutOfBound: if there is no tick near the value AND removeOutOfBound, the value is removed.
                if removeOutOfBound is False, then the original value is kept
            removeDuplicatedValues: If two values are quantized to the same grid tick, it's going to get removed

        # test
        grid = Signal(1, times=[1, 2, 3])

        values = Signal(1, times=[0, 1, 1.4, 3])
        values.quantizeTo(grid)
        print(values.values, values.times) -> [2. 1.] [1. 3.]

        values = Signal([1, 2, 3, 4, 5], sampleRate=2)
        values.quantizeTo(grid)
        print(values.values, values.times) ->[3 5] [1. 2.]
        """
        # computes the max threshold if not specified
        if maxThreshold == -1:
            if len(grid) > 1:
                maxThreshold = float(mode(np.diff(grid.times))[0][0]) / 2
            else:
                maxThreshold = 0

        # Change the position of all the values and keep track of the one too far away
        removeIdx = set()
        for i, value in enumerate(self.values):
            closestGridTick = grid.getTime(grid.getIndex(self.times[i]))
            if np.abs(self.times[i] - closestGridTick) <= maxThreshold:
                # The distance between the point and the grid is close enough
                self.times[i] = closestGridTick
            else:
                # The distance between the point and the grid is too far
                if removeOutOfBound:
                    removeIdx.add(i)

        # Remove the point too far away
        if removeIdx:
            self.values = [value for i, value in enumerate(self.values) if i not in removeIdx]
            self.times = [time for i, time in enumerate(self.times) if i not in removeIdx]
            self.sampleRate = None

        # Remove the points which are duplicated
        if removeDuplicatedValues:
            count = collections.Counter(self.times)
            newTimes = self.times
            for time, occurences in count.items():
                if occurences > 1:
                    occurencesIdx = [i for i, x in enumerate(newTimes) if x == time]
                    self.values[occurencesIdx[0]] = mergeDuplicate([self.values[i] for i in occurencesIdx])
                    self.values = [value for i, value in enumerate(self.values)
                                   if i not in occurencesIdx[1:]]  # TODO change the performances here
                    newTimes = [time for i, time in enumerate(newTimes) if i not in occurencesIdx[1:]]
            self.times = newTimes

    def plot(self, show=False, label=None, maxSamples=1000, color=None, asVerticalLine=False):
        """
        Plot the signal as a curve or as vertical lines with the asLine
        """
        import matplotlib.pyplot as plt

        if asVerticalLine:
            for i, line in enumerate(self.times[:maxSamples]):
                if i != 0:
                    label = None
                plt.axvline(line, alpha=0.5, color=color, label=label)

            plt.xticks(self.times, [str(int(xc / 60)) + ":" + str(np.round(xc % 60, 2)) for xc in self.times[:maxSamples]],
                       rotation="vertical")
        elif self.sparse:
            plt.scatter(self.times[:maxSamples], self.values[:maxSamples], label=label, color=color)
        elif len(self.values.shape) == 2:
            extent = [0, self.times[-1], 1, 2]
            plt.matshow(self.values[:maxSamples].T, extent=extent, aspect='auto', fignum=1)
        else:
            step = int(np.ceil(len(self.values) / maxSamples))
            plt.plot(self.times[::step], self.values[::step], label=label, color=color)

        if show:
            plt.legend()
            plt.show()

    def sonification(self, path="test.wav", sampleRate=22050):
        """
        Create a wave file with click at the location of the times of this signal
        """
        from scipy.io import wavfile
        if self.sparse is False or self.times is None:
            import warnings
            warnings.warn("the Signal might not produce a good sonication (needs to be sparse and have times")

        def getClick(clicks, fs, frequency=1000, offset=0, volume=1, length=0):
            """
            # Generate clicks (this should be done by mir_eval, but its latest release is not compatible with latest numpy)
            """
            import mir_eval
            times = np.array(clicks) + offset
            # 1 kHz tone, 100ms with  Exponential decay
            click = np.sin(2 * np.pi * np.arange(fs * .1) * frequency / (1. * fs)) * volume
            click *= np.exp(-np.arange(fs * .1) / (fs * .01))
            if not length:
                length = int(times.max() * fs + click.shape[0] + 1)
            return mir_eval.sonify.clicks(times, fs, click=click, length=length)

        # Assign the audio and the clicks
        audioClicks = getClick(self.times, sampleRate, frequency=1500)
        # Write to file
        wavfile.write(path, sampleRate, audioClicks)

    @staticmethod
    def clusterSignals(signals, minDistance=0.5, mergeValue=np.sum, mergeTime=np.median):
        """
        instantiate a new Signal by merging and clustering all the signal in input
        TODO: duplicate from signal.addSignal ?
        TODO: Duplicate from jams.deserialize ? 
        """
        # Get all the values and times in ascending order
        times = np.concatenate([signal.times for signal in signals])
        values = np.concatenate([signal.values for signal in signals])
        values = [x for _, x in sorted(zip(times, values))]
        times.sort()

        # Very naive clustering based on distance by iterating samples
        # TODO: use the meanshift ?
        # TODO: make it faster?
        i = 0
        newTimes = []
        newValues = []
        while i < len(times):  # for each index
            j = i + 1  # we look at all following indexes until the distance is above the threshold: This is our cluster
            for j in range(i + 1, len(times) + 1):
                if j == len(times) or np.abs(times[j - 1] - times[j]) > minDistance:
                    break
            newTimes.append(mergeTime(times[i:j]))
            newValues.append(mergeValue(values[i:j]))
            i = j

        return Signal(newValues, times=newTimes)

    @staticmethod
    def jsonDeserialize(jsonObject):
        """
        json decoding
        """
        signal = Signal([])
        for attribute, value in jsonObject.items():
            if hasattr(signal, attribute):
                if type(value) == list:
                    setattr(signal, attribute, np.array(value))
                else:
                    setattr(signal, attribute, value)
        return signal

    # @staticmethod
    # def buildFromSignal(signal):
    #     result = Signal()


class DenseSignal(Signal):
    def __init__(self, values, sampleRate):
        """
        Represents a dense signal. It is not constructed with an array of time position but a samplerate
        """
        super().__init__(values, sampleRate=sampleRate)


class SparseSignal(Signal):
    def __init__(self, values, times):
        """
        Represents a sparse signal where the values are located at specific time locations
        """
        super().__init__(values, times=times, sparse=True)


class SparseSegmentSignal(Signal):
    def __init__(self, values, timeIntervals: List[Tuple[float, float]]):
        """
        Represents a list of segments where the values are located at specific time intervals (list of tuples)
        """
        super().__init__(values, times=timeIntervals, sparse=True)

    def getIndex(self, time, toleranceWindow=None):
        """
        Returns the closest index to the time specified
        If the time difference to the closest beat is more than the tolerance Window, then return None.

        Parameters:
            time (float): The time from which we want the closest index
            toleranceWindow (float): the tolerance window (it's useful for float error)
        """
        # TODO implement toleranceWindow
        if toleranceWindow is None:
            toleranceWindow = 0

        candidates = []
        for i, startStop in enumerate(self.times):
            start, stop = startStop
            if time >= start - toleranceWindow and time < stop + toleranceWindow:
                candidates.append([i, max(start - time, time - stop)])

        if len(candidates):
            return min(candidates, key=lambda x: x[1])[0]

    @staticmethod
    def jsonDeserialize(jsonObject):
        """
        json decoding
        TODO: duplicated code
        """
        signal = SparseSegmentSignal([], [])
        for attribute, value in jsonObject.items():
            if hasattr(signal, attribute):
                if type(value) == list:
                    setattr(signal, attribute, np.array(value))
                else:
                    setattr(signal, attribute, value)
        return signal
